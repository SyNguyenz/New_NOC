"""
Does the sparsemax decoder HURT faint-minor survival vs softmax? (entropy-collapse concern: sparsemax
zeros low-score keys -> a faint minor's peak competing with tall majors gets EXACTLY zero attention.)

Three reads on FULL real test, per NOC oracle EM + faintest-minor recall + the faint query's attention
mass on ITS OWN peaks:
  (1) inc2_2d_sparse  native sparsemax decoder
  (2) inc2_2d_sparse  same weights, sparsemax -> softmax SWAPPED at inference (isolates the normalizer)
  (3) inc2_2b_pe_s3   trained-softmax decoder (fair deployable comparison)
"""
import json, types, math
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from models.set_transformer import SetTransformerMixture, SparseMAB

DATA=Path("data_insilico_w"); DEV="cuda" if torch.cuda.is_available() else "cpu"
tok=np.load(DATA/"tokens8_test.npy").astype(np.float32); msk=np.load(DATA/"mask_test.npy")
y=np.load(DATA/"y_test_set.npy"); noc=np.clip(np.load(DATA/"noc_test.npy").astype(int),1,5)
phi=np.load(DATA/"phi_test.npy").astype(np.float32); attr=np.load(DATA/"attr_test.npy"); C=y.shape[1]

def build(rd):
    cfg=json.load(open(Path(rd)/"metrics.json"))["config"]
    m=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
        dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","per_donor"),decoder_source=cfg.get("decoder_source","encoded"),
        n_token_feats=cfg.get("n_token_feats",8),encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),
        num_embed=cfg.get("num_embed","raw"),n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),
        periodic_sigma=cfg.get("periodic_sigma",1.0),aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
        attn_sink=int(cfg.get("attn_sink",0) or 0),donor_recon=cfg.get("donor_recon",False)).to(DEV)
    sd=torch.load(Path(rd)/"best_model.pt",map_location=DEV,weights_only=True); m.load_state_dict(sd,strict=False); m.eval()
    return m,cfg

def patch(m, use_softmax):
    """force decoder SparseMAB to use softmax (True) or native sparsemax (False); capture attn."""
    from models.set_transformer import sparsemax
    def fwd(self,X,Y,key_padding_mask=None):
        B,Nq,_=X.shape; Nk=Y.size(1)
        q=self.q(X).view(B,Nq,self.h,self.dh).transpose(1,2); k=self.k(Y).view(B,Nk,self.h,self.dh).transpose(1,2)
        v=self.v(Y).view(B,Nk,self.h,self.dh).transpose(1,2)
        s=(q@k.transpose(-2,-1))/math.sqrt(self.dh)
        if key_padding_mask is not None: s=s.masked_fill(key_padding_mask[:,None,None,:],-1e4)
        a=torch.softmax(s,-1) if use_softmax else sparsemax(s,-1)
        self._last_attn=a.detach()
        o=(self.drop(a)@v).transpose(1,2).reshape(B,Nq,-1); H=self.norm1(X+self.o(o)); return self.norm2(H+self.ff(H))
    for l in m.cls_decoder_module.layers:
        if isinstance(l,SparseMAB): l.forward=types.MethodType(fwd,l)

@torch.no_grad()
def evalmodel(m, capture):
    P=[]; selfmass=[]; nonzero=[]
    for s in range(0,len(tok),128):
        T=torch.from_numpy(tok[s:s+128]).to(DEV); M=torch.from_numpy(msk[s:s+128]).to(DEV)
        o=m(T,M); P.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
        if capture and getattr(m.cls_decoder_module.layers[-1],"_last_attn",None) is not None:
            a=m.cls_decoder_module.layers[-1]._last_attn.mean(1).cpu().numpy()  # (b,45,Nk) avg heads
            for j in range(len(T)):
                pres=np.where(y[s+j]==1)[0]
                if len(pres)==0: selfmass.append(np.nan); nonzero.append(np.nan); continue
                fmin=pres[int(np.argmin(phi[s+j][pres]))]
                own=np.where((attr[s+j]==fmin))[0]; own=own[own<a.shape[2]]
                if len(own)==0: selfmass.append(np.nan); nonzero.append(np.nan); continue
                row=a[j,fmin]; selfmass.append(float(row[own].sum())); nonzero.append(float((row[own]>1e-6).mean()))
    return np.concatenate(P), np.array(selfmass), np.array(nonzero)

def oracle_and_recall(P):
    orc={}; rec={}
    for k in range(1,6):
        sel=np.where(noc==k)[0]; e=0; r=0
        for j in sel:
            top=set(np.argsort(P[j])[::-1][:k]); ts=set(np.where(y[j]==1)[0])
            pres=list(ts); fmin=pres[int(np.argmin(phi[j][pres]))]
            e+=(top==ts); r+=(fmin in top)
        orc[k]=e/len(sel); rec[k]=r/len(sel)
    return orc,rec

print(f"test n={len(tok)}\n")
configs=[("inc2_2d_sparse SPARSEMAX(native)","results/inc2_2d_sparse_seed42","sparse"),
         ("inc2_2d_sparse SOFTMAX(swap)","results/inc2_2d_sparse_seed42","softmax"),
         ("inc2_2b_pe_s3 SOFTMAX(trained)","results/inc2_2b_pe_s3_seed42","native")]
for name,rd,mode in configs:
    m,cfg=build(rd)
    cap=cfg.get("sparse_attn",False)
    if mode=="sparse": patch(m,False)
    elif mode=="softmax": patch(m,True)
    P,sm,nz=evalmodel(m,cap)
    orc,rec=oracle_and_recall(P)
    print(f"== {name} ==")
    print(f"   oracle EM   N3 {orc[3]:.3f}  N4 {orc[4]:.3f}  N5 {orc[5]:.3f}")
    print(f"   faint recall N3 {rec[3]:.3f}  N4 {rec[4]:.3f}  N5 {rec[5]:.3f}")
    if cap and not np.all(np.isnan(sm)):
        print(f"   faint query mass on OWN peaks {np.nanmean(sm):.3f} | frac own peaks with >0 attn {np.nanmean(nz):.3f}")
    print()

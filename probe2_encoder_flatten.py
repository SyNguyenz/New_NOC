"""
FEASIBILITY (no-train) of the papers' direction: flatten the ENCODER's softmax attention (less major
concentration = QK-norm/sigmaReparam/entropy-control spirit) and ask whether the faint-minor isolation
improves WITHOUT breaking the majors. Temperature TEMP on the ISAB attention logits (TEMP>1 = flatter).

Fair to the OOD trap: the clean per-peak probe is REFIT on the tempered-H for each TEMP (so the readout
matches the representation). TEMP=1.0 reproduces the native model (manual MHA == nn.MultiheadAttention) = sanity.

Per TEMP, on real N5: rank-4 (faintest) ->self  +  rank-0 (major) ->self (must stay high)  +  decoder N5 oracle.
  rank4 up & rank0 kept & oracle not collapsing => flattening helps minor => QK-norm/STKIM direction FEASIBLE.
  only degrades                                  => no-train can't show it (train-time method) / not the knob.
"""
import json, types, math, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from models.set_transformer import SetTransformerMixture, MABpp

DATA=Path("data_insilico_w"); RUN="results/inc2_2d_sparse_seed42"; DEV="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
cfg=json.load(open(Path(RUN)/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
def build():
    m=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
        dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","per_donor"),decoder_source=cfg.get("decoder_source","encoded"),
        n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
        n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
        aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
        attn_sink=int(cfg.get("attn_sink",0) or 0),donor_recon=cfg.get("donor_recon",False)).to(DEV)
    sd=torch.load(Path(RUN)/"best_model.pt",map_location=DEV,weights_only=True); m.load_state_dict(sd,strict=False); m.eval()
    return m
model=build()

TEMP=[1.0]   # set per-run
def manual_mha(attn,Q,K,V,kpm,temp):
    d=attn.embed_dim; h=attn.num_heads; dh=d//h; W=attn.in_proj_weight; b=attn.in_proj_bias
    q=F.linear(Q,W[:d],b[:d]); k=F.linear(K,W[d:2*d],b[d:2*d]); v=F.linear(V,W[2*d:],b[2*d:])
    B,Nq,_=q.shape; Nk=k.shape[1]
    q=q.view(B,Nq,h,dh).transpose(1,2); k=k.view(B,Nk,h,dh).transpose(1,2); v=v.view(B,Nk,h,dh).transpose(1,2)
    s=(q@k.transpose(-2,-1))/(math.sqrt(dh)*temp)
    if kpm is not None: s=s.masked_fill(kpm[:,None,None,:],-1e4)
    return attn.out_proj((torch.softmax(s,-1)@v).transpose(1,2).reshape(B,Nq,d))
def patch_encoder(temp):
    def fwd(self,X,Y,q_mask=None,kv_mask=None):
        Xn=self.norm_q(X,q_mask) if self.norm_q is not None else X
        a=manual_mha(self.attn,Xn,self.norm_kv(Y,kv_mask),Y,kv_mask,temp)
        H=X+a; return H+self.ff(self.norm_h(H,q_mask))
    for isab in model.encoder:
        for mab in [isab.mab0,isab.mab1]:
            if isinstance(mab,MABpp): mab.forward=types.MethodType(fwd,mab)

def load(s): return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
                     np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
@torch.no_grad()
def enc(tk,mk,idx,bs=128):
    out={}
    for s in range(0,len(idx),bs):
        sel=idx[s:s+bs]; _,H,_=model._encode_set(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        H=H.cpu().numpy()
        for j,gi in enumerate(sel): out[int(gi)]=H[j]
    return out
@torch.no_grad()
def dec_scores(tk,mk,idx,bs=256):
    P=np.zeros((len(idx),45),np.float32)
    for s in range(0,len(idx),bs):
        sel=idx[s:s+bs]; o=model(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        P[s:s+len(sel)]=torch.sigmoid(o["logits_cls"]).cpu().numpy()
    return P

tk_tr,mk_tr,at_tr=load("train"); tk,mk,at=load("test")
yT=np.load(DATA/"y_test_set.npy"); nocT=np.clip(np.load(DATA/"noc_test.npy").astype(int),1,5)
fit_idx=np.random.default_rng(0).choice(len(at_tr),6000,replace=False)
# test N5 ranks
ranks={}
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0 or nocT[gi]!=5: continue
    lh=tk[gi][:,2]; info={int(d):float(np.exp(lh[a==d]).sum()) for d in np.unique(a[v])}
    if len(info)!=5: continue
    order=sorted(info,key=lambda d:-info[d]); ranks[gi]={d:r for r,d in enumerate(order)}
keepN5=np.array(sorted(ranks))

for temp in [1.0,1.3,1.7,2.5]:
    patch_encoder(temp)
    # refit clean probe on tempered train H
    Hm=enc(tk_tr,mk_tr,fit_idx); HH=[];DD=[]
    for gi,H in Hm.items():
        a=at_tr[gi]; v=np.where(a>=0)[0]; HH.append(H[v]); DD.append(a[v])
    Xt=torch.from_numpy(np.concatenate(HH).astype(np.float32)).to(DEV); yt=torch.from_numpy(np.concatenate(DD).astype(int)).long().to(DEV)
    clf=nn.Linear(Xt.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
    for ep in range(50):
        perm=torch.randperm(len(yt),device=DEV)
        for s in range(0,len(yt),8192):
            b=perm[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
    Wp=clf.weight.detach().cpu().numpy(); Bp=clf.bias.detach().cpu().numpy()
    # rank4 / rank0 ->self on test N5 (probe restricted to the 5 contributors)
    Hte=enc(tk,mk,keepN5)
    r4=[0,0]; r0=[0,0]
    for gi in keepN5:
        a=at[gi]; v=np.where(a>=0)[0]; contribs=list(ranks[gi]); z=Hte[gi][v]@Wp.T+Bp
        for tgt,acc in [(4,r4),(0,r0)]:
            d=[c for c in ranks[gi] if ranks[gi][c]==tgt][0]
            pk=[i for i,j in enumerate(v) if int(a[j])==d]
            for i in pk:
                pred=contribs[int(np.argmax([z[i,c] for c in contribs]))]; acc[0]+=(pred==d); acc[1]+=1
    P=dec_scores(tk,mk,keepN5); e=0
    for i,gi in enumerate(keepN5):
        top=set(np.argsort(P[i])[::-1][:5]); e+=(top==set(np.where(yT[gi]==1)[0]))
    print(f"TEMP={temp:>4}: rank4->self {r4[0]/r4[1]:.3f}  rank0->self {r0[0]/r0[1]:.3f}  N5 oracle {e/len(keepN5):.3f}")

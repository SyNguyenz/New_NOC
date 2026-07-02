"""
NO-TRAIN CEILING probe of the NON-COMPETITIVE ATTENTION direction for the ~.20 N5 encoder wall.

Mechanism (F35, 3-seed): softmax attention in the ENCODER forces mass onto the dominant donor's peaks, so
a faint minor's evidence is absorbed into majors (probe_encoder_isolate: missed-tail private peaks ->major
72-88%). The literature fix-family = NON-COMPETITIVE attention (sigmoid gates, no sum-to-1 competition;
arXiv 2502.00281). Earlier global temperature-flatten (probe2) FAILED — but maybe WRONG PLACEMENT.

Here: reimplement MABpp's attention from the TRAINED weights and swap the score normalization, PER STEP:
  mab0 = inducing points attend over PEAKS  (softmax over peaks -> tall majors dominate each inducing summary)
  mab1 = peaks attend over inducing points  (the attend-back that builds each peak's contextual H)
Forms: softmax(base) | sigmoid_norm (non-competitive, magnitude-safe) | temp2/temp4 (flatter softmax).

For each condition: refit a CLEAN per-peak donor probe on that condition's TRAIN-H (the representation ceiling),
then on N5 faint (rank4) minors measure BOTH, split by whether the REAL model already recalls the minor:
  (a) per-peak PRIVATE isolation  ->minor   (the encoder-level target)
  (b) donor SOFT-VOTE recall      top-5     (the donor-level outcome — must move too, else per-peak/donor decouple)

Read: a condition that lifts (a) AND (b) on the MISSED set without wrecking the KEPT set => non-competitive
attention targets the mechanism (a train-time bet worth taking), and tells us WHICH step (mab0 vs mab1).
"""
import sys, json, math, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, ".")
import models.set_transformer as st
from models.set_transformer import SetTransformerMixture, MABpp
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc2_2d_sparse_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
geno=load_raw_genotypes()
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]

# ---- non-competitive attention reimplementation of nn.MultiheadAttention (eval, no dropout) ----
def custom_mha(attn, q_in, k_in, v_in, kpm, mode, temp):
    B,Lq,d=q_in.shape; Lk=k_in.shape[1]; h=attn.num_heads; dh=d//h
    Wq,Wk,Wv=attn.in_proj_weight.chunk(3,0); bq,bk,bv=attn.in_proj_bias.chunk(3,0)
    q=(q_in@Wq.T+bq).view(B,Lq,h,dh).transpose(1,2)
    k=(k_in@Wk.T+bk).view(B,Lk,h,dh).transpose(1,2)
    v=(v_in@Wv.T+bv).view(B,Lk,h,dh).transpose(1,2)
    scores=(q@k.transpose(-2,-1))/math.sqrt(dh)                     # (B,h,Lq,Lk)
    if kpm is not None: scores=scores.masked_fill(kpm[:,None,None,:], float('-inf'))
    if mode=="softmax":   A=torch.softmax(scores,-1)
    elif mode=="temp":    A=torch.softmax(scores/temp,-1)
    elif mode=="sigmoid_norm":
        S=torch.sigmoid(scores); A=S/(S.sum(-1,keepdim=True)+1e-6) # non-competitive gates, renormalized (magnitude-safe)
    elif mode=="sigmoid":
        A=torch.sigmoid(scores)                                    # raw gates (magnitude OOD)
    A=torch.nan_to_num(A,0.0)
    out=(A@v).transpose(1,2).reshape(B,Lq,d)
    return attn.out_proj(out)

def patched_forward(self, X, Y, q_mask=None, kv_mask=None):
    mode=getattr(self,"_ncmode","softmax"); temp=getattr(self,"_nctemp",1.0)
    if mode=="softmax":                                            # exact original path
        Xn=self.norm_q(X,q_mask) if self.norm_q is not None else X
        a,_=self.attn(Xn,self.norm_kv(Y,kv_mask),Y,key_padding_mask=kv_mask)
    else:
        Xn=self.norm_q(X,q_mask) if self.norm_q is not None else X
        a=custom_mha(self.attn,Xn,self.norm_kv(Y,kv_mask),Y,kv_mask,mode,temp)
    H=X+a
    return H+self.ff(self.norm_h(H,q_mask))
MABpp.forward=patched_forward     # monkeypatch (reads per-module _ncmode/_nctemp)

def build():
    m=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
        dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","per_donor"),decoder_source=cfg.get("decoder_source","encoded"),
        n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
        n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
        aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
        attn_sink=int(cfg.get("attn_sink",0) or 0),donor_recon=cfg.get("donor_recon",False)).to(DEV)
    m.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); m.eval()
    return m
model=build()
def set_mode(mab0,mab1,t0=1.0,t1=1.0):
    for isab in model.encoder:
        isab.mab0._ncmode,isab.mab0._nctemp=mab0,t0
        isab.mab1._ncmode,isab.mab1._nctemp=mab1,t1

def load(s): return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
                     np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")

@torch.no_grad()
def encode(tk,mk,idx,bs=64):
    out={}
    for s in range(0,len(idx),bs):
        sel=idx[s:s+bs]; _,H,_=model._encode_set(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        for j,gi in enumerate(sel): out[int(gi)]=H[j].cpu().numpy()
    return out
@torch.no_grad()
def decoder_top5(tk,mk,idx,bs=128):
    out={}
    for s in range(0,len(idx),bs):
        sel=idx[s:s+bs]; o=model(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        P=torch.sigmoid(o["logits_cls"]).cpu().numpy()
        for j,gi in enumerate(sel): out[int(gi)]=set(np.argsort(P[j])[::-1][:5])
    return out

tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0); fit_idx=rng.choice(len(at_tr),4000,replace=False)
tk,mk,at=load("test")
def setup(gi):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: return None
    lh=tk[gi][:,2]; info={int(d):{"h":float(np.exp(lh[a==d]).sum())} for d in np.unique(a[v])}
    order=sorted(info,key=lambda d:-info[d]["h"])
    for r,d in enumerate(order): info[d]["rank"]=r
    return (info,v) if len(order)==5 else None
def private_of(gi,d,info):
    others=[KNOWN[o] for o in info if o!=d]; gX=geno.get(KNOWN[d],{}); pr=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: pr.add((L,a))
    return pr
n5=[g for g in range(len(at)) if setup(g) is not None]

# fixed MISSED/KEPT split from the REAL (softmax) decoder
set_mode("softmax","softmax")
real_top5=decoder_top5(tk,mk,np.array(n5))
faint_rec={}
for g in n5:
    info,v=setup(g); d=[c for c in info if info[c]["rank"]==4][0]; faint_rec[g]=(d, d in real_top5[g])
missed=[g for g in n5 if not faint_rec[g][1]]; kept=[g for g in n5 if faint_rec[g][1]]
print(f"N5 {len(n5)}: real decoder KEEPS faint minor in {len(kept)}, MISSES {len(missed)}\n")

def fit_probe():
    Hm=encode(tk_tr,mk_tr,fit_idx); HH=[];DD=[]
    for gi,H in Hm.items():
        a=at_tr[gi]; vv=np.where(a>=0)[0]; HH.append(H[vv]); DD.append(a[vv])
    Xt=torch.from_numpy(np.concatenate(HH).astype(np.float32)).to(DEV); yt=torch.from_numpy(np.concatenate(DD).astype(int)).long().to(DEV)
    clf=nn.Linear(Xt.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
    for ep in range(45):
        perm=torch.randperm(len(yt),device=DEV)
        for s in range(0,len(yt),8192):
            b=perm[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
    return clf.weight.detach().cpu().numpy(), clf.bias.detach().cpu().numpy()

def measure(W,B,gs):
    iso_hit=iso_tot=rec_hit=0
    for g in gs:
        info,v=setup(g); contribs=list(info.keys()); d=faint_rec[g][0]
        H=Hte[g]; Z=H[v]@W.T+B
        # (a) per-peak isolation of the faint minor's private peaks
        priv=private_of(g,d,info)
        pk=[i for i,j in enumerate(v) if int(at[g][j])==d and (int(tk[g][j,0]),akey(tk[g][j,1])) in priv]
        for i in pk:
            row=Z[i]; pred=contribs[int(np.argmax([row[c] for c in contribs]))]; iso_hit+=(pred==d); iso_tot+=1
        # (b) donor soft-vote recall (top-5 over the 45)
        sm=np.exp(Z-Z.max(1,keepdims=True)); sm/=sm.sum(1,keepdims=True)
        vote=sm.sum(0); rec_hit+=(d in set(np.argsort(vote)[::-1][:5]))
    return iso_hit/max(iso_tot,1), rec_hit/max(len(gs),1)

CONDS=[("BASE softmax",        "softmax","softmax",1,1),
       ("mab0 sigmoid_norm",   "sigmoid_norm","softmax",1,1),
       ("mab0 temp4",          "temp","softmax",4,1),
       ("mab1 sigmoid_norm",   "softmax","sigmoid_norm",1,1),
       ("both sigmoid_norm",   "sigmoid_norm","sigmoid_norm",1,1),
       ("both temp4",          "temp","temp",4,4)]
print(f"{'condition':22s} | KEPT iso  rec  | MISSED iso  rec")
print("-"*64)
for name,m0,m1,t0,t1 in CONDS:
    set_mode(m0,m1,t0,t1)
    W,B=fit_probe()
    Hte=encode(tk,mk,np.array(n5))
    ki,kr=measure(W,B,kept); mi,mr=measure(W,B,missed)
    print(f"{name:22s} |  {ki:.3f}  {kr:.3f} |  {mi:.3f}  {mr:.3f}")
print("\nMISSED iso&rec UP without KEPT collapse => non-competitive attn targets the mechanism (which step).")
print("MISSED iso UP but rec flat => per-peak/donor decouple again. all flat => direction/placement wrong.")

"""Decisive: is there LEARNABLE true-vs-decoy signal beyond the model + beyond crude rules?
For each sample, take the model's top-K donors. Per candidate donor c, compute RAW features CONDITIONED on
the model's top-5 set (deployable; no ground-truth peak filtering):
  private_support = |present(c) - union present(top5 \ {c})|   (peaks ONLY c explains given the set)
  damning_absence, completeness, rarity_sum, mean_height_pct, present_count, model_prob.
Label = c is a TRUE contributor (1) else 0. Train a small MLP on TRAIN+VAL rows; on TEST, RE-RANK each
sample's top-K by predicted membership prob, take top-5 -> N5 oracle.
  beats 0.788  => learnable refinement headroom exists (build a learned set-refinement head).
  ~0.788/below => signal at the representation floor -> fix must be in ENCODER training, not a head."""
import json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"; DEV=torch.device("cuda")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy", allow_pickle=True)

g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
geno=[set() for _ in range(45)]
for c in range(45):
    for j in range(g.shape[1]):
        if gm[c,j]: geno[c].add((int(g[c,j,0]), round(float(g[c,j,1]),1)))
panel=Counter()
for c in range(45):
    for k in geno[c]: panel[k]+=1
psz=[max(len(geno[c]),1) for c in range(45)]

cfg=json.load(open(ROOT/"results/inc6_maskp_seed42/metrics.json"))["config"]
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
  dropout=0.1,cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",
  periodic_sigma=cfg["periodic_sigma"],aux_heads=True,sparse_attn=True).to(DEV)
m.load_state_dict(torch.load(ROOT/"results/inc6_maskp_seed42/best_model.pt",weights_only=True,map_location=DEV)); m.eval()
_n=L("tokens8_train")[:,:,1:8][L("mask_train").astype(bool)]
m.feat_mean.copy_(torch.tensor(_n.mean(0),device=DEV)); m.feat_std.copy_(torch.tensor(_n.std(0)+1e-6,device=DEV))

def preds(tok,mk):
    P=np.zeros((len(tok),45))
    with torch.no_grad():
        for s in range(0,len(tok),128):
            x=torch.from_numpy(tok[s:s+128]).to(DEV); mb=torch.from_numpy(mk[s:s+128]).to(DEV)
            P[s:s+128]=torch.sigmoid(m(x,mb)["logits_cls"]).cpu().numpy()
    return P

K=8
def build_rows(split, cap=None):
    tok=L(f"tokens8_{split}").astype(np.float32); mk=L(f"mask_{split}").astype(bool); y=L(f"y_{split}_set").astype(np.float32)
    P=preds(tok,mk); idx=np.arange(len(tok))
    if cap and len(idx)>cap: idx=np.random.RandomState(0).choice(idx,cap,replace=False)
    X=[]; Y=[]; meta=[]
    for i in idx:
        loc={}
        for j in np.where(mk[i])[0]:
            l=int(tok[i,j,0]); al=round(float(tok[i,j,1]),1); h=float(np.expm1(tok[i,j,2]))
            loc.setdefault(l,{}); loc[l][al]=max(loc[l].get(al,0.),h)
        hs=sorted(h for ld in loc.values() for h in ld.values());
        pct=lambda h:(np.searchsorted(hs,h)/max(len(hs),1))
        top=list(np.argsort(P[i])[::-1][:K]); top5=set(int(c) for c in top[:5])
        presc={c:{(l,al) for (l,al) in geno[c] if l in loc and al in loc[l]} for c in top}
        for c in top:
            c=int(c); others=top5-{c}
            uo=set().union(*[presc[o] for o in others]) if others else set()
            priv=presc[c]-uo
            dam=sum(1 for (l,al) in geno[c] if l in loc and al not in loc[l])
            rar=sum(1.0/panel[(l,al)] for (l,al) in presc[c])
            mh=np.mean([pct(loc[l][al]) for (l,al) in presc[c]]) if presc[c] else 0.0
            X.append([len(priv),dam,len(presc[c])/psz[c],rar,mh,len(presc[c]),P[i][c]])
            Y.append(1.0 if y[i][c]>0.5 else 0.0); meta.append((int(i),c))
    return np.array(X,np.float32), np.array(Y,np.float32), meta, P, y, L(f"noc_{split}").astype(int)

print("building rows (train subset / val / test) ...", flush=True)
Xtr,Ytr,_,_,_,_=build_rows("train",cap=15000)
Xv,Yv,_,_,_,_=build_rows("val")
Xte,Yte,mte,Pte,yte,noc=build_rows("test")
Xtr=np.concatenate([Xtr,Xv]); Ytr=np.concatenate([Ytr,Yv])
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
def nz(X): return (X-mu)/sd
clf=nn.Sequential(nn.Linear(7,32),nn.ReLU(),nn.Linear(32,1)).to(DEV)
opt=torch.optim.Adam(clf.parameters(),lr=2e-3,weight_decay=1e-4)
xt=torch.tensor(nz(Xtr)).to(DEV); yt=torch.tensor(Ytr).to(DEV)
pos=yt.mean(); w=torch.where(yt>0.5, 1/pos, 1/(1-pos))
for ep in range(300):
    opt.zero_grad(); lg=clf(xt).squeeze(-1)
    loss=(nn.functional.binary_cross_entropy_with_logits(lg,yt,reduction='none')*w).mean()
    loss.backward(); opt.step()
with torch.no_grad():
    sc=clf(torch.tensor(nz(Xte)).to(DEV)).squeeze(-1).cpu().numpy()
# re-rank test top-K by classifier score
from collections import defaultdict
bys=defaultdict(list)
for r,(i,c) in enumerate(mte): bys[i].append((c,sc[r]))
def n5_of(scoremap_fn):
    ii=np.where(noc==5)[0]; e=[]
    for i in ii:
        cand=bys[i]; ranked=sorted(cand,key=scoremap_fn(i),reverse=True); top5=set(c for c,_ in ranked[:5])
        e.append(top5==set(int(x) for x in np.where(yte[i]>0.5)[0]))
    return float(np.mean(e))
n5_clf=n5_of(lambda i: (lambda cv: cv[1]))
n5_model=n5_of(lambda i: (lambda cv: Pte[i][cv[0]]))
# AUC of classifier vs model-prob for true-vs-decoy, restricted to top-K rows
from numpy import argsort
def auc(s,y):
    o=argsort(s); r=np.empty_like(o,dtype=float); r[o]=np.arange(len(s))
    pos=y>0.5; npos=pos.sum(); nneg=(~pos).sum()
    return (r[pos].sum()-npos*(npos-1)/2)/(npos*nneg)
print(f"\ntrue-vs-candidate AUC on top-{K} rows:  model_prob={auc(Xte[:,6],Yte):.3f} | learned_clf={auc(sc,Yte):.3f}")
print(f"N5 oracle (re-rank top-{K}):  model={n5_model:.3f}  | learned refine={n5_clf:.3f}  | delta={n5_clf-n5_model:+.3f}")
print("verdict: delta>0 => learnable refinement headroom; delta<=0 => floor is in the encoder/representation.")

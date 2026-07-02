"""Re-check the session's 'encoder wall / raw signal exhausted' conclusions on a model trained with the
REALISTIC generator (genprop_real), since the original probes used inc6_maskp = ORIGINAL over-skewed
generator (minors 2.4x too faint). Were the conclusions an artifact of the bad generator?
Reports, on the REAL test, comparably to before:
  (sanity) per-NOC oracle  (genprop_real N5 should ~0.815)
  (1) model_prob true-vs-decoy AUC within its own top-8  (the '0.998 exhausted' number)
  (2) per-N5-miss: missed-true PRIVATE alleles present, decoy DAMNING absence, model RANK of missed-true
  (3) learned classifier (raw conditional feats + model_prob), trained on REAL VAL, re-rank TEST N5 -> delta
Run: MODEL_DIR=results/genprop_real_seed42 python recheck_decoy.py   (default genprop_real)."""
import os, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import Counter, defaultdict
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"; DEV=torch.device("cuda")
MODEL_DIR=ROOT/os.environ.get("MODEL_DIR","results/genprop_real_seed42")
OVERWRITE_FEATS=int(os.environ.get("OVERWRITE_FEATS","0"))   # 1 = use data_insilico_w train stats (for inc6_maskp)
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

cfg=json.load(open(MODEL_DIR/"metrics.json"))["config"]
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
  dropout=0.1,cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",
  periodic_sigma=cfg["periodic_sigma"],aux_heads=True,sparse_attn=True).to(DEV)
sd=torch.load(MODEL_DIR/"best_model.pt",weights_only=True,map_location=DEV)
miss,unexp=m.load_state_dict(sd,strict=False); m.eval()
print(f"MODEL_DIR={MODEL_DIR.name} | missing={len(miss)} unexpected={len(unexp)} | feat_mean[:3]={m.feat_mean[:3].tolist()}")
if OVERWRITE_FEATS:
    _n=L("tokens8_train")[:,:,1:8][L("mask_train").astype(bool)]
    m.feat_mean.copy_(torch.tensor(_n.mean(0),device=DEV)); m.feat_std.copy_(torch.tensor(_n.std(0)+1e-6,device=DEV))

def preds(split):
    tok=L(f"tokens8_{split}").astype(np.float32); mk=L(f"mask_{split}").astype(bool)
    P=np.zeros((len(tok),45))
    with torch.no_grad():
        for s in range(0,len(tok),128):
            x=torch.from_numpy(tok[s:s+128]).to(DEV); mb=torch.from_numpy(mk[s:s+128]).to(DEV)
            P[s:s+128]=torch.sigmoid(m(x,mb)["logits_cls"]).cpu().numpy()
    return tok,mk,P

te_tok,te_mk,Pte=preds("test"); te_y=L("y_test_set").astype(np.float32); te_noc=L("noc_test").astype(int)
# sanity per-NOC oracle
print("sanity per-NOC oracle:", end=" ")
for k in range(1,6):
    ii=np.where(te_noc==k)[0]; e=[(np.argsort(Pte[i])[::-1][:k], te_y[i]) for i in ii]
    acc=np.mean([ (lambda pr: (pr==te_y[i]).all())(np.isin(np.arange(45),t).astype(float)) for i,(t,_) in zip(ii,e)])
    print(f"N{k}={acc:.3f}",end=" ")
print()

def mix_of(tok,mk,i):
    loc={}
    for j in np.where(mk[i])[0]:
        l=int(tok[i,j,0]); al=round(float(tok[i,j,1]),1); h=float(np.expm1(tok[i,j,2]))
        loc.setdefault(l,{}); loc[l][al]=max(loc[l].get(al,0.),h)
    return loc
def present(c,loc): return {(l,al) for (l,al) in geno[c] if l in loc and al in loc[l]}

# ---- per-N5-miss diagnostics + model_prob AUC ----
ii=np.where(te_noc==5)[0]; Tpriv=[]; Drank=[]; Ddam=[]; auc_rows=[]
for i in ii:
    true=set(int(x) for x in np.where(te_y[i]>0.5)[0]); top=list(np.argsort(Pte[i])[::-1]); pred=set(int(x) for x in top[:5])
    loc=mix_of(te_tok,te_mk,i)
    for c in top[:8]:                                  # AUC rows: candidate in top-8, label = true member
        auc_rows.append((Pte[i][c], 1.0 if c in true else 0.0))
    if pred==true: continue
    expl_t=set().union(*[geno[c] for c in true])
    for t in (true-pred):
        others=true-{t}; eo=set().union(*[geno[c] for c in others]) if others else set()
        Tpriv.append(len(present(t,loc)-eo)); Drank.append(top.index(t))
    for d in (pred-true):
        Ddam.append(sum(1 for (l,al) in geno[d] if l in loc and al not in loc[l]))
ar=np.array(auc_rows);
def auc(s,y):
    o=np.argsort(s); r=np.empty_like(o,dtype=float); r[o]=np.arange(len(s)); pos=y>0.5
    return (r[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*(~pos).sum())
print(f"\n(1) model_prob true-vs-decoy AUC (top-8 rows): {auc(ar[:,0],ar[:,1]):.3f}   [inc6_maskp was 0.998]")
print(f"(2) N5 misses={len(Drank)} | missed-T private-present mean={np.mean(Tpriv):.2f} (==0: {100*np.mean(np.array(Tpriv)==0):.0f}%) | model rank of missed-T median={int(np.median(Drank))} | decoy damning-absence mean={np.mean(Ddam):.1f}")
print(f"    [inc6_maskp: private 4.25 / rank median 9 / damning 13.0]")

# ---- (3) learned classifier on REAL VAL -> re-rank TEST N5 ----
def rows(split):
    tok,mk,P=preds(split); y=L(f"y_{split}_set").astype(np.float32); noc=L(f"noc_{split}").astype(int)
    X=[];Y=[];meta=[]
    for i in range(len(tok)):
        loc=mix_of(tok,mk,i); hs=sorted(h for ld in loc.values() for h in ld.values())
        pct=lambda h:(np.searchsorted(hs,h)/max(len(hs),1))
        top=list(np.argsort(P[i])[::-1][:8]); top5=set(int(c) for c in top[:5])
        pr={c:present(c,loc) for c in top}
        for c in top:
            c=int(c); oth=top5-{c}; uo=set().union(*[pr[o] for o in oth]) if oth else set()
            priv=pr[c]-uo; dam=sum(1 for (l,al) in geno[c] if l in loc and al not in loc[l])
            rar=sum(1.0/panel[k] for k in pr[c]); mh=np.mean([pct(loc[l][al]) for (l,al) in pr[c]]) if pr[c] else 0.0
            X.append([len(priv),dam,len(pr[c])/psz[c],rar,mh,len(pr[c]),P[i][c]]); Y.append(1.0 if y[i][c]>0.5 else 0.0); meta.append((i,c,P[i][c]))
    return np.array(X,np.float32),np.array(Y,np.float32),meta,P,y,noc
Xv,Yv,_,_,_,_=rows("val"); Xt,Yt,mt,Pt,yt,noct=rows("test")
mu=Xv.mean(0); sdv=Xv.std(0)+1e-6
clf=nn.Sequential(nn.Linear(7,32),nn.ReLU(),nn.Linear(32,1)).to(DEV)
opt=torch.optim.Adam(clf.parameters(),lr=2e-3,weight_decay=1e-4)
xt=torch.tensor((Xv-mu)/sdv).to(DEV); yv=torch.tensor(Yv).to(DEV); w=torch.where(yv>0.5,1/yv.mean(),1/(1-yv.mean()))
for _ in range(300):
    opt.zero_grad(); lg=clf(xt).squeeze(-1); (nn.functional.binary_cross_entropy_with_logits(lg,yv,reduction='none')*w).mean().backward(); opt.step()
with torch.no_grad(): sc=clf(torch.tensor((Xt-mu)/sdv).to(DEV)).squeeze(-1).cpu().numpy()
bys=defaultdict(list)
for r,(i,c,p) in enumerate(mt): bys[i].append((c,sc[r],p))
def n5(keyf):
    iis=np.where(noct==5)[0]; e=[]
    for i in iis:
        rk=sorted(bys[i],key=keyf,reverse=True); e.append(set(c for c,_,_ in rk[:5])==set(int(x) for x in np.where(yt[i]>0.5)[0]))
    return float(np.mean(e))
nm=n5(lambda t:t[2]); nl=n5(lambda t:t[1])
print(f"(3) N5 re-rank: model={nm:.3f}  learned-clf={nl:.3f}  delta={nl-nm:+.3f}")

# ---- (4) CLEAN deployable: partition test N5 by whether its combo appears in VAL (leak) ----
valy=L("y_val_set").astype(np.float32); valnoc=L("noc_val").astype(int)
val_combos=set(frozenset(int(x) for x in np.where(valy[i]>0.5)[0]) for i in np.where(valnoc==5)[0])
iis=np.where(noct==5)[0]
def combo(i): return frozenset(int(x) for x in np.where(yt[i]>0.5)[0])
def n5_sub(idxs,keyf):
    e=[ (sorted(bys[i],key=keyf,reverse=True)) for i in idxs ]
    return float(np.mean([set(c for c,_,_ in rk[:5])==combo(i) for i,rk in zip(idxs,e)])) if idxs else float('nan')
print(f"    test N5 combos total={len(set(combo(i) for i in iis))} | of which in VAL={len(set(combo(i) for i in iis)&val_combos)}")
for tag,idxs in [("ALL",list(iis)),
                 ("val-OVERLAP(leak)",[i for i in iis if combo(i) in val_combos]),
                 ("val-DISJOINT(clean)",[i for i in iis if combo(i) not in val_combos])]:
    mm=n5_sub(idxs,lambda t:t[2]); ll=n5_sub(idxs,lambda t:t[1])
    print(f"      {tag:22s} n={len(idxs):3d}: model={mm:.3f} learned={ll:.3f} delta={ll-mm:+.3f}")

"""Trainable CEILING of the discrete allele-accounting direction.
Signal: given a context set of donors, a candidate d's UNIQUE contribution = its OBSERVED alleles that the
context CANNOT explain, rarity-weighted (rare private alleles = decisive). A true 5th donor has positive
unique contribution; a decoy whose alleles are all explained-away by the context has ~0.
  ORACLE context (the other 4 TRUE donors)  -> the CEILING: can perfect allele-accounting pick the 5th?
  DEPLOYABLE context (model's top-4)         -> realistic.
Report: (a) AUC separating true-missed vs decoy on hard pairs; (b) "given 4 correct, pick the 5th" accuracy.
"""
import json, numpy as np, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
geno=[set() for _ in range(45)]; from collections import Counter; panel=Counter()
for c in range(45):
    for j in range(g.shape[1]):
        if gm[c,j]: geno[c].add((int(g[c,j,0]),round(float(g[c,j,1]),1)))
for c in range(45):
    for k in geno[c]: panel[k]+=1
tok=L("tokens8_test").astype(np.float32); mk=L("mask_test").astype(bool); y=L("y_test_set").astype(np.float32); noc=L("noc_test").astype(int); B=len(tok)
cfg=json.load(open(ROOT/"results/inc6_maskp_seed42/metrics.json"))["config"]
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
  dropout=0.1,cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",
  periodic_sigma=cfg["periodic_sigma"],aux_heads=True,sparse_attn=True).to(DEV)
m.load_state_dict(torch.load(ROOT/"results/inc6_maskp_seed42/best_model.pt",weights_only=True,map_location=DEV)); m.eval()
P=np.zeros((B,45))
with torch.no_grad():
    for s in range(0,B,256):
        tk=torch.from_numpy(tok[s:s+256]).to(DEV); mb=torch.from_numpy(mk[s:s+256]).to(DEV)
        P[s:s+256]=torch.sigmoid(m(tk,mb)["logits_cls"]).cpu().numpy()
def obs(i): return set((int(tok[i,j,0]),round(float(tok[i,j,1]),1)) for j in np.where(mk[i])[0])
def unique(d, ok, ctx):
    """rarity-weighted OBSERVED alleles of d NOT explained by the context set ctx."""
    expl=set().union(*[geno[c] for c in ctx]) if ctx else set()
    return sum(1.0/panel[a] for a in geno[d] if a in ok and a not in expl)

# (a) hard-pair AUC: true-missed (ctx=other 4 true) vs decoy (ctx=all 5 true)
yl=[]; orc=[]; dep=[]
for i in np.where(noc==5)[0]:
    ok=obs(i); tru=set(np.where(y[i]>0.5)[0].tolist()); top=set(np.argsort(P[i])[::-1][:5].tolist())
    if top==tru: continue
    top4=list(np.argsort(P[i])[::-1][:4])                       # deployable context
    for d in tru-top:
        yl.append(1); orc.append(unique(d,ok,tru-{d})); dep.append(unique(d,ok,[c for c in top4 if c!=d]))
    for d in top-tru:
        yl.append(0); orc.append(unique(d,ok,tru)); dep.append(unique(d,ok,top4))
print(f"hard-pair true-vs-decoy separation (n={len(yl)}):")
print(f"  ORACLE-context unique-contribution AUC   = {roc_auc_score(yl,orc):.3f}   <- CEILING")
print(f"  DEPLOYABLE (model top-4) unique AUC       = {roc_auc_score(yl,dep):.3f}")
print(f"  decoder AUC (same pairs)                  = {roc_auc_score(yl,[ -1  for _ in yl] if False else [orc[k]*0 - 0 for k in range(len(yl))]):.3f}" if False else "")

# (b) "given 4 correct, pick the 5th" accuracy (oracle 4 = the 4 true donors w/ highest decoder score)
def pick5(i, ctxmode):
    ok=obs(i); tru=np.where(y[i]>0.5)[0]
    true_by_score=sorted(tru,key=lambda d:-P[i,d]); ctx4=list(true_by_score[:4]); true5=true_by_score[4]
    cand=[d for d in range(45) if d not in ctx4]
    if ctxmode=="oracle": sc=[unique(d,ok,ctx4) for d in cand]
    elif ctxmode=="decoder": sc=[P[i,d] for d in cand]
    return cand[int(np.argmax(sc))]==true5
idx5=np.where(noc==5)[0]
acc_dec=np.mean([pick5(i,"decoder") for i in idx5])
acc_orc=np.mean([pick5(i,"oracle") for i in idx5])
print(f"\n'given the 4 strongest true donors, pick the 5th' accuracy (N5, n={len(idx5)}):")
print(f"  decoder picks 5th                = {acc_dec:.3f}")
print(f"  ORACLE allele-accounting picks 5th = {acc_orc:.3f}   <- CEILING of the direction")

"""Headroom of the PROCESS lever, on RAW data only (no neural net): can analysis-by-synthesis solve N5?
Greedy set selection = the 'process' done by hand on raw alleles:
  residual = present mix peaks not yet explained by the chosen set.
  score(d) = |present(d) & residual|  (NEW peaks d explains)  - lam * damning_absence(d)
  pick argmax, mark its present peaks explained, repeat 5x -> a 5-set. Compare to truth.
If greedy-raw recovers the true N5 set far above the model's 0.788, the info IS in the raw signal and the
CONDITIONAL/residual process is exactly what the model fails to do. Sweep lam. Also report recovery on the
79 model-MISS samples specifically. Deployable: mix peaks + panel genotypes only."""
import json, numpy as np, torch
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
geno_loci=[{l for (l,_) in geno[c]} for c in range(45)]

te_tok=L("tokens8_test").astype(np.float32); te_mk=L("mask_test").astype(bool); te_y=L("y_test_set").astype(np.float32); te_noc=L("noc_test").astype(int)
# model preds (to know which are the 79 misses)
cfg=json.load(open(ROOT/"results/inc6_maskp_seed42/metrics.json"))["config"]
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
  dropout=0.1,cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",
  periodic_sigma=cfg["periodic_sigma"],aux_heads=True,sparse_attn=True).to(DEV)
m.load_state_dict(torch.load(ROOT/"results/inc6_maskp_seed42/best_model.pt",weights_only=True,map_location=DEV)); m.eval()
_n=L("tokens8_train")[:,:,1:8][L("mask_train").astype(bool)]
m.feat_mean.copy_(torch.tensor(_n.mean(0),device=DEV)); m.feat_std.copy_(torch.tensor(_n.std(0)+1e-6,device=DEV))
Pm=np.zeros((len(te_tok),45))
with torch.no_grad():
    for s in range(0,len(te_tok),128):
        x=torch.from_numpy(te_tok[s:s+128]).to(DEV); mb=torch.from_numpy(te_mk[s:s+128]).to(DEV)
        Pm[s:s+128]=torch.sigmoid(m(x,mb)["logits_cls"]).cpu().numpy()

def mix_of(i):
    loc={}
    for j in np.where(te_mk[i])[0]:
        l=int(te_tok[i,j,0]); al=round(float(te_tok[i,j,1]),1)
        loc.setdefault(l,set()).add(al)
    return loc

def greedy(loc, lam):
    presentset={(l,al) for l,als in loc.items() for al in als}
    pres=[presentset & geno[c] for c in range(45)]
    dam=[sum(1 for (l,al) in geno[c] if l in loc and al not in loc[l]) for c in range(45)]
    chosen=[]; explained=set()
    for _ in range(5):
        best=-1; bi=-1
        for c in range(45):
            if c in chosen: continue
            new=len(pres[c]-explained)
            sc=new - lam*dam[c]
            if sc>best: best=sc; bi=c
        chosen.append(bi); explained|=pres[bi]
    return set(chosen)

ii=np.where(te_noc==5)[0]
true=[set(int(x) for x in np.where(te_y[i]>0.5)[0]) for i in ii]
predm=[set(int(x) for x in np.argsort(Pm[i])[::-1][:5]) for i in ii]
miss=[k for k in range(len(ii)) if predm[k]!=true[k]]
mloc=[mix_of(i) for i in ii]
print(f"N5: {len(ii)} samples | model misses {len(miss)} | model N5 oracle={1-len(miss)/len(ii):.3f}\n")
print("RAW greedy 'process' (residual coverage - lam*damning):")
for lam in [0.0,0.05,0.1,0.2,0.3,0.5]:
    gsets=[greedy(mloc[k],lam) for k in range(len(ii))]
    alln5=np.mean([gsets[k]==true[k] for k in range(len(ii))])
    recov=np.mean([gsets[k]==true[k] for k in miss]) if miss else 0
    print(f"  lam={lam:<4}: greedy-raw N5 oracle={alln5:.3f} | recovers {recov*100:.0f}% of the model's 79 misses")
# best-lam complementarity: union potential

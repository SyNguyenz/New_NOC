"""Is the lever a CONDITIONAL refinement SEEDED by the model (not naive greedy)?
Start from the model's top-5 (keep its combinatorial strength), then hill-climb on a RAW set-score by
swapping at most a couple donors, restricted to the model's top-K shortlist (so we only fix the marginal
decoy<->true-donor decision, not re-search 45):
  set-score A(S) = coverage(S) - lam*sum_damning(S)
     coverage(S) = |union of present alleles explained by S|   (rewards donors that explain NEW peaks)
     damning(d)  = d's panel alleles ABSENT at an observed locus (penalizes free-riders)
  swap out d in S, in e in shortlist, if it raises A. Repeat. -> refined 5-set.
Report N5 oracle after refinement vs model 0.788, and how many of the 79 misses it recovers.
Also: swap-distance of misses, and how many decoys are pure free-riders (marginal coverage 0 within the set).
Deployable: mix peaks + panel + model probs only (no ground truth)."""
import json, numpy as np, torch
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"; DEV=torch.device("cuda")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy", allow_pickle=True)

g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
geno=[set() for _ in range(45)]
for c in range(45):
    for j in range(g.shape[1]):
        if gm[c,j]: geno[c].add((int(g[c,j,0]), round(float(g[c,j,1]),1)))

te_tok=L("tokens8_test").astype(np.float32); te_mk=L("mask_test").astype(bool); te_y=L("y_test_set").astype(np.float32); te_noc=L("noc_test").astype(int)
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
        l=int(te_tok[i,j,0]); al=round(float(te_tok[i,j,1]),1); loc.setdefault(l,set()).add(al)
    return loc

ii=np.where(te_noc==5)[0]
true=[set(int(x) for x in np.where(te_y[i]>0.5)[0]) for i in ii]
predm=[set(int(x) for x in np.argsort(Pm[i])[::-1][:5]) for i in ii]
miss=[k for k in range(len(ii)) if predm[k]!=true[k]]
swapd=[len(predm[k]-true[k]) for k in miss]
print(f"N5 {len(ii)} | model misses {len(miss)} (oracle 0.788) | swap-distance of misses: 1={swapd.count(1)} 2={swapd.count(2)} 3+={sum(1 for s in swapd if s>=3)}\n")

def refine(i, K, lam, rounds=3):
    loc=mix_of(i); presentset={(l,al) for l,als in loc.items() for al in als}
    pres={c:(presentset & geno[c]) for c in range(45)}
    dam={c:sum(1 for (l,al) in geno[c] if l in loc and al not in loc[l]) for c in range(45)}
    order=list(np.argsort(Pm[i])[::-1]); short=set(int(c) for c in order[:K])
    S=set(int(c) for c in order[:5])
    def cover(s):
        u=set()
        for c in s: u|=pres[c]
        return len(u)
    def A(s): return cover(s) - lam*sum(dam[c] for c in s)
    for _ in range(rounds):
        cur=A(S); bestd=0; best=None
        for d in list(S):
            for e in short-S:
                s2=(S-{d})|{e}; delta=A(s2)-cur
                if delta>bestd: bestd=delta; best=(d,e)
        if best is None: break
        S=(S-{best[0]})|{best[1]}
    return S

print("model-seeded CONDITIONAL refine (coverage - lam*damning), shortlist=top-K:")
for K in [8,12,20]:
    for lam in [0.1,0.2,0.3]:
        rs=[refine(ii[k],K,lam) for k in range(len(ii))]
        n5=np.mean([rs[k]==true[k] for k in range(len(ii))])
        rec=np.mean([rs[k]==true[k] for k in miss])
        brk=np.mean([rs[k]!=true[k] and predm[k]==true[k] for k in range(len(ii))])  # newly broken (were right)
        print(f"  K={K:<3} lam={lam}: refined N5={n5:.3f} (model 0.788) | recovers {rec*100:.0f}% of 79 misses | breaks {brk*100:.1f}% of previously-correct")

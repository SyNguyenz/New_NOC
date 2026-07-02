"""
Which token feature(s) drive the synthetic-train -> real-test gap that strands N5?

Token8 features (cols): 0 locus | 1 allele | 2 log_h | 3 Hb | 4 SR(back-stutter) |
                        5 rank_inv | 6 n_at_locus/10 | 7 glob_rel
Model standardizes cols 1..7 with feat_mean/feat_std (computed on SYNTH train).

(A) per-feature distribution shift synth-train vs real-test, in the model's STANDARDIZED
    space, for MINOR-donor peaks (the ones that get missed) and all peaks.
(B) domain classifier (synth vs real) on per-peak features -> permutation importance =
    which feature the two domains differ on most.
"""
import json, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"; CKPT=ROOT/"results"/"inc6_maskp_seed42"
import torch
def L(n): return np.load(DATA/f"{n}.npy", allow_pickle=True)
NAMES=["allele","log_h","Hb","SR","rank_inv","n/10","glob_rel"]   # cols 1..7

sd=torch.load(CKPT/"best_model.pt",weights_only=True,map_location="cpu")
fmean=sd["feat_mean"].numpy(); fstd=sd["feat_std"].numpy()
print("feat_mean(synth-train):", fmean.round(3))
print("feat_std (synth-train):", fstd.round(3))

def minor_peaks(split, minor_only=True):
    tk=L(f"tokens8_{split}").astype(np.float32); mk=L(f"mask_{split}").astype(bool)
    at=L(f"attr_{split}").astype(int); phi=L(f"phi_{split}").astype(np.float32); noc=L(f"noc_{split}").astype(int)
    rows=[]
    n5=np.where(noc==5)[0]
    for i in n5:
        for c in range(45):
            if not (0<phi[i,c]<(0.2 if minor_only else 1.01)): continue
            pe=(at[i]==c)&mk[i]
            if pe.sum(): rows.append(tk[i][pe][:,1:8])   # cols 1..7
    return np.concatenate(rows) if rows else np.zeros((0,7))

S=minor_peaks("train"); R=minor_peaks("test")
print(f"\nN5 MINOR-donor peaks: synth-train={len(S)}  real-test={len(R)}")
Sz=(S-fmean)/fstd; Rz=(R-fmean)/fstd            # standardized as the model sees them
print("\n(A) per-feature STANDARDIZED distribution shift (minor peaks)")
print(f"  {'feature':<9} {'synth_mu':>9} {'real_mu':>9} {'d_mu(std)':>10} {'synth_sd':>9} {'real_sd':>9}   raw quantile shift p10/p50/p90")
shifts=[]
for k,nm in enumerate(NAMES):
    dmu=Rz[:,k].mean()-Sz[:,k].mean()
    qs=[np.percentile(R[:,k],q)-np.percentile(S[:,k],q) for q in (10,50,90)]
    shifts.append((abs(dmu),nm,dmu))
    print(f"  {nm:<9} {Sz[:,k].mean():>9.3f} {Rz[:,k].mean():>9.3f} {dmu:>10.3f} "
          f"{Sz[:,k].std():>9.3f} {Rz[:,k].std():>9.3f}   {qs[0]:+.2f}/{qs[1]:+.2f}/{qs[2]:+.2f}")
print("\n  ranked by |d_mu| (std units):")
for a,nm,d in sorted(shifts,reverse=True): print(f"    {nm:<9} {d:+.3f}")

# (B) domain classifier on minor peaks (standardized)
X=np.vstack([Sz,Rz]); yv=np.r_[np.zeros(len(Sz)),np.ones(len(Rz))]
perm=np.random.RandomState(0).permutation(len(X)); X,yv=X[perm],yv[perm]
ntr=int(0.7*len(X))
rf=RandomForestClassifier(n_estimators=200,max_depth=8,random_state=0,n_jobs=-1).fit(X[:ntr],yv[:ntr])
from sklearn.metrics import roc_auc_score
auc=roc_auc_score(yv[ntr:],rf.predict_proba(X[ntr:])[:,1])
print(f"\n(B) domain classifier synth-vs-real on MINOR peaks: AUC={auc:.3f} "
      f"(0.5=indistinguishable, high=big domain gap)")
imp=rf.feature_importances_
print("  feature importance (which feature separates synth vs real):")
for k in np.argsort(imp)[::-1]:
    print(f"    {NAMES[k]:<9} {imp[k]:.3f}")

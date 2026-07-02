"""
DECODER exact mechanism: is the memorized margin a TRAINING CO-OCCURRENCE PRIOR?
Hypothesis: on a novel combo the decoder substitutes the true faint minor with a donor that
FREQUENTLY CO-OCCURRED with the present majors in training (a memorized "buddy"), instead of
reading the faint minor's own evidence.

Test (eval-only, saved test preds + train labels):
  - Co[i,j] = #train mixtures containing both donors i,j.
  - For each N5 test sample: majors = top-3 by height; true minors = 2 faintest; FP = predicted-present
    but not true; missed = true but not predicted.
  - Compare mean train co-occurrence with the 3 majors for: FP substitutes vs true MISSED minors vs random.
If Co(FP, majors) >> Co(missed-true-minor, majors) ~ Co(random, majors): decoder picks training buddies
of the majors = co-occurrence memorization. That is the exact decoder sub-mechanism.
"""
import numpy as np
from pathlib import Path
DATA=Path("data_insilico_w"); RUN=Path("results/inc7_masspool_seed42")

ytr=np.load(DATA/"y_train_set.npy").astype(bool); noctr=np.load(DATA/"noc_train.npy")
attr=np.load(DATA/"attr_test.npy"); tok=np.load(DATA/"tokens8_test.npy"); LH=tok[:,:,2]; noc=np.load(DATA/"noc_test.npy")
yp=np.load(RUN/"y_test_pred.npy").astype(bool); yt=np.load(RUN/"y_test_true.npy").astype(bool)

# training co-occurrence over multi-person mixtures
M=ytr[noctr>=2]                       # (Nmix,45)
Co=M.T.astype(np.int64)@M.astype(np.int64)      # (45,45) pair counts
np.fill_diagonal(Co,0)
freq=M.sum(0)                          # marginal donor frequency in train

def cooc_with(maj_set, d):
    return np.mean([Co[d,mj] for mj in maj_set])

fp_co=[]; miss_co=[]; rnd_co=[]; fp_freq=[]; miss_freq=[]
for i in np.where(noc==5)[0]:
    a=attr[i]; valid=np.where(a>=0)[0]
    dh={int(d):float(np.exp(LH[i][a==d]).sum()) for d in np.unique(a[valid])}
    order=sorted(dh,key=lambda d:-dh[d]); majors=order[:3]; minors=order[3:]
    true=set(np.where(yt[i])[0]); pred=set(np.where(yp[i])[0])
    fps=pred-true; missed=true-pred
    for d in fps:
        fp_co.append(cooc_with(majors,d)); fp_freq.append(freq[d])
    for d in missed:
        miss_co.append(cooc_with(majors,d)); miss_freq.append(freq[d])
        # random control: 50 random non-present donors, their co-occ with these majors
        cand=[x for x in range(45) if x not in true]
        rnd=np.random.default_rng(d).choice(cand,size=min(50,len(cand)),replace=False)
        rnd_co.append(np.mean([cooc_with(majors,x) for x in rnd]))

fp_co=np.array(fp_co); miss_co=np.array(miss_co); rnd_co=np.array(rnd_co)
print(f"N5 substitutions analyzed: FP donors n={len(fp_co)} | missed-true-minors n={len(miss_co)}")
print("\n== Mean TRAIN co-occurrence with the sample's 3 MAJOR donors ==")
print(f"  FP substitute  (decoder's wrong pick) : {fp_co.mean():.2f}   median {np.median(fp_co):.1f}")
print(f"  TRUE missed minor (right answer)      : {miss_co.mean():.2f}   median {np.median(miss_co):.1f}")
print(f"  RANDOM absent donor (chance baseline) : {rnd_co.mean():.2f}   median {np.median(rnd_co):.1f}")
print(f"\n  FP / true-minor co-occ ratio = {fp_co.mean()/max(miss_co.mean(),1e-9):.2f}x   FP / random = {fp_co.mean()/max(rnd_co.mean(),1e-9):.2f}x")
print(f"  marginal train freq: FP picks {np.mean(fp_freq):.0f} vs missed-true {np.mean(miss_freq):.0f}  (is the decoder just picking high-frequency donors?)")
# within-pair test: in each sample, is FP more major-co-occurring than the true minor it replaced?
pair=[]
for i in np.where(noc==5)[0]:
    a=attr[i]; valid=np.where(a>=0)[0]
    dh={int(d):float(np.exp(LH[i][a==d]).sum()) for d in np.unique(a[valid])}
    order=sorted(dh,key=lambda d:-dh[d]); majors=order[:3]
    true=set(np.where(yt[i])[0]); pred=set(np.where(yp[i])[0])
    fps=list(pred-true); missed=list(true-pred)
    if fps and missed:
        pair.append(np.mean([cooc_with(majors,f) for f in fps]) - np.mean([cooc_with(majors,m) for m in missed]))
pair=np.array(pair)
print(f"\n  within-sample (FP co-occ - missed-true co-occ): mean {pair.mean():+.2f}, % samples FP>true = {(pair>0).mean()*100:.0f}%  (n={len(pair)})")
print("  >0 and high % => decoder substitutes a major's TRAINING BUDDY for the true faint minor = co-occurrence memorization.")

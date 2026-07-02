"""
stutter_test.py — Test the hypothesis (user): NOC counting is limited by IMPOVERISHED
features (no stutter handling), NOT physics. deepNoC counts NOC4/5 >0.9 with 89 feats/peak;
our mac_feats counts alleles above RFU thresholds WITHOUT removing back-stutter -> tall
parents make stutter peaks exceed threshold -> over-count. Add stutter-aware + intra-locus
balance features; compare two-stage stage2 count accuracy on in-silico dev (no retrain).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, xgboost as xgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_set_transformer import mac_feats
D = ROOT / "data_insilico_w"; C = ROOT / "forensim_io" / "_diag_cache"
THR = [50, 100, 150, 250]
STUTTER_MAX = 0.18                                # back-stutter typically <=~15%


def stutter_feats(tokens, mask):
    """Per-sample: stutter-FILTERED allele counts + intra-locus balance, at RFU thresholds.
    A peak at allele a is flagged back-stutter if a parent peak exists at a+1 with
    height(a) <= STUTTER_MAX*height(a+1)."""
    h = np.expm1(tokens[:, :, 2]); out = []
    for i in range(len(tokens)):
        v0 = mask[i]; loci = tokens[i, :, 0].astype(int); al = tokens[i, :, 1]; hi = h[i]
        row = []
        for thr in THR:
            ntrue = np.zeros(24); bal = []
            for L in np.unique(loci[v0]):
                m = v0 & (loci == L) & (hi > thr)
                a = al[m]; hh = hi[m]
                if len(a) == 0:
                    continue
                # flag back-stutter: peak a with parent a+1 present and h(a)<=0.18*h(a+1)
                keep = np.ones(len(a), bool)
                for j in range(len(a)):
                    par = np.where(np.abs(a - (a[j] + 1.0)) < 1e-3)[0]
                    if len(par) and hh[j] <= STUTTER_MAX * hh[par[0]]:
                        keep[j] = False
                nt = int(keep.sum()); ntrue[L] = nt
                if nt >= 2:
                    hk = hh[keep]; bal.append(hk.min() / (hk.max() + 1e-9))   # het balance
            row += [ntrue.max(), int((ntrue >= 2).sum()), int((ntrue >= 3).sum()),
                    int((ntrue >= 4).sum()), float(np.mean(bal) if bal else 0.0),
                    float(np.std(bal) if bal else 0.0)]
        out.append(row)
    return np.array(out, dtype=np.float32)


def main():
    noc = np.clip(np.load(D / "noc_train.npy").astype(int), 1, 5)
    tok = np.load(D / "tokens_train.npy"); mk = np.load(D / "mask_train.npy")
    # cached prob-profile+MAC base; we add stutter feats
    base = np.load(C / "feat_train.npz")["base"]
    print("computing current MAC vs stutter-aware features ...")
    mac = mac_feats(tok, mk)                       # current (no stutter filter)
    stu = stutter_feats(tok, mk)                   # stutter-filtered + balance

    rng = np.random.default_rng(0)
    multi = np.where(noc >= 2)[0]; rng.shuffle(multi)
    nd = len(multi) // 5; dev, fit = multi[:nd], multi[nd:]

    def s2(X, reg={"max_depth": 4, "min_child_weight": 10}):
        kw = dict(n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  eval_metric="mlogloss", random_state=42); kw.update(reg)
        m = xgb.XGBClassifier(**kw); m.fit(X[fit], noc[fit] - 2); return m

    feats = {
        "MAC only (current)": mac,
        "stutter-aware only": stu,
        "MAC + stutter": np.hstack([mac, stu]),
        "base(prob+MAC) [ref]": base,
        "base + stutter": np.hstack([base, stu]),
    }
    print(f"\n  {'count features':<24}{'N2':>6}{'N3':>6}{'N4':>6}{'N5':>6}{'multi-acc':>10}")
    for tag, X in feats.items():
        m = s2(X); p = m.predict(X[dev]) + 2; t = noc[dev]
        accs = [(p[t == k] == k).mean() for k in [2, 3, 4, 5]]
        print(f"  {tag:<24}" + "".join(f"{a:>6.2f}" for a in accs) + f"{(p==t).mean():>10.4f}")
    print(f"\n  dev n per NOC: {dict((k,int((noc[dev]==k).sum())) for k in [2,3,4,5])}")


if __name__ == "__main__":
    main()

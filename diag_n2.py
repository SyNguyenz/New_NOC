"""
diag_n2.py — Why does adding pgNOC features drop NOC2? Test the user's hypotheses:
  (A) feature conflict / XGB overfit  — does regularization / where-errors-go reveal it?
  (B) pgNOC sub-optimal at NOC2       — is the pgNOC feature itself misleading on NOC2,
                                        or does it have an in-silico->real domain gap?
Decomposes the two-stage into stage1 (NOC1-vs-multi) + stage2 (count 2-5) so we can
see WHICH stage loses NOC2 and whether it is fixable (not intrinsic redistribution).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, torch, xgboost as xgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
from train_set_transformer import card_features
import pgnoc

D = ROOT / "data_insilico_w"; DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = ROOT / "forensim_io" / "_diag_cache"; CACHE.mkdir(exist_ok=True)


def load_model():
    m = SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2,
        m_inducing=32, n_classes=45, n_noc=6, dropout=0.1, cls_decoder="hybrid",
        n_flat=590, decouple_reject=True).to(DEV)
    m.load_state_dict(torch.load(ROOT / "results/hybrid_50k_weight/best_model.pt", weights_only=True))
    m.eval(); return m


@torch.no_grad()
def probs(m, s):
    t = np.load(D / f"tokens_{s}.npy"); mk = np.load(D / f"mask_{s}.npy")
    xf = np.load(D / f"Xflat_{s}.npy").astype(np.float32); P = []
    for i in range(0, len(t), 512):
        P.append(torch.sigmoid(m(torch.from_numpy(t[i:i+512]).to(DEV), torch.from_numpy(mk[i:i+512]).to(DEV),
            torch.from_numpy(xf[i:i+512]).to(DEV))["logits_cls"]).cpu().numpy())
    return np.concatenate(P)


def build():
    m = load_model(); G = pgnoc.build_refs()
    feats = {}
    for s in ["train", "val", "test"]:
        fp = CACHE / f"feat_{s}.npz"
        if fp.exists():
            z = np.load(fp); feats[s] = (z["base"], z["pg"]); continue
        P = probs(m, s)
        base = card_features(P, np.load(D / f"tokens_{s}.npy"), np.load(D / f"mask_{s}.npy"))
        H = np.expm1(np.load(D / f"Xflat_{s}.npy").astype(np.float64))
        C = np.zeros((len(P), pgnoc.KMAX))
        for i in range(len(P)):
            _, C[i] = pgnoc.estimate_noc(H[i], P[i], G)
        pg = np.hstack([C, C[:, :-1] - C[:, 1:]]).astype(np.float32)
        np.savez(fp, base=base, pg=pg); feats[s] = (base, pg)
    return feats


def stages(Xtr, ntr, Xva, nva, Xte, reg=None):
    """explicit two-stage; returns test k + stage1 multi-call + stage2 count."""
    kw = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, eval_metric="mlogloss", random_state=42)
    if reg: kw.update(reg)
    s1 = xgb.XGBClassifier(**{**kw, "eval_metric": "logloss"})
    s1.fit(Xva, (np.clip(nva, 1, 5) >= 2).astype(int))
    multi = s1.predict(Xte).astype(bool)
    mb = ntr >= 2
    s2 = xgb.XGBClassifier(**kw); s2.fit(Xtr[mb], np.clip(ntr[mb], 1, 5) - 2)
    k = np.ones(len(Xte), int)
    if multi.any(): k[multi] = s2.predict(Xte[multi]) + 2
    return k, multi


def main():
    t0 = time.time(); feats = build()
    noc = {s: np.load(D / f"noc_{s}.npy").astype(int) for s in ["train", "val", "test"]}
    nte = noc["test"]; print(f"features ready {time.time()-t0:.0f}s")
    base = {s: feats[s][0] for s in feats}
    comb = {s: np.hstack(feats[s]) for s in feats}
    pgonly = {s: feats[s][1] for s in feats}

    def report(tag, Xd, reg=None):
        k, multi = stages(Xd["train"], noc["train"], Xd["val"], noc["val"], Xd["test"], reg)
        n2 = nte == 2
        acc2 = (k[n2] == 2).mean()
        # where do true-NOC2 go? + stage1 multi-call rate on NOC2
        dist = {int(v): int(c) for v, c in zip(*np.unique(k[n2], return_counts=True))}
        s1mul = multi[n2].mean()
        accs = [(nte == j).sum() and (k[nte == j] == j).mean() for j in range(1, 6)]
        print(f"  {tag:<34} N2={acc2:.3f} (pred {dist}, stage1->multi {s1mul:.2f}) | "
              f"N3={accs[2]:.2f} N4={accs[3]:.2f} N5={accs[4]:.2f} all={(k==nte).mean():.3f}")
        return k

    print("\n--- baseline vs +pgNOC (default XGB) ---")
    report("base prob+MAC", base)
    report("+pgNOC feats", comb)
    print("\n--- HYP A: regularize combined (prevent overfit to in-silico) ---")
    for reg in [{"max_depth": 3}, {"max_depth": 3, "reg_lambda": 5.0},
                {"max_depth": 4, "subsample": 0.6, "colsample_bytree": 0.5},
                {"n_estimators": 150, "max_depth": 3, "min_child_weight": 10}]:
        report(f"+pgNOC {reg}", comb, reg)
    print("\n--- HYP B: pgNOC-only features (is its own NOC2 signal bad?) ---")
    report("pgNOC-only", pgonly)
    print("\n--- domain gap probe: pgNOC NOC2 cost-curve, in-silico TRAIN vs real TEST ---")
    for s in ["train", "test"]:
        m2 = noc[s] == 2
        if m2.sum():
            C = feats[s][1][m2][:, :pgnoc.KMAX]
            print(f"  {s:<6} NOC2 mean cost[k=1..5]: {np.round(C[:, :5].mean(0), 3)}  (n={m2.sum()})")


if __name__ == "__main__":
    main()

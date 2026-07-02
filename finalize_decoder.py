"""
finalize_decoder.py — Pick the count-decoder config WITHOUT touching test.
Selection rule (theory: regularization controls overfit; choice made on real VAL):
  for (featureset in {base prob+MAC, +pgNOC}) x (stage2 regularization grid):
     fit stage2 on in-silico TRAIN-multi, score COUNT ACCURACY on real VAL-multi
     (val-multi is NOT used to fit stage2 -> clean selection signal).
  pick the config with best val-multi count accuracy; THEN evaluate test ONCE.
Reuses cached features from diag_n2.py (_diag_cache).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, xgboost as xgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_set_transformer import topk_decode, per_noc_em
import pgnoc

D = ROOT / "data_insilico_w"; CACHE = ROOT / "forensim_io" / "_diag_cache"


def feats(s):
    z = np.load(CACHE / f"feat_{s}.npz"); return z["base"], np.hstack([z["base"], z["pg"]])


def stage2_only(Xtr, ntr, Xte, reg):
    kw = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, eval_metric="mlogloss", random_state=42); kw.update(reg)
    mb = ntr >= 2
    s2 = xgb.XGBClassifier(**kw); s2.fit(Xtr[mb], np.clip(ntr[mb], 1, 5) - 2)
    return s2


def full(Xtr, ntr, Xva, nva, Xte, reg):
    s1 = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", random_state=42)
    s1.fit(Xva, (np.clip(nva, 1, 5) >= 2).astype(int))
    s2 = stage2_only(Xtr, ntr, Xte, reg)
    k = np.ones(len(Xte), int); multi = s1.predict(Xte).astype(bool)
    if multi.any(): k[multi] = s2.predict(Xte[multi]) + 2
    return k


def main():
    noc = {s: np.load(D / f"noc_{s}.npy").astype(int) for s in ["train", "val", "test"]}
    P_te = None  # probs needed for EM decode
    import torch
    from models.set_transformer import SetTransformerMixture
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32,
        n_classes=45, n_noc=6, dropout=0.1, cls_decoder="hybrid", n_flat=590, decouple_reject=True).to(dev)
    m.load_state_dict(torch.load(ROOT / "results/hybrid_50k_weight/best_model.pt", weights_only=True)); m.eval()
    t = np.load(D / "tokens_test.npy"); mk = np.load(D / "mask_test.npy"); xf = np.load(D / "Xflat_test.npy").astype(np.float32)
    Pl = []
    with torch.no_grad():
        for i in range(0, len(t), 512):
            Pl.append(torch.sigmoid(m(torch.from_numpy(t[i:i+512]).to(dev), torch.from_numpy(mk[i:i+512]).to(dev),
                torch.from_numpy(xf[i:i+512]).to(dev))["logits_cls"]).cpu().numpy())
    P_te = np.concatenate(Pl)
    y_te = np.load(D / "y_test_set.npy")

    base, comb = {}, {}
    for s in ["train", "val", "test"]:
        base[s], comb[s] = feats(s)

    grid = [{}, {"max_depth": 3}, {"max_depth": 3, "reg_lambda": 5.0},
            {"max_depth": 4, "min_child_weight": 10},
            {"n_estimators": 150, "max_depth": 3, "min_child_weight": 10}]

    # ---- SELECTION on VAL-multi count accuracy (clean: val-multi not used to fit stage2) ----
    vmask = noc["val"] >= 2
    best = None
    print("Selection on VAL-multi count accuracy (NO test):")
    for tag, F in [("base", base), ("+pgNOC", comb)]:
        for reg in grid:
            s2 = stage2_only(F["train"], noc["train"], F["val"], reg)
            pv = s2.predict(F["val"][vmask]) + 2
            acc = (pv == np.clip(noc["val"][vmask], 1, 5)).mean()
            print(f"  {tag:<8} reg={str(reg):<48} val-multi acc={acc:.4f}")
            if best is None or acc > best[0]: best = (acc, tag, F, reg)
    print(f"\nSELECTED: {best[1]} reg={best[3]} (val-multi acc={best[0]:.4f})")

    # ---- EVALUATE on TEST once ----
    def ev(tag, F, reg):
        k = full(F["train"], noc["train"], F["val"], noc["val"], F["test"], reg)
        return per_noc_em(y_te, topk_decode(P_te, k), noc["test"]), float((k == noc["test"]).mean())
    orc = per_noc_em(y_te, topk_decode(P_te, noc["test"]), noc["test"])
    hdr = ["all", "N1", "N2", "N3", "N4", "N5"]
    print(f"\n{'config':<30}" + "".join(f"{h:>7}" for h in hdr) + f"{'cntAcc':>8}")
    print(f"{'oracle':<30}" + "".join(f"{x:>7.3f}" for x in orc) + "     —")
    em, ca = ev("base default", base, {}); print(f"{'base prob+MAC (default)':<30}" + "".join(f"{x:>7.3f}" for x in em) + f"{ca:>8.3f}")
    em, ca = ev(best[1], best[2], best[3]); print(f"{'SELECTED ('+best[1]+', val-tuned reg)':<30}" + "".join(f"{x:>7.3f}" for x in em) + f"{ca:>8.3f}")


if __name__ == "__main__":
    main()

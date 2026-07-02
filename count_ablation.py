"""
count_ablation.py — Ablation for the NOC (cardinality) estimator that decodes
top-k donor predictions. Goal: close per-NOC EM gap to oracle (target +-0.03).

Diagnosis (hybrid_50k_weight): ranking is already at oracle (true-donor prob ~0.9
even at NOC4/5); the ENTIRE decoded-vs-oracle gap is COUNT error, concentrated at
NOC4/5 which the current estimator under-counts (reads only the donor-prob profile,
which saturates at high NOC).

This script ablates, post-hoc (no retrain), two axes:
  FEATURES:  A = prob-profile [top8 probs, sum, #>=0.5]        (current)
             B = MAC/height combo-invariant (train_noc_tabular) (physical count signal)
             A+B = concatenation
  FIT SET:   val = real val (current posthoc_cardinality source)
             bal = balanced in-silico (synth_train NOC2-5 + real train NOC1)

Estimator: XGBoost multiclass on true NOC (1-5). Decode top-k* of donor probs.
Reports per-NOC Exact Match vs oracle for each (features x fitset), both models.

Usage:  python count_ablation.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
# data_insilico_w = combo-diverse in-silico OVERLAY (make_insilico, weight-per-noc 50k),
# the SAME distribution hybrid_50k_weight / st_50k_weight were trained on.
# NOT data/synth_train (that is the failed R simDNAmixtures peak-model synth).
DATA = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data_insilico_w")))
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
DEV = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLDS = [0, 50, 100, 150, 250, 500]


# ── feature builders ─────────────────────────────────────────────────────────
def mac_feats(tok: np.ndarray, msk: np.ndarray) -> np.ndarray:
    """46 combo-invariant MAC/height features (= train_noc_tabular.featurize)."""
    h = np.expm1(tok[:, :, 2]); out = []
    for i in range(len(tok)):
        v0 = msk[i]; loci_all = tok[i, :, 0].astype(int); row = []
        for thr in THRESHOLDS:
            v = v0 & (h[i] > thr); loci = loci_all[v]
            if len(loci):
                c = np.bincount(loci, minlength=24); nz = c[c > 0]
                row += [c.max(), len(loci), int((c >= 2).sum()),
                        int((c >= 3).sum()), int((c >= 4).sum()), float(nz.mean())]
            else:
                row += [0, 0, 0, 0, 0, 0.0]
        hv = h[i][v0]
        if len(hv):
            row += list(np.percentile(hv, [10, 25, 50, 75, 90]))
            row += [hv.mean(), hv.std(), hv.max(), float(np.std(np.log1p(hv)))]
        else:
            row += [0] * 9
        row += [int(np.unique(loci_all[v0]).size)]
        out.append(row)
    return np.array(out, dtype=np.float32)


def prob_feats(P: np.ndarray, top_m: int = 8) -> np.ndarray:
    s = np.sort(P, 1)[:, ::-1][:, :top_m]
    return np.concatenate([s, P.sum(1, keepdims=True), (P >= 0.5).sum(1, keepdims=True)], 1)


def topk_decode(P, k_arr):
    yp = np.zeros_like(P, dtype=int)
    for i in range(len(P)):
        k = int(max(1, min(5, round(k_arr[i])))); yp[i, np.argsort(P[i])[::-1][:k]] = 1
    return yp


def per_noc_em(yt, yp, noc):
    e = np.all(yt == yp, axis=1)
    return [e.mean()] + [e[noc == j].mean() if (noc == j).sum() else float("nan") for j in range(1, 6)]


# ── model probs ───────────────────────────────────────────────────────────────
def build_model(kind):
    if kind == "hybrid":
        return SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2,
            m_inducing=32, n_classes=45, n_noc=6, dropout=0.1, cls_decoder="hybrid",
            n_flat=590, decouple_reject=True).to(DEV)
    return SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2,
        m_inducing=32, n_classes=45, n_noc=6, dropout=0.1).to(DEV)


@torch.no_grad()
def probs(model, tok, msk, xf):
    P = []
    for i in range(0, len(tok), 512):
        b = dict(tokens=torch.from_numpy(tok[i:i+512]).to(DEV),
                 mask=torch.from_numpy(msk[i:i+512]).to(DEV))
        if xf is not None:
            b["xflat"] = torch.from_numpy(xf[i:i+512].astype(np.float32)).to(DEV)
        P.append(torch.sigmoid(model(**b)["logits_cls"]).cpu().numpy())
    return np.concatenate(P)


def load_split(sp):
    return (np.load(DATA/f"tokens_{sp}.npy"), np.load(DATA/f"mask_{sp}.npy"),
            np.load(DATA/f"Xflat_{sp}.npy"), np.load(DATA/f"y_{sp}_set.npy"),
            np.load(DATA/f"noc_{sp}.npy").astype(int))


def run_model(kind, ckpt):
    print(f"\n{'='*78}\n{kind.upper()}  ({ckpt})\n{'='*78}")
    m = build_model(kind); m.load_state_dict(torch.load(ckpt, weights_only=True)); m.eval()
    use_xf = (kind == "hybrid")

    # splits (DATA = data_insilico_w: test/val are REAL, train is in-silico overlay)
    tk_te, mk_te, xf_te, y_te, noc_te = load_split("test")
    tk_va, mk_va, xf_va, y_va, noc_va = load_split("val")
    # balanced fit set = the combo-diverse in-silico train split itself (weight-per-noc)
    tk_b, mk_b, xf_b, _, noc_b = load_split("train")
    noc_b = np.clip(noc_b, 1, 5)

    t0 = time.time()
    P_te, P_va, P_b = (probs(m, tk_te, mk_te, xf_te if use_xf else None),
                       probs(m, tk_va, mk_va, xf_va if use_xf else None),
                       probs(m, tk_b, mk_b, xf_b if use_xf else None))
    print(f"  probs done {time.time()-t0:.0f}s | bal fit n={len(noc_b)} "
          f"{dict(zip(*np.unique(np.clip(noc_b,1,5),return_counts=True)))}")

    # feature blocks
    t0 = time.time()
    MAC_te, MAC_va, MAC_b = mac_feats(tk_te, mk_te), mac_feats(tk_va, mk_va), mac_feats(tk_b, mk_b)
    print(f"  MAC feats done {time.time()-t0:.0f}s")
    PR_te, PR_va, PR_b = prob_feats(P_te), prob_feats(P_va), prob_feats(P_b)

    feat = {  # name -> (train_blocks_by_fitset, test_block)
        "A prob-profile": dict(val=PR_va, bal=PR_b, test=PR_te),
        "B MAC/height":   dict(val=MAC_va, bal=MAC_b, test=MAC_te),
        "A+B combined":   dict(val=np.hstack([PR_va, MAC_va]), bal=np.hstack([PR_b, MAC_b]),
                               test=np.hstack([PR_te, MAC_te])),
    }
    fitnoc = dict(val=np.clip(noc_va, 1, 5), bal=noc_b)

    oracle = per_noc_em(y_te, topk_decode(P_te, noc_te), noc_te)
    rows = [("oracle (true k)", oracle, float("nan"))]
    for fname, blocks in feat.items():
        for fs in ("val", "bal"):
            clf = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42)
            clf.fit(blocks[fs], fitnoc[fs] - 1)
            k = clf.predict(blocks["test"]) + 1
            em = per_noc_em(y_te, topk_decode(P_te, k), noc_te)
            cacc = float((k == noc_te).mean())
            rows.append((f"{fname} | fit={fs}", em, cacc))

    # ── refined decoders that resolve the bal-prior-vs-boundary tension ──────
    Xtr_AB = np.hstack([PR_b, MAC_b]); Xte_AB = np.hstack([PR_te, MAC_te])
    Xva_AB = np.hstack([PR_va, MAC_va])

    # (1) prior-correction: fit on bal (learns high-k boundary) then reweight
    #     posterior by (val_prior / bal_prior) so it matches the REAL test prior.
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42)
    clf.fit(Xtr_AB, noc_b - 1)
    proba = clf.predict_proba(Xte_AB)
    pri_bal = np.bincount(noc_b - 1, minlength=5) / len(noc_b)
    pri_tgt = np.bincount(np.clip(noc_va, 1, 5) - 1, minlength=5) / len(noc_va)
    w = pri_tgt / np.clip(pri_bal, 1e-6, None)
    k_pc = (proba * w).argmax(1) + 1
    rows.append(("A+B bal +prior-corr", per_noc_em(y_te, topk_decode(P_te, k_pc), noc_te),
                 float((k_pc == noc_te).mean())))

    # (2) two-stage: stage1 NOC1-vs-multi (fit on real val = right prior, easy task),
    #     stage2 among multi count 2-5 (fit on bal multi = plenty of high-NOC).
    s1 = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
    s1.fit(Xva_AB, (np.clip(noc_va, 1, 5) >= 2).astype(int))
    is_multi = s1.predict(Xte_AB).astype(bool)
    mb = noc_b >= 2
    s2 = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42)
    s2.fit(Xtr_AB[mb], noc_b[mb] - 2)            # labels 2-5 -> 0-3
    k_ts = np.ones(len(P_te), int)
    if is_multi.any():
        k_ts[is_multi] = s2.predict(Xte_AB[is_multi]) + 2
    rows.append(("two-stage(N1|multi)", per_noc_em(y_te, topk_decode(P_te, k_ts), noc_te),
                 float((k_ts == noc_te).mean())))

    hdr = ["all", "N1", "N2", "N3", "N4", "N5"]
    print(f"\n  {'config':<26}" + "".join(f"{h:>7}" for h in hdr) + f"{'cntAcc':>8}")
    for nm, em, cacc in rows:
        ca = "  —  " if np.isnan(cacc) else f"{cacc:>7.3f}"
        print(f"  {nm:<26}" + "".join(f"{x:>7.3f}" for x in em) + f"{ca:>8}")
    return rows


if __name__ == "__main__":
    run_model("hybrid", ROOT/"results/hybrid_50k_weight/best_model.pt")
    run_model("set_transformer", ROOT/"results/st_50k_weight/best_model.pt")

"""
eval_principled.py — Theory-grounded NOC selection, NO test tuning, NO hand thresholds.

pgNOC fits k=1..6 nested contributor models (gamma-weighted NNLS, reference genotypes).
Model order is chosen by an INFORMATION CRITERION on the (concentrated Gaussian)
log-likelihood of the weighted residual — the canonical model-selection theory that
NOCIt/EuroForMix instantiate (a-posteriori NOC / penalized likelihood):

    RSS_k = cost_k^2 ;  logL_k ∝ -N/2 * ln(RSS_k / N)
    AIC_k = N*ln(RSS_k/N) + 2*p_k ;  BIC_k = N*ln(RSS_k/N) + ln(N)*p_k
    p_k   = (k-1) mixture proportions + 2 (scale, shape) = k+1
    k*    = argmin_k IC_k

No parameter is tuned on test (or val). two-stage XGB is a learned classifier fit on
train/val (legitimate supervised learning); reported alongside for evaluation only.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
from train_set_transformer import card_features, two_stage_cardinality, topk_decode, per_noc_em
import pgnoc

D = ROOT / "data_insilico_w"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BINS = 590                                      # residual dimension (constant across k)


def load_model():
    m = SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2,
        m_inducing=32, n_classes=45, n_noc=6, dropout=0.1, cls_decoder="hybrid",
        n_flat=590, decouple_reject=True).to(DEV)
    m.load_state_dict(torch.load(ROOT / "results/hybrid_50k_weight/best_model.pt", weights_only=True))
    m.eval(); return m


@torch.no_grad()
def get_probs(m, split):
    t = np.load(D / f"tokens_{split}.npy"); mk = np.load(D / f"mask_{split}.npy")
    xf = np.load(D / f"Xflat_{split}.npy").astype(np.float32); P = []
    for i in range(0, len(t), 512):
        P.append(torch.sigmoid(m(torch.from_numpy(t[i:i+512]).to(DEV),
            torch.from_numpy(mk[i:i+512]).to(DEV),
            torch.from_numpy(xf[i:i+512]).to(DEV))["logits_cls"]).cpu().numpy())
    return np.concatenate(P)


def cost_curve(P, split, G):
    H = np.expm1(np.load(D / f"Xflat_{split}.npy").astype(np.float64))
    C = np.zeros((len(P), pgnoc.KMAX))
    for i in range(len(P)):
        _, C[i] = pgnoc.estimate_noc(H[i], P[i], G)
    return C


def ic_select(C, kind="AIC"):
    """k* by information criterion on the concentrated Gaussian log-lik. No tuning."""
    rss = np.clip(C ** 2, 1e-12, None)            # RSS_k
    p = np.arange(1, pgnoc.KMAX + 1) + 1          # params = k+1
    pen = 2.0 if kind == "AIC" else np.log(N_BINS)
    ic = N_BINS * np.log(rss / N_BINS) + pen * p
    return ic.argmin(1) + 1


def main():
    t0 = time.time(); m = load_model(); G = pgnoc.build_refs()
    P = {s: get_probs(m, s) for s in ["train", "val", "test"]}
    noc = {s: np.load(D / f"noc_{s}.npy").astype(int) for s in ["train", "val", "test"]}
    y_te = np.load(D / "y_test_set.npy")
    Cte = cost_curve(P["test"], "test", G)
    print(f"setup {time.time()-t0:.0f}s")

    orc = per_noc_em(y_te, topk_decode(P["test"], noc["test"]), noc["test"])
    # theory-grounded pgNOC selections (NO tuning)
    k_aic = ic_select(Cte, "AIC"); k_bic = ic_select(Cte, "BIC")
    # discriminative two-stage (fit on train/val, eval test) — legitimate supervised baseline
    bf = {s: card_features(P[s], np.load(D / f"tokens_{s}.npy"), np.load(D / f"mask_{s}.npy"))
          for s in ["train", "val", "test"]}
    k_ts = two_stage_cardinality(bf["train"], noc["train"], bf["val"], noc["val"], bf["test"])
    # two-stage + pgNOC cost-curve as FEATURES (supervised; pgNOC features theory-justified)
    print("  computing pgNOC cost curves for train/val (for combined two-stage) ...")
    def pgfeat(s):
        C = cost_curve(P[s], s, G); dr = C[:, :-1] - C[:, 1:]
        kk = ic_select(C, "BIC").reshape(-1, 1)
        return np.hstack([C, dr, kk]).astype(np.float32)
    cf = {s: np.hstack([bf[s], pgfeat(s)]) for s in ["train", "val", "test"]}
    k_cb = two_stage_cardinality(cf["train"], noc["train"], cf["val"], noc["val"], cf["test"])

    rows = [("oracle (true k)", orc, float("nan")),
            ("pgNOC + AIC (theory)", per_noc_em(y_te, topk_decode(P["test"], k_aic), noc["test"]),
             float((k_aic == noc["test"]).mean())),
            ("pgNOC + BIC (theory)", per_noc_em(y_te, topk_decode(P["test"], k_bic), noc["test"]),
             float((k_bic == noc["test"]).mean())),
            ("two-stage prob+MAC", per_noc_em(y_te, topk_decode(P["test"], k_ts), noc["test"]),
             float((k_ts == noc["test"]).mean())),
            ("two-stage +pgNOC feats", per_noc_em(y_te, topk_decode(P["test"], k_cb), noc["test"]),
             float((k_cb == noc["test"]).mean()))]
    hdr = ["all", "N1", "N2", "N3", "N4", "N5"]
    print(f"\n  {'estimator':<24}" + "".join(f"{h:>7}" for h in hdr) + f"{'cntAcc':>8}")
    for nm, em, ca in rows:
        cs = "   —  " if np.isnan(ca) else f"{ca:>7.3f}"
        print(f"  {nm:<24}" + "".join(f"{x:>7.3f}" for x in em) + cs)


if __name__ == "__main__":
    main()

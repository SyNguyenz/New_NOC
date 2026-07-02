"""
pgnoc.py — Mini continuous-model NOC estimator (literature-faithful to NOCIt /
EuroForMix), reference-conditioned. Python so correctness is fully controlled;
cross-checked against real EuroForMix (R) in pgnoc_euroformix.R.

Model (per locus/bin, gamma peak-height + dropout, cf. Bleka 2016 / Swaminathan 2015):
  expected mean  mu_b = T * sum_{d in S} phi_d * G_d[b]      (S = candidate donors)
  observed peak  h_b ~ Gamma(shape=rho, scale=mu_b/rho)       (var = mu^2/rho)
  dropout        if expected but h_b<AT:  + log P(Gamma < AT) (regularized lower inc.)
  drop-in        if observed but mu_b~0:  + lambda_dropin penalty
Reference genotypes G_d = mean relative-RFU profile from donor d's NOC=1 samples.

NOC decision: greedily add the donor (from a top-pool by the hybrid ID head) that
most reduces NLL; record NLL_k for k=1..K; pick k* by BIC (parsimony — the
likelihood-ratio/penalty step that NOCIt/EFM use). Reports per-NOC count accuracy,
downstream EM (top-k* of hybrid probs), and the 4-vs-5 separability AUC.

Usage:  python pgnoc.py            # fit globals on train, eval test
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import gammainc, gammaln

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
from train_set_transformer import topk_decode, per_noc_em

D = ROOT / "data_insilico_w"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
POOL = 8            # candidate pool size (top ID donors)
KMAX = 6
RHO = 3.0          # gamma shape (peak-height CV ~ 1/sqrt(rho)); tune on train
AT = 0.004         # analytical threshold as fraction of total template
DROPIN = 2.0       # per-bin drop-in penalty (nats)
BIC_PEN = 6.0      # extra nats per added donor (parsimony; ~ 0.5*k*ln n style)


def build_refs():
    X = np.expm1(np.load(D / "Xflat_train.npy").astype(np.float64))
    y = np.load(D / "y_train_set.npy"); n = np.load(D / "noc_train.npy").astype(int)
    ss = n == 1; G = np.zeros((45, X.shape[1]))
    for d in range(45):
        idx = np.where(ss & (y[:, d] == 1))[0]
        if len(idx):
            rel = X[idx] / (X[idx].sum(1, keepdims=True) + 1e-12)
            G[d] = rel.mean(0)
    return G


def hybrid_probs(split):
    m = SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2,
        m_inducing=32, n_classes=45, n_noc=6, dropout=0.1, cls_decoder="hybrid",
        n_flat=590, decouple_reject=True).to(DEV)
    m.load_state_dict(torch.load(ROOT / "results/hybrid_50k_weight/best_model.pt", weights_only=True)); m.eval()
    t = np.load(D / f"tokens_{split}.npy"); mk = np.load(D / f"mask_{split}.npy")
    xf = np.load(D / f"Xflat_{split}.npy").astype(np.float32); P = []
    with torch.no_grad():
        for i in range(0, len(t), 512):
            P.append(torch.sigmoid(m(torch.from_numpy(t[i:i+512]).to(DEV),
                torch.from_numpy(mk[i:i+512]).to(DEV),
                torch.from_numpy(xf[i:i+512]).to(DEV))["logits_cls"]).cpu().numpy())
    return np.concatenate(P)


from scipy.optimize import nnls


def fit_cost(h, G, S, alpha=0.5):
    """Gamma-heteroscedastic weighted NNLS fit for candidate set S (allows phi_d=0,
    so adding a donor can never worsen the fit -> residual is MONOTONE in k, the
    property the softmax-simplex MLE violated). Returns weighted residual norm.
    alpha = variance exponent: w = 1/(hrel)^alpha. alpha=0 OLS (emphasize major peaks),
    0.5 ~ gamma, 1.0 ~ Poisson (strong minor-peak emphasis -> better high-NOC)."""
    hrel = h / (h.sum() + 1e-12)
    w = 1.0 / np.power(hrel + 1e-4, alpha)
    A = (G[S].T) * w[:, None]; b = hrel * w
    phi, _ = nnls(A, b)
    return float(np.linalg.norm(b - A @ phi))


def estimate_noc(h, P_row, G, pen=None, pool=None, alpha=0.5):
    """Greedy donor add (top-`pool` by ID head) + BIC-style k*. Returns (k_star, cost[1..KMAX])."""
    pen = BIC_PEN if pen is None else pen
    pool = POOL if pool is None else pool
    remaining = list(np.argsort(P_row)[::-1][:pool]); chosen, cost = [], []
    for k in range(1, KMAX + 1):
        best_d, best_c = None, np.inf
        for d in remaining:
            c = fit_cost(h, G, chosen + [d], alpha)
            if c < best_c:
                best_c, best_d = c, d
        chosen.append(best_d); remaining.remove(best_d); cost.append(best_c)
    cost = np.array(cost)
    bic = cost + pen * np.arange(1, KMAX + 1)            # parsimony penalty
    return int(np.argmin(bic) + 1), cost


def tune_penalty(costs, noc, grid=np.linspace(0.0, 0.05, 51)):
    """Pick the parsimony penalty maximizing count accuracy on a calibration split."""
    ar = np.arange(1, KMAX + 1); best_p, best_acc = grid[0], -1
    for p in grid:
        k = (costs + p * ar).argmin(1) + 1
        acc = (k == np.clip(noc, 1, 5)).mean()
        if acc > best_acc:
            best_acc, best_p = acc, p
    return best_p, best_acc


def cost_matrix(split, G):
    P = hybrid_probs(split); H = np.expm1(np.load(D / f"Xflat_{split}.npy").astype(np.float64))
    C = np.zeros((len(P), KMAX))
    for i in range(len(P)):
        _, C[i] = estimate_noc(H[i], P[i], G)
    return P, C


def main():
    print("Building references + hybrid probs ..."); t0 = time.time()
    G = build_refs()
    nva = np.load(D / "noc_val.npy").astype(int); nte = np.load(D / "noc_test.npy").astype(int)
    yte = np.load(D / "y_test_set.npy")
    print(f"  computing cost curves (val {len(nva)} + test {len(nte)}) ...")
    _, Cva = cost_matrix("val", G)
    Pte, Cte = cost_matrix("test", G)
    pen, acc_va = tune_penalty(Cva, nva)
    print(f"  tuned penalty={pen:.4f} (val count acc {acc_va:.3f}); total {time.time()-t0:.0f}s")
    ar = np.arange(1, KMAX + 1)
    k_star = (Cte + pen * ar).argmin(1) + 1
    NLL = Cte

    # per-NOC count accuracy
    print("\n  per-NOC count accuracy (pgNOC k*):")
    print(f"  {'NOC':<5}{'acc':>7}{'n':>6}   pred-dist")
    for k in [1, 2, 3, 4, 5]:
        m = nte == k
        if m.sum():
            acc = (k_star[m] == k).mean()
            pd = {int(v): int(c) for v, c in zip(*np.unique(k_star[m], return_counts=True))}
            print(f"  {k:<5}{acc:>7.3f}{m.sum():>6}   {pd}")
    print(f"  overall count acc = {(k_star==nte).mean():.3f}")

    # downstream EM
    em = per_noc_em(yte, topk_decode(Pte, k_star), nte)
    orc = per_noc_em(yte, topk_decode(Pte, nte), nte)
    print(f"\n  {'':12}{'all':>7}{'N1':>7}{'N2':>7}{'N3':>7}{'N4':>7}{'N5':>7}")
    print("  oracle    " + "".join(f"{x:>7.3f}" for x in orc))
    print("  pgNOC     " + "".join(f"{x:>7.3f}" for x in em))

    # 4-vs-5 separability of the NLL-gain feature (does adding 5th donor help?)
    from sklearn.metrics import roc_auc_score
    m45 = np.isin(nte, [4, 5])
    gain5 = NLL[:, 3] - NLL[:, 4]          # NLL drop from adding the 5th donor
    auc = roc_auc_score((nte[m45] == 5).astype(int), gain5[m45])
    print(f"\n  AUC(4 vs 5) from NLL-gain of 5th donor = {auc:.3f}  (height-only 0.67, deconv-v2 0.69)")

    np.save(ROOT / "results" / "pgnoc_kstar_test.npy", k_star)
    json.dump({"count_acc": float((k_star == nte).mean()),
               "per_noc_em": [float(x) for x in em], "oracle_em": [float(x) for x in orc],
               "auc_4v5": float(auc)},
              open(ROOT / "results" / "pgnoc_metrics.json", "w"), indent=2)
    print("\n  saved -> results/pgnoc_metrics.json")


if __name__ == "__main__":
    main()

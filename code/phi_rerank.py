"""
phi_rerank.py — DEPLOYABLE post-hoc reranking of the per-donor cls logits with an INDEPENDENT
EM mixture-proportion (Mx) deconvolution.  Kept as a SEPARATE module so the whole feature can be
disabled (or deleted) without touching the model / training code — rollback = don't pass --phi_rerank.

WHY (paper-grounded):
  • The combination rule `score = z(logit) + alpha * z(log phi)` is a LOGARITHMIC OPINION POOL /
    PRODUCT-OF-EXPERTS (Genest & Zidek 1986; Hinton 2002): under Bayes + INDEPENDENCE, combining two
    expert distributions = add log-probabilities = add logits.  Its validity REQUIRES independence —
    which is exactly why the UNIFORM-compat deconvolution phi (corr ~0.73 with logits_cls) lifts the
    oracle while the neural attr-phi (corr ~0.95, redundant) does not.
  • The phi here is the EuroForMix-class height+genotype deconvolution (Bleka et al., 2016): peak
    height is competitively split among the donors whose reference genotype carries the allele
    (UNIFORM compat — NO neural signal, so the channel stays independent) plus a background sink,
    iterated by EM to a per-donor mixture proportion.

VERIFIED: on inc22_fixed_aslot_seed42 this reranks N5 oracle 0.831 -> 0.901 (alpha tuned on val).
It changes only the RANKING (argsort); the count head is left to decide k.  n=1 checkpoint — confirm
across seeds before trusting the magnitude (per the project's C6/F5 selection discipline).
"""
import numpy as np


def build_carriers(donor_geno: np.ndarray, donor_geno_mask: np.ndarray):
    """allele-key (locus, round(allele*10)) -> list of donor indices whose genotype carries it."""
    C = donor_geno.shape[0]
    carr: dict[tuple[int, int], list[int]] = {}
    for c in range(C):
        for j in range(donor_geno.shape[1]):
            if donor_geno_mask[c, j]:
                key = (int(round(float(donor_geno[c, j, 0]))),
                       int(round(float(donor_geno[c, j, 1]) * 10)))
                lst = carr.setdefault(key, [])
                if c not in lst:
                    lst.append(c)
    return carr, C


def deconv_phi(tokens: np.ndarray, mask: np.ndarray,
               donor_geno: np.ndarray, donor_geno_mask: np.ndarray,
               n_iters: int = 500, tol: float = 1e-3) -> np.ndarray:
    """Uniform-compat height-EM mixture-proportion deconvolution. Returns (N, C) proportions.
    Deterministic in (tokens, genotypes) — NO model weights — so it is a deployable, independent signal.

    Iterates to CONVERGENCE (max|delta phi| < tol), not to a fixed count. The old n_iters=10 stopped
    with 89.5% of the entries still moving and max|phi - phi_converged| = 0.235, i.e. it returned a
    point on the way rather than the solution; converging is worth +1.35pp EM on real test and costs
    no parameter, since tol is a numerical criterion and not something to fit."""
    carr, C = build_carriers(donor_geno, donor_geno_mask)
    N = len(tokens); mask = mask.astype(bool)
    PH = np.zeros((N, C), dtype=np.float64)
    for i in range(N):
        idx, keys = [], []
        for p in np.where(mask[i])[0]:
            key = (int(round(float(tokens[i, p, 0]))), int(round(float(tokens[i, p, 1]) * 10)))
            if key in carr:
                idx.append(p); keys.append(key)
        if not idx:
            continue
        h = np.expm1(tokens[i, idx, 2].astype(np.float64))      # observed RFU heights
        n = len(idx); S = np.full((n, C + 1), -1e9)
        for r, key in enumerate(keys):
            for c in carr[key]:
                S[r, c] = 0.0                                    # UNIFORM compat among carriers
            S[r, C] = -2.0                                       # background sink
        phi = np.ones(C + 1) / (C + 1)
        for _ in range(n_iters):
            z = S + np.log(phi + 1e-9); z -= z.max(1, keepdims=True)
            A = np.exp(z); A /= A.sum(1, keepdims=True)          # peak -> {donors, bg} responsibilities
            w = (A[:, :C] * h[:, None]).sum(0); bg = (A[:, C] * h).sum()
            tot = w.sum() + bg
            new = np.concatenate([w, [bg]]) / max(tot, 1e-9)
            done = np.abs(new - phi).max() < tol
            phi = new
            if done:
                break
        PH[i] = phi[:C]
    return PH


def _z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64); s = a.std()
    return (a - a.mean()) / (s if s > 1e-9 else 1.0)


def rerank_scores(logits: np.ndarray, PH: np.ndarray, alpha: float) -> np.ndarray:
    """Row-wise logarithmic-opinion-pool score z(logit) + alpha*z(log phi). Returns a RANKING score
    (NOT a probability) — use argsort for top-k. alpha=0 reproduces the model's own ranking exactly."""
    out = np.empty(logits.shape, dtype=np.float64)
    for i in range(len(logits)):
        out[i] = _z(logits[i]) + alpha * _z(np.log(PH[i] + 1e-6))
    return out


def tune_alpha(logits_val: np.ndarray, PH_val: np.ndarray, y_val: np.ndarray, noc_val: np.ndarray,
               grid=(0.0, 0.2, 0.3, 0.5, 0.75, 1.0), ks=(5, 4, 3)) -> float:
    """Pick alpha maximizing mean oracle EM (top-true-k) over the high-NOC strata on VAL (C6-clean:
    selection on val, never test)."""
    noc_val = np.clip(noc_val, 1, 5); C = logits_val.shape[1]
    ks = [k for k in ks if (noc_val == k).any()]
    best_a, best_v = 0.0, -1.0
    for a in grid:
        R = rerank_scores(logits_val, PH_val, a)
        accs = []
        for k in ks:
            sel = np.where(noc_val == k)[0]; hit = 0
            for i in sel:
                top = np.argsort(R[i])[::-1][:k]; pr = np.zeros(C, int); pr[top] = 1
                hit += int((pr == y_val[i]).all())
            accs.append(hit / max(1, len(sel)))
        v = float(np.mean(accs)) if accs else -1.0
        if v > best_v:
            best_v, best_a = v, a
    return best_a

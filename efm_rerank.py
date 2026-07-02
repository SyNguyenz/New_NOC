"""
efm_rerank.py — EuroForMix-CLASS per-donor mixture-proportion deconvolution (Bleka et al. 2016),
the PROPER continuous-model upgrade of phi_rerank's crude uniform-compat EM. SEPARATE module
(rollback = don't call it). Adds the three terms the crude φ omits, each PROVEN in the EFM model:
  • COPY NUMBER  — a homozygous donor contributes 2× to its allele (responsibility ∝ φ_c·n_{c,a}).
  • BACK-STUTTER — a peak at allele a includes ξ·(height at a+1); subtract it so a stutter peak no
    longer feeds a decoy that merely carries a-1 (the decoy mechanism, native to the EFM likelihood).
  • DEGRADATION  — large fragments are attenuated; optionally correct height by exp(λ·(size-size_min)).
Deterministic in (tokens, sizes, genotypes) — NO neural signal — so it stays an INDEPENDENT channel
for the logarithmic-opinion-pool rerank (reuse phi_rerank.rerank_scores / tune_alpha on the output).
NOTE: this is the EFM *proportion* deconvolution (ranking-relevant terms), not the full gamma-MLE LR;
if copy-number+stutter+degradation move the oracle, the gamma height-likelihood is the next refinement.
"""
import numpy as np


def build_cn_carriers(donor_geno: np.ndarray, donor_geno_mask: np.ndarray):
    """allele-key (locus, round(allele*10)) -> {donor_idx: copy_number ∈ {1,2}}."""
    C = donor_geno.shape[0]
    cn: dict[tuple[int, int], dict[int, int]] = {}
    for c in range(C):
        for j in range(donor_geno.shape[1]):
            if donor_geno_mask[c, j]:
                key = (int(round(float(donor_geno[c, j, 0]))),
                       int(round(float(donor_geno[c, j, 1]) * 10)))
                d = cn.setdefault(key, {})
                d[c] = d.get(c, 0) + 1
    return cn, C


def deconv_efm(tokens: np.ndarray, mask: np.ndarray, sizes: np.ndarray,
               donor_geno: np.ndarray, donor_geno_mask: np.ndarray,
               n_iters: int = 12, xi: float = 0.08, degr_lambda: float = 0.0) -> np.ndarray:
    """EuroForMix-class height-EM with copy-number, back-stutter subtraction and (optional)
    degradation correction. Returns (N, C) mixture proportions. xi = back-stutter ratio (~0.05-0.10
    typical); degr_lambda>0 corrects large-fragment attenuation (0 = off)."""
    cn, C = build_cn_carriers(donor_geno, donor_geno_mask)
    N = len(tokens); mask = mask.astype(bool)
    PH = np.zeros((N, C), dtype=np.float64)
    for i in range(N):
        ps = np.where(mask[i])[0]
        if len(ps) == 0:
            continue
        # per-locus allele->height map (for back-stutter parent lookup at a+1 repeat = key+10)
        loc_h: dict[int, dict[int, float]] = {}
        info = []
        for p in ps:
            L = int(round(float(tokens[i, p, 0]))); a = int(round(float(tokens[i, p, 1]) * 10))
            h = float(np.expm1(tokens[i, p, 2])); sz = float(sizes[i, p])
            d = loc_h.setdefault(L, {}); d[a] = max(d.get(a, 0.0), h)
            info.append((L, a, h, sz))
        smin = min(x[3] for x in info)
        keys, heff = [], []
        for (L, a, h, sz) in info:
            key = (L, a)
            if key not in cn:
                continue
            parent = loc_h.get(L, {}).get(a + 10, 0.0)          # peak one repeat LARGER = stutter parent
            h_true = max(h - xi * parent, 0.0)                  # subtract back-stutter contribution
            if degr_lambda > 0:
                h_true *= np.exp(degr_lambda * (sz - smin))     # correct large-fragment attenuation
            keys.append(key); heff.append(h_true)
        if not keys:
            continue
        n = len(keys); S = np.full((n, C + 1), -1e9)
        for r, key in enumerate(keys):
            for c, copies in cn[key].items():
                S[r, c] = np.log(copies)                        # copy-number-weighted responsibility
            S[r, C] = -2.0                                       # background sink
        h = np.asarray(heff)
        phi = np.ones(C + 1) / (C + 1)
        for _ in range(n_iters):
            z = S + np.log(phi + 1e-9); z -= z.max(1, keepdims=True)
            A = np.exp(z); A /= A.sum(1, keepdims=True)
            w = (A[:, :C] * h[:, None]).sum(0); bg = (A[:, C] * h).sum()
            tot = w.sum() + bg
            phi = np.concatenate([w, [bg]]) / max(tot, 1e-9)
        PH[i] = phi[:C]
    return PH

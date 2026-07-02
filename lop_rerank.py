"""
lop_rerank.py — calibrated MULTI-channel logarithmic-opinion-pool reranking. SEPARATE module
(rollback = don't call it).  Two additions over phi_rerank:
  (1) LR-FITTED pool weights (calibrated linear opinion pool / stacking; Genest-Zidek 1986,
      Wolpert 1992) — replaces the hand-tuned single alpha that pinned to the grid ceiling.
  (2) NOISE-AWARE donor support channel: per donor, sum over its PRESENT alleles of
      IDF(rarity) * height * (1 - Pr_noise(height)).  Pr_noise is a val-fit height logistic
      (Tvedebrink peak-quality / Balding-Buckleton drop-in); IDF = Robertson-Sparck-Jones rarity.
All channels come from genotypes + heights + a val-fit logistic — NO neural signal — so each stays
INDEPENDENT of the cls logit (the property the LOP / ensemble-ambiguity gain requires).
"""
import numpy as np
from phi_rerank import build_carriers, _z


def fit_noise_model(tokens, mask, donor_geno, donor_geno_mask, y):
    """Logistic Pr(peak is noise | log1p height). Label: peak key carried by NO true donor of the
    sample => noise(1), else real(0). Returns (b0, b1). Height-dependent peak-quality (Tvedebrink)."""
    from sklearn.linear_model import LogisticRegression
    carr, C = build_carriers(donor_geno, donor_geno_mask)
    mask = mask.astype(bool); H, Y = [], []
    for i in range(len(tokens)):
        true = set(np.where(y[i] > 0.5)[0])
        for p in np.where(mask[i])[0]:
            key = (int(round(float(tokens[i, p, 0]))), int(round(float(tokens[i, p, 1]) * 10)))
            if key in carr:
                h = float(np.expm1(tokens[i, p, 2]))
                H.append(np.log1p(max(h, 0.0))); Y.append(0 if (set(carr[key]) & true) else 1)
    lr = LogisticRegression(max_iter=1000).fit(np.array(H).reshape(-1, 1), np.array(Y))
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def donor_support(tokens, mask, donor_geno, donor_geno_mask, noise_b):
    """(N, C) rarity(IDF) * height * (1 - Pr_noise(height)) per-donor support. A tall private allele
    scores high for its single owner; a low-height phantom is discounted toward 0 -> decoy suppressed."""
    carr, C = build_carriers(donor_geno, donor_geno_mask)
    idf = {k: np.log((C + 1.0) / (1.0 + len(v))) for k, v in carr.items()}
    b0, b1 = noise_b; mask = mask.astype(bool)
    S = np.zeros((len(tokens), C))
    for i in range(len(tokens)):
        for p in np.where(mask[i])[0]:
            key = (int(round(float(tokens[i, p, 0]))), int(round(float(tokens[i, p, 1]) * 10)))
            if key in carr:
                h = float(np.expm1(tokens[i, p, 2]))
                pn = 1.0 / (1.0 + np.exp(-(b0 + b1 * np.log1p(max(h, 0.0)))))
                w = idf[key] * h * (1.0 - pn)
                for c in carr[key]:
                    S[i, c] += w
    return S


def _stack_z(channels):
    """list of (N,C) -> (N*C, n_ch) with per-row z-scoring of each channel."""
    N = channels[0].shape[0]
    cols = [np.stack([_z(ch[i]) for i in range(N)]).reshape(-1) for ch in channels]
    return np.stack(cols, 1)


def fit_pool_weights(channels_val, y_val, balanced=True):
    """Calibrated linear opinion pool: LR(y ~ per-row-z(channels)) pooled over (sample, donor).
    Returns weight vector (one per channel). Ranking uses these weights on per-row z-scored channels."""
    from sklearn.linear_model import LogisticRegression
    X = _stack_z(channels_val); Y = y_val.reshape(-1).astype(int)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced" if balanced else None).fit(X, Y)
    return lr.coef_[0]


def pool_scores(channels, weights):
    """Row-wise sum_k w_k * z(channel_k). Returns ranking score (argsort for top-k)."""
    N, C = channels[0].shape; out = np.zeros((N, C))
    for i in range(N):
        for w, ch in zip(weights, channels):
            out[i] += w * _z(ch[i])
    return out

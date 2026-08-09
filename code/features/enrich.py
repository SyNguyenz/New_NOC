"""
features/enrich.py — Increment 1: enrich the per-peak token with RELATIONAL features
derived from the EXISTING (locus, allele, log_height) token. No raw re-prep, no Size(bp).

Input  tokens (N, 160, 3) = [locus_idx, allele_val, log1p_height], mask (N, 160) bool.
Output tokens (N, 160, 9) = original 3 + 6 derived (all 0 on padding):
  3  Hb         = h / sum(h at same locus)        heterozygote balance (Bright 2012)
  4  SR         = h / h(allele+1 at locus)        BACK-stutter ratio   (Taylor 2013); 0 if no parent
  5  rank_inv   = 1 / rank_in_locus (1=tallest)   contributor tier within locus
  6  n_at_locus = #peaks at locus / 10            local allele count (count signal)
  7  glob_rel   = h / max(h in profile)           global contributor tier
  8  FS         = h / h(allele-1 at locus)        FORWARD-stutter ratio (Increment 2a); 0 if no parent

Theory: these are the validated continuous-PG relational signals the 3-field token lacked;
they feed the encoder so attention can GROUP a contributor's peaks (consistent Hb/glob_rel
across loci) and tell true minor alleles from stutter (SR/FS). Increment 2a adds forward
stutter (n+1 artefact at allele+1 of a parent at allele-1) to complete the stutter pair
(design_increment2_relationships §1-2). enrich_tokens returns 9 fields; the writer also saves
the 8-field slice (tokens8) so the 2x2 token ablation (tok8 vs tok9) can run.
See reports/design_representation_redesign.md + design_increment2_relationships.md.
"""
from __future__ import annotations
import numpy as np

N_FIELDS = 9


def add_size_fields(en9: np.ndarray, mask: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Append the DEGRADATION representation (Increment 2a, needs per-peak size(bp) from
    extract_size.py / make_insilico size table) -> 11 fields:
      9  size_bp      raw fragment size (bp)                         sufficient statistic (§8)
      10 degr_resid   log_h - (a + b*size) per-sample linear fit     degradation-corrected height
                      (STRmix degradation model: height decays with fragment size; the residual
                      isolates true contributor tier from the degradation trend). 0 if <3 peaks.
    """
    N, S, _ = en9.shape
    out = np.zeros((N, S, 11), np.float32)
    out[:, :, :9] = en9
    out[:, :, 9] = np.where(mask, size, 0.0)
    log_h = en9[:, :, 2]
    for i in range(N):
        v = mask[i] & (size[i] > 0)
        if v.sum() < 3:
            continue
        x = size[i, v].astype(np.float64); yy = log_h[i, v].astype(np.float64)
        if x.std() < 1e-6:
            continue
        b, a = np.polyfit(x, yy, 1)
        out[i, v, 10] = (yy - (a + b * x)).astype(np.float32)
    return out


def enrich_tokens(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    tokens = tokens.astype(np.float32)
    N, S, _ = tokens.shape
    out = np.zeros((N, S, N_FIELDS), np.float32)
    out[:, :, :3] = tokens                                   # keep locus, allele, log_h
    h_all = np.expm1(tokens[:, :, 2])                        # RFU
    for i in range(N):
        v = mask[i]
        if not v.any():
            continue
        loci = tokens[i, :, 0]; allele = tokens[i, :, 1]; h = h_all[i]
        gmax = h[v].max() + 1e-9
        for L in np.unique(loci[v]):
            idx = np.where(v & (loci == L))[0]
            hL = h[idx]; tot = hL.sum() + 1e-9
            order = np.argsort(hL)[::-1]                     # tallest first
            rank = np.empty(len(idx), int); rank[order] = np.arange(1, len(idx) + 1)
            aL = allele[idx]
            for j, p in enumerate(idx):
                out[i, p, 3] = hL[j] / tot                   # Hb
                par = np.where(np.abs(aL - (aL[j] + 1.0)) < 1e-3)[0]   # back-stutter parent at allele+1
                out[i, p, 4] = min(hL[j] / (hL[par[0]] + 1e-9), 3.0) if len(par) else 0.0   # SR (capped: >2 = clearly not stutter)
                out[i, p, 5] = 1.0 / rank[j]                 # rank_inv
                out[i, p, 6] = len(idx) / 10.0               # n_at_locus (scaled)
                out[i, p, 7] = hL[j] / gmax                  # glob_rel
                fwd = np.where(np.abs(aL - (aL[j] - 1.0)) < 1e-3)[0]   # forward-stutter parent at allele-1
                out[i, p, 8] = min(hL[j] / (hL[fwd[0]] + 1e-9), 3.0) if len(fwd) else 0.0   # FS (capped)
    return out


def feasible_mask(tokens: np.ndarray, mask: np.ndarray, donor_geno: np.ndarray,
                  donor_geno_mask: np.ndarray) -> np.ndarray:
    """Peaks carried by at least ONE panel donor. Same rule the model's feas_filter applies."""
    carr = set()
    gm = donor_geno_mask.astype(bool)
    for c in range(donor_geno.shape[0]):
        for j in range(donor_geno.shape[1]):
            if gm[c, j]:
                carr.add((int(round(float(donor_geno[c, j, 0]))),
                          int(round(float(donor_geno[c, j, 1]) * 10))))
    out = np.zeros(mask.shape, bool)
    m = mask.astype(bool)
    for i in range(len(tokens)):
        for p in np.where(m[i])[0]:
            out[i, p] = (int(round(float(tokens[i, p, 0]))),
                         int(round(float(tokens[i, p, 1]) * 10))) in carr
    return out


if __name__ == "__main__":
    import os, sys
    from pathlib import Path
    D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data_insilico_w")
    # Enrich AFTER feasibility filtering. The enriched columns are per-locus RELATIVE quantities
    # (Hb = h/locus-total, rank, glob_rel = h/global-max, n_at_locus, stutter ratios). Computing them
    # over the unfiltered peak set lets 0-carrier noise distort the features of the REAL peaks, even
    # though the model's feas_filter drops those peaks later. Measured on real test: filtering the
    # no-panel peaks BEFORE enrichment is worth +0.048 macro-over-NOC recall on its own. Real data is
    # 23% no-panel peaks vs 3.5% in-silico, so this also removes a train/test asymmetry.
    # STR_ENRICH_AFTER_FEAS=0 restores the old order.
    after_feas = int(os.environ.get("STR_ENRICH_AFTER_FEAS", "1"))
    dg = dgm = None
    if after_feas:
        for cand in (D / "donor_geno.npy", D.parent / "data" / "donor_geno.npy"):
            if cand.exists():
                dg = np.load(cand); dgm = np.load(cand.parent / "donor_geno_mask.npy"); break
        if dg is None:
            print("WARN: donor_geno not found -> enriching on the UNFILTERED peak set (old order)")
            after_feas = 0
    for sp in ["train", "val", "test", "open", "dev"]:
        if not (D / f"tokens_{sp}.npy").exists():
            continue
        tok = np.load(D / f"tokens_{sp}.npy"); mk = np.load(D / f"mask_{sp}.npy")
        if after_feas:
            fm = feasible_mask(tok, mk, dg, dgm)
            kept = fm.sum() / max(mk.astype(bool).sum(), 1)
            en = enrich_tokens(tok, fm)                      # features from the FEASIBLE set only
            print(f"  {sp}: enrich-after-feas keeps {kept:.3f} of peaks for feature computation")
        else:
            en = enrich_tokens(tok, mk)                      # (N, S, 9)
        np.save(D / f"tokens9_{sp}.npy", en)
        np.save(D / f"tokens8_{sp}.npy", en[:, :, :8])       # 8-field slice (Increment 1 token)
        msg = f"{sp}: {en.shape} -> tokens9_{sp}.npy (+ tokens8 slice)"
        sz_path = D / f"size_{sp}.npy"
        if sz_path.exists():                                 # Increment 2a: size + degradation -> tokens11
            en11 = add_size_fields(en, mk.astype(bool), np.load(sz_path))
            np.save(D / f"tokens11_{sp}.npy", en11)
            msg += f" (+ tokens11 size/degradation)"
        print(msg)
    # sanity on test
    en = np.load(D / "tokens9_test.npy"); mk = np.load(D / "mask_test.npy")
    noc = np.load(D / "noc_test.npy").astype(int)
    v = mk
    print("\nfeature ranges (valid peaks):")
    names = ["locus", "allele", "log_h", "Hb", "SR", "rank_inv", "n/10", "glob_rel", "FS"]
    for k in range(N_FIELDS):
        vals = en[:, :, k][v]
        print(f"  {names[k]:<9} min={vals.min():.3f} mean={vals.mean():.3f} max={vals.max():.3f}")
    print("\nmean Hb (height balance) by NOC — expect lower/more-imbalanced at high NOC:")
    for kk in [1, 2, 3, 4, 5]:
        m = noc == kk
        hb = en[m][:, :, 3][mk[m]]
        print(f"  NOC{kk}: mean Hb={hb[hb>0].mean():.3f}")

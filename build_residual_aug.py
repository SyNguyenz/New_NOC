"""
build_residual_aug.py — Increment 5 prep (NEW; does not modify existing files).

Write `data_res/` = a copy of the base data dir whose TRAIN arrays are augmented with
RESIDUAL copies: for each NOC>=2 train mixture (true donors S), subtract a random subset
R⊂S of donors' fitted contributions (NNLS on reference profiles G in linear RFU space — the
EXACT operation eval_peel_decode.py uses at decode), re-tokenize the residual, and label it
with the REMAINING donors S∖R. Training on original ⊕ residual teaches the model to score
residual/low-template profiles (fixes the frozen-model OOD-on-residual failure, neural_peel.py)
and decouples each donor's decision from which others co-occur (combo-invariance).

Run AFTER make_dev_split (so residuals never leak into dev) and AFTER features/enrich (we append
already-enriched tokens8 rows). Only TRAIN arrays change; val/test/dev/open + Xflat_train (kept
CLEAN so the reference G stays single-source) are copied unchanged.

Usage:  python build_residual_aug.py <base_dir> <out_dir> [--mode rand1|randk] [--seed 42] [--copies 1]
Smoke:  RES_SMOKE=1 python build_residual_aug.py <base_dir> <out_dir>
"""
import os, sys, shutil, argparse
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

# xflat_to_tokens needs make_insilico's meta (BIN_LOCUS/ALLELE, MAX_SEQ). STR_DATA_DIR must point
# at a dir with meta_set.json; the runner sets it. Fall back to the base_dir given on the CLI.
if len(sys.argv) > 1 and not os.environ.get("STR_DATA_DIR"):
    os.environ["STR_DATA_DIR"] = sys.argv[1]
from make_insilico import xflat_to_tokens
from features.enrich import enrich_tokens
from train_set_transformer import build_pgnoc_refs


def build(base_dir, out_dir, mode="rand1", seed=42, copies=1, limit=None):
    base = Path(base_dir); out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # copy EVERYTHING first (val/test/dev/open/meta/Xflat all stay; train arrays overwritten below)
    for f in list(base.glob("*.npy")) + list(base.glob("*.json")):
        shutil.copy(f, out / f.name)

    rng = np.random.default_rng(seed)
    Xf   = np.load(base / "Xflat_train.npy").astype(np.float64)     # log1p flat
    tok8 = np.load(base / "tokens8_train.npy").astype(np.float32)   # enriched 8-field (original)
    msk  = np.load(base / "mask_train.npy").astype(bool)
    y    = np.load(base / "y_train_set.npy").astype(np.float32)
    noc  = np.load(base / "noc_train.npy").astype(np.int64)
    attr = np.load(base / "attr_train.npy").astype(np.int64)
    phi  = np.load(base / "phi_train.npy").astype(np.float32)
    tok3 = np.load(base / "tokens_train.npy").astype(np.float32)    # 3-field (for re-enrich consistency)
    size = np.load(base / "size_train.npy").astype(np.float32) if (base / "size_train.npy").exists() else None

    # CLEAN single-source references (build from the ORIGINAL train NOC1 — residuals are NOT added to Xflat)
    G = build_pgnoc_refs(Xf, y, noc)                                # (45, n_flat) LINEAR relative profiles

    r_tok3, r_mask, r_y, r_noc, r_attr, r_phi = [], [], [], [], [], []
    n_skip = 0
    idx2 = np.where(noc >= 2)[0]
    if limit:
        idx2 = idx2[:limit]
    for _c in range(copies):
        for i in idx2:
            S = list(np.where(y[i] == 1)[0])
            K = len(S)
            if K < 2:
                continue
            n_peel = 1 if mode == "rand1" else int(rng.integers(1, K))   # randk: 1..K-1
            R = list(rng.choice(S, size=n_peel, replace=False))
            remaining = [d for d in S if d not in R]
            mix = np.expm1(Xf[i])                                   # linear RFU
            A = G[R].T
            phi_fit, _ = nnls(A, mix)
            resid = np.clip(mix - A @ phi_fit, 0.0, None)           # residual RFU (same op as decode)
            rflat = np.log1p(resid).astype(np.float32)
            t3, m3, _, _ = xflat_to_tokens(rflat)                   # (MAX_SEQ,3),(MAX_SEQ,)
            if m3.sum() < 2:                                        # degenerate (over-subtracted) → skip
                n_skip += 1; continue
            yr = np.zeros(45, np.float32); yr[remaining] = 1.0
            pr = phi[i] * yr                                        # renormalised proportions over remaining
            pr = pr / pr.sum() if pr.sum() > 0 else pr
            r_tok3.append(t3); r_mask.append(m3); r_y.append(yr)
            r_noc.append(len(remaining)); r_attr.append(np.full(t3.shape[0], -1, np.int64)); r_phi.append(pr)

    if r_tok3:
        r_tok3 = np.stack(r_tok3).astype(np.float32)
        r_mask = np.stack(r_mask).astype(bool)
        r_tok8 = enrich_tokens(r_tok3, r_mask)[:, :, :8].astype(np.float32)   # enrich residual → 8-field
        r_y = np.stack(r_y).astype(np.float32); r_noc = np.array(r_noc, np.int64)
        r_attr = np.stack(r_attr).astype(np.int64); r_phi = np.stack(r_phi).astype(np.float32)
        # concatenate original ⊕ residual and OVERWRITE train arrays in out_dir
        np.save(out / "tokens8_train.npy", np.concatenate([tok8, r_tok8]))
        np.save(out / "tokens_train.npy",  np.concatenate([tok3, r_tok3]))
        np.save(out / "mask_train.npy",    np.concatenate([msk,  r_mask]))
        np.save(out / "y_train_set.npy",   np.concatenate([y,    r_y]))
        np.save(out / "noc_train.npy",     np.concatenate([noc,  r_noc]))
        np.save(out / "attr_train.npy",    np.concatenate([attr, r_attr]))
        np.save(out / "phi_train.npy",     np.concatenate([phi,  r_phi]))
        if size is not None:               # keep enrich(tok11) from crashing on length mismatch
            np.save(out / "size_train.npy", np.concatenate([size, np.zeros((len(r_tok3), size.shape[1]), np.float32)])
                    if size.ndim == 2 else np.concatenate([size, np.zeros(len(r_tok3), np.float32)]))
        # NOTE: Xflat_train left CLEAN (original length) so build_pgnoc_refs stays single-source.
        print(f"residual-aug ({mode}, copies={copies}): +{len(r_tok3)} residual rows "
              f"(orig {len(tok8)} -> {len(tok8)+len(r_tok3)}); skipped {n_skip} degenerate")
    else:
        print("WARNING: no residual rows produced")
    print(f"wrote {out}")


if __name__ == "__main__":
    if os.environ.get("RES_SMOKE"):
        ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("out")
        a, _ = ap.parse_known_args()
        build(a.base, a.out, mode="rand1", copies=1, limit=40); sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("out")
    ap.add_argument("--mode", default="rand1", choices=["rand1", "randk"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copies", type=int, default=1)
    a = ap.parse_args()
    build(a.base, a.out, a.mode, a.seed, a.copies)

"""
make_replicates.py — RICHER DATA (lever #1): build a REPLICATE-augmented dataset so the model can
consume R replicate amplifications of each mixture (EFMrep idea; Riman & Bleka 2022). For each
existing mixture it draws the SAME ground truth (contributors from y_{split}_set, proportions from
phi_{split}, one template+degradation per mixture) and generates R profiles with INDEPENDENT
stochastic dropout/stutter/Gamma noise (make_insilico's EuroForMix peak model), then POOLS their
peaks into one set (the union, replicate-id tagged). A faint minor recovered across replicates is
genuinely NEW information — not diversity-bounded like better modelling of one profile.

Writes ONLY the replicate-specific per-peak arrays, suffixed _rep{R}, INTO the source dir, so all
labels/meta/open-set/genotypes are REUSED unchanged (rollback = don't pass --replicates):
    tokens{8,9,11}_{split}_rep{R}.npy, tokens_{split}_rep{R}.npy, mask/attr/size/repid_{split}_rep{R}.npy
y/noc/phi are per-mixture (unchanged) and loaded without the suffix by the trainer.

Usage (the kaggle runner calls this on the WRITABLE data dir, like make_dev_split / features.enrich):
        STR_REPLICATES=3 python make_replicates.py <DATA_DIR>          # all splits, R=3
        STR_REPLICATES=2 STR_LIMIT=100 python make_replicates.py <DATA_DIR>   # smoke (first 100/split)
PREREQ: run AFTER copy + make_dev_split (so y/noc/phi + the 'dev' selection split exist in DATA_DIR).
"""
import os, sys
from pathlib import Path
import numpy as np
import make_insilico as mi
from features.enrich import enrich_tokens, add_size_fields

R     = int(os.environ.get("STR_REPLICATES", "3"))
# Read/WRITE the dir passed as argv[1] (the runner's writable DATA_W); fall back to env/default for standalone use.
SRC   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("STR_SRC_DIR", "data_insilico_w"))
LIMIT = int(os.environ.get("STR_LIMIT", "0"))            # 0 = all; else first N per split (smoke)
SPLITS = os.environ.get("STR_SPLITS", "train,val,test,dev").split(",")   # dev = selection set (made by make_dev_split)
MAXR  = mi.MAX_SEQ * R
SIZE  = mi.build_bin_size()
if SIZE is None:                                         # size_train.npy absent -> degradation off (minor effect; don't crash)
    SIZE = np.zeros(mi.N_FLAT, dtype=np.float64)
    print("[warn] build_bin_size() returned None (no size_train in STR_DATA_DIR) -> degradation OFF")
N_FLAT, DOS, EFF, AT, CV = mi.N_FLAT, mi.DONOR_DOSAGE, mi.EFF_BIN, mi.AT, mi.PH_CV
ST, BA, DEGMAX = mi.STUTTER_TARGET, mi.BIN_ALLELE, mi.DEG_BETA_MAX
SR_S, SR_I, SR_SG = mi.SR_SLOPE, mi.SR_INTERCEPT, mi.SR_SIGMA


def gen_rep(cols, phc, T, beta, rng):
    """One replicate of a FIXED mixture (shared cols/phi/T/beta) — re-draw Gamma+stutter+dropout.
    Returns (xflat = log1p RFU, attr_bin = dominant donor col per bin)."""
    deg = np.exp(-beta * np.maximum(SIZE - 100.0, 0.0)); shape = 1.0 / CV**2
    contrib = np.zeros((len(cols), N_FLAT))
    for di, (c, p) in enumerate(zip(cols, phc)):
        mu = T * p * DOS[c] * deg * EFF; nz = np.where(mu > 0)[0]
        if len(nz): contrib[di, nz] = rng.gamma(shape, mu[nz] * CV**2)
    mix = contrib.sum(0)
    ph = mix.copy(); js = np.where((ST >= 0) & (ph > 0))[0]
    if len(js):
        a = BA[js].astype(float); sr = np.clip(SR_S * a + SR_I, 0.01, 0.18) * rng.lognormal(0, SR_SG, len(js))
        np.add.at(mix, ST[js], sr * ph[js])
    drop = mix < AT; mix[drop] = 0.0
    attr_bin = np.full(N_FLAT, -1, dtype=np.int16); live = (~drop) & (contrib.max(0) > 0)
    if len(cols): attr_bin[live] = np.asarray(cols, dtype=np.int16)[contrib.argmax(0)[live]]
    return np.log1p(mix).astype(np.float32), attr_bin


def build_split(sp, rng):
    y   = np.load(SRC / f"y_{sp}_set.npy"); noc = np.load(SRC / f"noc_{sp}.npy"); phi = np.load(SRC / f"phi_{sp}.npy")
    N = len(y) if LIMIT <= 0 else min(LIMIT, len(y))
    TOK = np.zeros((N, MAXR, 3), np.float32); MASK = np.zeros((N, MAXR), bool)
    ATTR = np.full((N, MAXR), -1, np.int16); SZ = np.zeros((N, MAXR), np.float32); REP = np.full((N, MAXR), -1, np.int16)
    for i in range(N):
        cols = np.where(y[i] > 0.5)[0]
        if len(cols) == 0:
            continue
        phc = phi[i, cols].astype(np.float64); phc = phc / max(phc.sum(), 1e-9)
        T = float(np.exp(rng.normal(mi.PEAK_TMU, mi.PEAK_TSIG)))   # one template per mixture, shared across its replicates
        beta = rng.uniform(0, DEGMAX)
        pos = 0
        for r in range(R):
            xf, ab = gen_rep(cols, phc, T, beta, rng)
            tok, msk, attr, size = mi.xflat_to_tokens(xf, ab, SIZE)   # (160,3),(160,),(160,),(160,)
            sel = np.where(msk)[0]
            take = min(len(sel), MAXR - pos)
            if take <= 0:
                break
            sel = sel[:take]
            TOK[i, pos:pos + take] = tok[sel]; MASK[i, pos:pos + take] = True
            ATTR[i, pos:pos + take] = attr[sel]; SZ[i, pos:pos + take] = size[sel]; REP[i, pos:pos + take] = r
            pos += take
    en9 = enrich_tokens(TOK, MASK); en8 = en9[:, :, :8]; en11 = add_size_fields(en9, MASK, SZ)
    sfx = f"_{sp}_rep{R}"
    np.save(SRC / f"tokens{sfx}.npy", TOK);   np.save(SRC / f"mask{sfx}.npy", MASK)
    np.save(SRC / f"tokens8{sfx}.npy", en8);  np.save(SRC / f"tokens9{sfx}.npy", en9); np.save(SRC / f"tokens11{sfx}.npy", en11)
    np.save(SRC / f"attr{sfx}.npy", ATTR);    np.save(SRC / f"size{sfx}.npy", SZ);     np.save(SRC / f"repid{sfx}.npy", REP)
    print(f"{sp}: N={N} R={R} -> tokens8{sfx} {en8.shape} | pooled present/sample median {np.median(MASK.sum(1)):.0f} "
          f"(single-profile cap {mi.MAX_SEQ}, pooled cap {MAXR})")


if __name__ == "__main__":
    rng = np.random.default_rng(int(os.environ.get("STR_SEED", "42")))
    print(f"building R={R} replicates into {SRC}  (MAXR={MAXR}, splits={SPLITS}, limit={LIMIT or 'all'})")
    for sp in SPLITS:
        sp = sp.strip()
        if not (SRC / f"y_{sp}_set.npy").exists() or not (SRC / f"phi_{sp}.npy").exists():
            print(f"{sp}: skipped (no y_{sp}_set/phi_{sp} in {SRC})"); continue
        build_split(sp, rng)

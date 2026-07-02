"""gen_lupi.py — leak-safe PEAK-MODEL regeneration (with drop-in) + PRIVILEGED LUPI labels.
TRAIN = real single-source N1 (kept, sentinel labels) + synthetic N2-5 (peak model, with labels:
  beta_ (degradation), mu_ (clean height), var_ (heteroscedastic variance), dropin_ (phantom mask)).
Excludes the REAL val/test donor-combos (NOT make_insilico's hardcoded synthetic ones) -> no leak.
VAL/TEST = copied real (sentinel labels). Run `python features/enrich.py data_lupi` afterwards."""
import os, json, shutil
import numpy as np
from pathlib import Path
import make_insilico as mk
mk.DROPIN = 1   # LUPI track KEEPS drop-in (make_insilico default is now opt-in/0 for inc22 bit-identity)
# Datapath convention (consistent with make_insilico + the driver's INSILICO_W):
#   STR_DATA_DIR = real source (donor_geno/size/real val-test), default "data"
#   STR_OUT_DIR  = generated dataset out, default "data_lupi" -> point INSILICO_W=$STR_OUT_DIR for training
D = Path(os.environ.get("STR_DATA_DIR", "data")); OUT = Path(os.environ.get("STR_OUT_DIR", "data_lupi"))
OUT.mkdir(parents=True, exist_ok=True)
MAXSEQ = mk.MAX_SEQ; NF = mk.N_FLAT; K = 45
N_SYNTH = int(os.environ.get("N_SYNTH", "45000"))
rng = np.random.default_rng(42)

def real_combos(sp):
    y = np.load(D / f"y_{sp}_set.npy"); return {tuple(sorted(np.where(y[i] > 0.5)[0].tolist())) for i in range(len(y))}
leak = {c for c in (real_combos("val") | real_combos("test")) if len(c) >= 2}
print(f"leak combos excluded (real val+test): {sorted(leak)}")
bin_size = mk.build_bin_size()

def tok_lab(xflat, attr_bin, mu_bin, var_bin, dropin_bin):
    nz = np.where(xflat > 0)[0]
    if len(nz) > MAXSEQ: nz = nz[np.argsort(xflat[nz])[::-1][:MAXSEQ]]
    tok = np.zeros((MAXSEQ, 3), np.float32); mask = np.zeros(MAXSEQ, bool); attr = np.full(MAXSEQ, -1, np.int16)
    size = np.zeros(MAXSEQ, np.float32); mu = np.zeros(MAXSEQ, np.float32); var = np.zeros(MAXSEQ, np.float32); di = np.zeros(MAXSEQ, np.float32)
    for i, j in enumerate(nz):
        tok[i] = [mk.BIN_LOCUS[j], mk.BIN_ALLELE[j], xflat[j]]; mask[i] = True; attr[i] = attr_bin[j]
        size[i] = bin_size[j] if bin_size is not None else 0.0
        mu[i] = mu_bin[j]; var[i] = var_bin[j]; di[i] = 1.0 if dropin_bin[j] else 0.0
    return tok, mask, attr, size, mu, var, di

# ---- synthetic N2-5 with privileged labels ----
TOK=[];MASK=[];Y=[];NOC=[];ATTR=[];PHI=[];SIZE=[];MU=[];VAR=[];DI=[];BETA=[];XF=[]
made = 0
while made < N_SYNTH:
    k = int(rng.integers(2, 6)); cols = tuple(sorted(rng.choice(45, k, replace=False).tolist()))
    if cols in leak: continue
    xf, y, kk, beta, ab, ph, mu_b, var_b, di_b = mk.gen_mixture_peak_labeled(list(cols), rng, bin_size, mode="wide")
    tok, mask, attr, size, mu, var, di = tok_lab(xf, ab, mu_b, var_b, di_b)
    TOK.append(tok); MASK.append(mask); Y.append(y); NOC.append(kk); ATTR.append(attr); PHI.append(ph); XF.append(xf)
    SIZE.append(size); MU.append(mu); VAR.append(var); DI.append(di); BETA.append(beta); made += 1
    if made % 10000 == 0: print(f"  generated {made}/{N_SYNTH}")
print(f"synthetic done: {made}")

# ---- keep real single-source N1 (sentinel privileged labels) ----
ntr = np.load(D / "noc_train.npy"); ss = ntr == 1
tok_r = np.load(D / "tokens_train.npy")[ss].astype(np.float32); mask_r = np.load(D / "mask_train.npy")[ss].astype(bool)
y_r = np.load(D / "y_train_set.npy")[ss].astype(np.float32)
size_r = (np.load(D / "size_train.npy")[ss] if (D / "size_train.npy").exists() else np.zeros(mask_r.shape, np.float32)).astype(np.float32)
xflat_r = np.load(D / "Xflat_train.npy")[ss].astype(np.float32) if (D / "Xflat_train.npy").exists() else None   # keep Xflat consistent
d_ss = y_r.argmax(1).astype(np.int16)
attr_r = np.full(mask_r.shape, -1, np.int16)
for r in range(len(d_ss)): attr_r[r, mask_r[r]] = d_ss[r]
phi_r = np.zeros((len(d_ss), 45), np.float32); phi_r[np.arange(len(d_ss)), d_ss] = 1.0
zero_r = np.zeros(mask_r.shape, np.float32)   # mu/var/dropin sentinel = 0 (masked in LUPI loss via beta=-1)
beta_r = np.full(len(d_ss), -1.0, np.float32)  # sentinel: real row, no degradation label
print(f"real N1 kept: {ss.sum()}")

# ---- concat + shuffle + save TRAIN ----
def C(a, b): return np.concatenate([np.asarray(a), b], 0)
TOK=C(TOK,tok_r).astype(np.float32); MASK=C(MASK,mask_r).astype(bool); Y=C(Y,y_r).astype(np.float32)
NOC=C(NOC,ntr[ss]).astype(np.int32); ATTR=C(ATTR,attr_r).astype(np.int16); PHI=C(PHI,phi_r).astype(np.float32)
SIZE=C(SIZE,size_r).astype(np.float32); MU=C(MU,zero_r).astype(np.float32); VAR=C(VAR,zero_r).astype(np.float32)
DI=C(DI,zero_r).astype(np.float32); BETA=C(BETA,beta_r).astype(np.float32)
perm = rng.permutation(len(TOK))
save = {"tokens_train":TOK,"mask_train":MASK,"y_train_set":Y,"noc_train":NOC,"attr_train":ATTR,
        "phi_train":PHI,"size_train":SIZE,"mu_train":MU,"var_train":VAR,"dropin_train":DI,"beta_train":BETA}
if xflat_r is not None:                                       # Xflat_train consistent (make_dev_split + in-place regen)
    save["Xflat_train"] = C(XF, xflat_r).astype(np.float32)
for name, arr in save.items(): np.save(OUT / f"{name}.npy", arr[perm])
print(f"saved TRAIN: {len(TOK)} ({made} synth + {ss.sum()} real N1)  -> {OUT}")

# ---- copy real val/test (+ meta, geno, open) ----
def _cp(src, dst):   # safe even when D==OUT (Kaggle in-place generate): skip self-copy
    if src.exists() and src.resolve() != dst.resolve():
        shutil.copy(src, dst)
for split in ["val", "test"]:
    for pre in ["Xflat_","y_","noc_","tokens_","mask_","phi_","condition_","condition_level_",
                "meta_template_","meta_qindex_","attr_","size_"]:
        suf = "_set" if pre == "y_" else ""
        src = D / f"{pre}{split}{suf}.npy"
        _cp(src, OUT / src.name)
for f in ["donor_geno.npy","donor_geno_mask.npy","meta_set.json",
          "tokens_open.npy","mask_open.npy","Xflat_open.npy","size_open.npy"]:   # open-set: reject-head training
    _cp(D / f, OUT / f)
print(f"copied real val/test + open + geno/meta. NEXT: python features/enrich.py {OUT}  (then INSILICO_W={OUT})")

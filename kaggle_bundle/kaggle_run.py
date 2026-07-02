"""
kaggle_run.py — Run in-silico scaling experiments on Kaggle/Colab GPU.

Setup (Kaggle):
  1. Upload the `data/` folder (the .npy + meta_set.json) as a Kaggle Dataset, e.g. "noc-data".
  2. Upload the code (*.py + models/) as a Dataset "noc-code"  OR  !git clone your repo.
  3. In a GPU notebook, copy code to /kaggle/working and run:
        !cp -r /kaggle/input/noc-code/* /kaggle/working/
        !SCALE=100000 python /kaggle/working/kaggle_run.py
     (set REAL_DATA if your data path differs from /kaggle/input/noc-data/data)

Env vars:
  REAL_DATA  : path to real data dir (default /kaggle/input/noc-data/data)
  SCALE      : number of in-silico mixtures (default 100000)
  DO_ST      : also train pure ST (default 1)
"""
import os, subprocess, sys, json
from pathlib import Path

WORK = Path(os.environ.get("WORK_DIR", "/kaggle/working"))
REAL = Path(os.environ.get("REAL_DATA", "/kaggle/input/noc-data/data"))
SCALE = int(os.environ.get("SCALE", 100000))
DO_ST = os.environ.get("DO_ST", "1") == "1"
PY = sys.executable
INSILICO = WORK / "data_insilico"

assert (REAL / "meta_set.json").exists(), f"Real data not found at {REAL} — set REAL_DATA"
print(f"Real data: {REAL} | in-silico out: {INSILICO} | SCALE={SCALE} | device check below")
subprocess.run([PY, "-c", "import torch;print('CUDA:',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"])


def run(cmd, extra_env):
    e = os.environ.copy(); e.update(extra_env)
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], env=e, check=True, cwd=str(WORK))


# 1) Generate in-silico version B (excludes test+val combos = generalization) from REAL single-source
run([PY, "make_insilico.py", "--build", SCALE],
    {"STR_DATA_DIR": str(REAL), "STR_OUT_DIR": str(INSILICO)})

# 2) Train on in-silico (val/test = real, copied into data_insilico by make_insilico)
train_env = {"STR_DATA_DIR": str(INSILICO)}
run([PY, "train_hybrid.py", "--loss", "asl", "--decouple_reject", "--out_subdir", f"hybrid_B_{SCALE}"], train_env)
if DO_ST:
    run([PY, "train_set_transformer.py", "--out_subdir", f"st_B_{SCALE}"], train_env)

# 3) Compact eval on REAL test: oracle top-k + donor-48 probe
import numpy as np, torch
sys.path.insert(0, str(WORK))
from models.set_transformer import SetTransformerMixture
DEV = "cuda" if torch.cuda.is_available() else "cpu"
meta = json.load(open(REAL / "meta_set.json")); col = {d: i for i, d in enumerate(meta["known_donors"])}
yt = np.load(REAL/"y_test_set.npy"); noc = np.load(REAL/"noc_test.npy")
t = torch.from_numpy(np.load(REAL/"tokens_test.npy")); msk = torch.from_numpy(np.load(REAL/"mask_test.npy"))
xf = torch.from_numpy(np.load(REAL/"Xflat_test.npy").astype("float32"))


def probs(subdir, hybrid):
    m = SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32,
                              n_classes=45, n_noc=6, dropout=0.1,
                              cls_decoder="hybrid" if hybrid else "pooled", n_flat=590,
                              decouple_reject=hybrid).to(DEV)
    m.load_state_dict(torch.load(WORK/"results"/subdir/"best_model.pt", weights_only=True)); m.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(t), 256):
            a = (t[i:i+256].to(DEV), msk[i:i+256].to(DEV)) + ((xf[i:i+256].to(DEV),) if hybrid else ())
            P.append(torch.sigmoid(m(*a)["logits_cls"]).cpu().numpy())
    return np.concatenate(P)


def report(P, tag):
    idx = np.where(noc == 3)[0]
    yp = np.zeros_like(P, dtype=int)
    for i in range(len(P)):
        k = int(noc[i]); yp[i, np.argsort(P[i])[::-1][:k]] = 1
    e = (yt == yp).all(1)
    per = [e[noc == j].mean() if (noc == j).sum() else float("nan") for j in range(1, 6)]
    print(f"  {tag:<18} oracle={e.mean():.3f}  per-NOC={[round(x,3) for x in per]}  donor-48={P[idx][:,col[48]].mean():.2f}")


print("\n=== REAL test (oracle top-k) ===")
report(probs(f"hybrid_B_{SCALE}", True), f"hybrid B {SCALE}")
if DO_ST:
    report(probs(f"st_B_{SCALE}", False), f"pure ST B {SCALE}")
print("\nDone. Checkpoints in", WORK/"results")

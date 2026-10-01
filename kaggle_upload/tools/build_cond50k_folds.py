"""
build_cond50k_folds.py — sinh bo du lieu 50k --conditioned cho tung fold, MOT may, MOT cong thuc.

Nguon generator: bundle cua nhom incoming/noc_bundle_2026-09-30/code_ho (KHONG sua generator). Ban sao lam viec
chi khac ban goc o hai cho, va ca hai deu in ra trong build_info.json:
  1. preprocess.py doc so hon hop tu STR_BUILD_N (mac dinh van 50000).
  2. them buoc calibrate_crowd.py (TRAIN NOC1 -> data/noise_cr_inc.json) truoc make_insilico, vi RUN_10FOLD.md 7.1
     liet ke file nay la mot luat cua generator; calibrate_crowd.py doi duong dan Windows cung thanh thu muc cua no.
Moi fold: STR_FOLD=F preprocess.py (csv -> data/ -> make_insilico --build 50000 --noc_weights 1,1.5,2.5,2 --seed 42
--conditioned), kiem ro ri, roi dong goi kaggle_upload/team_10fold/noc-cond50k-fold<F>.zip (thu muc data_cond50k_fold<F>/)
va ghi md5 vao kaggle_upload/team_10fold/cond50k_manifest.json. Generator tat dinh tren cung may (kiem 01/10: cung md5 qua PYTHONHASHSEED).

usage:  python3 kaggle_upload/tools/build_cond50k_folds.py            # ca 10 fold
        python3 kaggle_upload/tools/build_cond50k_folds.py 0 3 7      # chi cac fold nay
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "incoming" / "noc_bundle_2026-09-30" / "code_ho"
WORK = ROOT / "work" / "cond50k_build"
CODE = WORK / "code"
UPLOAD = ROOT / "kaggle_upload" / "team_10fold"      # every file the team uploads lives here
MANIFEST = UPLOAD / "cond50k_manifest.json"
N_MIX = 50000


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prepare_code():
    """Ban sao bundle + hai cho va. Lam lai moi lan de chac chan khong con du lieu fold truoc."""
    if CODE.exists():
        shutil.rmtree(CODE)
    shutil.copytree(BUNDLE, CODE)
    os.symlink(ROOT / "data_raw", CODE / "data_raw")
    p = CODE / "calibrate_crowd.py"; s = p.read_text()
    old = 'CODE = Path(r"C:\\Tailieu\\TinSinh\\Project\\new_NOC\\code")'
    assert s.count(old) == 1, "calibrate_crowd.py: khong thay duong dan Windows can thay"
    p.write_text(s.replace(old, "CODE = Path(__file__).resolve().parent"))
    p = CODE / "preprocess.py"; s = p.read_text()
    old1 = '    step("make_insilico.py",'
    old2 = 'args=("--build", "50000",'
    assert s.count(old1) == 1 and s.count(old2) == 1, "preprocess.py: khong thay cho can va"
    s = s.replace(old1, '    step("calibrate_crowd.py")                                        '
                        '# TRAIN NOC1 -> data/noise_cr_inc.json\n' + old1)
    s = s.replace(old2, 'args=("--build", os.environ.get("STR_BUILD_N", "50000"),')
    p.write_text(s)


def audit(d, fold):
    fi = json.load(open(d / "fold_info.json"))
    assert fi["fold"] == fold, f"fold_info noi fold {fi['fold']}, can {fold}"
    noc = np.load(d / "noc_train.npy"); y = np.load(d / "y_train_set.npy")
    yt = np.load(d / "y_test_set.npy"); nt = np.load(d / "noc_test.npy")
    assert int((noc >= 2).sum()) == N_MIX, f"{int((noc >= 2).sum())} hon hop, can {N_MIX}"
    tr = {tuple(np.flatnonzero(r)) for r in y[noc >= 2]}
    te = {tuple(np.flatnonzero(r)) for r in yt[nt >= 2]}
    shared = len(tr & te)
    assert shared == 0, f"RO RI: {shared} to hop test nam trong train sinh"
    return fi, noc, len(tr), len(te)


def build(fold):
    t0 = time.time()
    for sub in ("data", "data_insilico_w"):
        if (CODE / sub).exists():
            shutil.rmtree(CODE / sub)
    log = WORK / f"build_fold{fold}.log"
    env = dict(os.environ, STR_FOLD=str(fold), STR_BUILD_N=str(N_MIX), PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    with open(log, "w") as lf:
        r = subprocess.run([sys.executable, "preprocess.py"], cwd=CODE, env=env, stdout=lf, stderr=subprocess.STDOUT)
    assert r.returncode == 0, f"fold {fold}: preprocess that bai, xem {log}"
    src = CODE / "data_insilico_w"
    fi, noc, n_tr, n_te = audit(src, fold)
    name = f"data_cond50k_fold{fold}"
    info = {
        "fold": fold, "unknown_donors": fi["unknown_donors"], "n_mix": N_MIX, "n_single_source": int((noc == 1).sum()),
        "noc_train_counts": np.bincount(noc, minlength=6)[1:].tolist(),
        "train_combos": n_tr, "test_combos": n_te, "shared_combos": 0,
        "md5": {f: md5(src / f) for f in ("noc_train.npy", "Xflat_train.npy", "tokens_train.npy", "noc_test.npy",
                                          "tokens_test.npy")},
        "generator": {"bundle": str(BUNDLE.relative_to(ROOT)),
                      "bundle_md5": {f: md5(BUNDLE / f) for f in ("make_insilico.py", "calibrate.py", "preprocess.py",
                                                                  "calibrate_crowd.py", "features/enrich.py")},
                      "command": f"STR_FOLD={fold} STR_BUILD_N={N_MIX} python preprocess.py "
                                 "(-> make_insilico --build 50000 --noc_weights 1,1.5,2.5,2 --seed 42 --conditioned)",
                      "patches": ["preprocess.py: so hon hop tu STR_BUILD_N",
                                  "preprocess.py: them buoc calibrate_crowd.py truoc make_insilico",
                                  "calibrate_crowd.py: duong dan Windows -> thu muc cua file"]},
        "env": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(info, open(src / "build_info.json", "w"), indent=1)
    zp = UPLOAD / f"noc-cond50k-fold{fold}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.iterdir()):
            z.write(f, f"{name}/{f.name}")
    shutil.copy(log, UPLOAD / f"noc-cond50k-fold{fold}.build.log")
    shutil.rmtree(src); shutil.rmtree(CODE / "data")
    man = json.load(open(MANIFEST)) if MANIFEST.exists() else {}
    man[str(fold)] = {"zip": zp.name, "dir": name, "noc_train_md5": info["md5"]["noc_train.npy"],
                      "unknown_donors": info["unknown_donors"], "noc_train_counts": info["noc_train_counts"],
                      "zip_mb": round(zp.stat().st_size / 1e6, 1)}
    json.dump(dict(sorted(man.items(), key=lambda kv: int(kv[0]))), open(MANIFEST, "w"), indent=1)
    print(f"fold {fold}: unknown {info['unknown_donors']} | train NOC {info['noc_train_counts']} | "
          f"shared 0 | {zp.name} {zp.stat().st_size / 1e6:.0f} MB | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    folds = [int(a) for a in sys.argv[1:]] or list(range(10))
    WORK.mkdir(parents=True, exist_ok=True); UPLOAD.mkdir(parents=True, exist_ok=True)
    prepare_code()
    for f in folds:
        build(f)
    print("DONE", folds)

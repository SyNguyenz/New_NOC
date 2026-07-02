"""
train_noc_tabular.py — NOC counting as TABULAR classification on combo-invariant
engineered features (forensic MAC-style), per deepNoC/PACE literature.

Core idea: the embedding NOC head learned combo-RECOGNITION (leaky 0.99 artifact,
no-leak ~0.3). Counting should use combo-INVARIANT allele-count + height features,
which generalize to novel combos. Train XGBoost + TabPFN, plug predicted NOC into
the hybrid ASL-decoupled ranking, report decoded Exact Match.

Features (per sample, all combo-invariant — independent of WHICH donors):
  for RFU threshold in {0,50,100,150,250,500}:
     MAC (max alleles/locus), total alleles, n_loci>=2/3/4, mean alleles/active-locus
  height stats: RFU percentiles, mean, std, log-height std (mixture skew)
  n_active_loci

Usage:
  python train_noc_tabular.py              # XGB only (fast)
  python train_noc_tabular.py --tabpfn     # also TabPFN (subsampled)
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data")))

# load TABPFN_TOKEN from .env
_env = ROOT / ".env"
if _env.exists() and not os.environ.get("TABPFN_TOKEN"):
    for ln in _env.read_text().splitlines():
        ln = ln.strip()
        if ln.startswith("TABPFN_TOKEN=") and not ln.startswith("#"):
            os.environ["TABPFN_TOKEN"] = ln.split("=", 1)[1].strip()

THRESHOLDS = [0, 50, 100, 150, 250, 500]


def featurize(split: str) -> np.ndarray:
    tok = np.load(DATA / f"tokens_{split}.npy")
    msk = np.load(DATA / f"mask_{split}.npy")
    h_rfu = np.expm1(tok[:, :, 2])
    N = len(tok)
    feats = []
    for i in range(N):
        valid0 = msk[i]
        loci_all = tok[i, :, 0].astype(int)
        row = []
        for thr in THRESHOLDS:
            v = valid0 & (h_rfu[i] > thr)
            loci = loci_all[v]
            if len(loci):
                counts = np.bincount(loci, minlength=24)
                nz = counts[counts > 0]
                row += [counts.max(), len(loci), int((counts >= 2).sum()),
                        int((counts >= 3).sum()), int((counts >= 4).sum()),
                        float(nz.mean())]
            else:
                row += [0, 0, 0, 0, 0, 0.0]
        # height stats (valid alleles)
        hv = h_rfu[i][valid0]
        if len(hv):
            row += list(np.percentile(hv, [10, 25, 50, 75, 90]))
            row += [hv.mean(), hv.std(), hv.max(),
                    float(np.std(np.log1p(hv)))]
        else:
            row += [0]*9
        row += [int(np.unique(loci_all[valid0]).size)]  # active loci
        feats.append(row)
    return np.array(feats, dtype=np.float32)


def per_noc(pred, true):
    acc = (pred == true).mean()
    w1 = (np.abs(pred - true) <= 1).mean()
    out = [acc, w1]
    for k in range(1, 6):
        m = true == k
        out.append((pred[m] == k).mean() if m.sum() else float("nan"))
    return out


def downstream_em(k_arr):
    """Plug NOC estimate into hybrid ASL-decoupled ranking -> Exact Match."""
    import torch, sys
    sys.path.insert(0, str(ROOT))
    from models.set_transformer import SetTransformerMixture
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    m = SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2,
                              m_inducing=32, n_classes=45, n_noc=6, dropout=0.1,
                              cls_decoder="hybrid", n_flat=590, decouple_reject=True).to(DEV)
    ckpt = ROOT / "results" / "set_transformer_hybrid_asl_decoup" / "best_model.pt"
    m.load_state_dict(torch.load(ckpt, weights_only=True)); m.eval()
    t = torch.from_numpy(np.load(DATA/"tokens_test.npy")); msk = torch.from_numpy(np.load(DATA/"mask_test.npy"))
    xf = torch.from_numpy(np.load(DATA/"Xflat_test.npy").astype(np.float32))
    P = []
    with torch.no_grad():
        for i in range(0, len(t), 256):
            P.append(torch.sigmoid(m(t[i:i+256].to(DEV), msk[i:i+256].to(DEV), xf[i:i+256].to(DEV))["logits_cls"]).cpu().numpy())
    P = np.concatenate(P)
    yt = np.load(DATA/"y_test_set.npy"); noc = np.load(DATA/"noc_test.npy")
    yp = np.zeros_like(P, dtype=int)
    for i in range(len(P)):
        k = int(max(1, min(5, round(k_arr[i])))); yp[i, np.argsort(P[i])[::-1][:k]] = 1
    em = (yt == yp).all(1)
    return [em.mean()] + [em[noc == j].mean() if (noc == j).sum() else float("nan") for j in range(1, 6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabpfn", action="store_true")
    ap.add_argument("--tabpfn_n", type=int, default=3000, help="TabPFN train subsample")
    args = ap.parse_args()

    print("Featurizing..."); t0 = time.time()
    Xtr, Xva, Xte = featurize("train"), featurize("val"), featurize("test")
    ntr = np.load(DATA/"noc_train.npy"); nva = np.load(DATA/"noc_val.npy"); nte = np.load(DATA/"noc_test.npy")
    print(f"  feats {Xtr.shape[1]} dims, {time.time()-t0:.0f}s")
    Xtv = np.concatenate([Xtr, Xva]); ntv = np.concatenate([ntr, nva])

    hdr = f"  {'model':<16}{'acc':>6}{'w1':>6}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}"

    # ---- XGBoost ----
    import xgboost as xgb
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                            random_state=42)
    clf.fit(Xtv, ntv - 1)  # labels 1-5 -> 0-4
    k_xgb = clf.predict(Xte) + 1
    print("\n=== NOC count accuracy (test) ===")
    print(hdr)
    print(f"  {'XGBoost':<16}" + "".join(f"{x:>7.3f}" if i > 1 else f"{x:>6.3f}" for i, x in enumerate(per_noc(k_xgb, nte))))

    results = {"xgb_count": per_noc(k_xgb, nte)}

    # ---- TabPFN (optional) ----
    k_tab = None
    if args.tabpfn:
        from tabpfn import TabPFNClassifier
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Xtv), size=min(args.tabpfn_n, len(Xtv)), replace=False)
        print(f"\nTabPFN fit on {len(idx)} subsample...")
        t1 = time.time()
        tp = TabPFNClassifier(n_estimators=8, ignore_pretraining_limits=True, random_state=42, show_progress_bar=False)
        tp.fit(Xtv[idx], ntv[idx] - 1)
        k_tab = tp.predict(Xte) + 1
        print(f"  TabPFN done {time.time()-t1:.0f}s")
        print(f"  {'TabPFN':<16}" + "".join(f"{x:>7.3f}" if i > 1 else f"{x:>6.3f}" for i, x in enumerate(per_noc(k_tab, nte))))
        results["tabpfn_count"] = per_noc(k_tab, nte)

    # ---- Downstream EM on hybrid ranking ----
    print("\n=== Downstream Exact Match (hybrid ASL-decoupled ranking + tabular NOC) ===")
    print(f"  {'k-source':<22}{'overall':>8}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
    print(f"  {'oracle (true NOC)':<22}" + "".join(f"{x:>7.3f}" for x in downstream_em(nte.astype(float))))
    print(f"  {'XGB NOC':<22}" + "".join(f"{x:>7.3f}" for x in downstream_em(k_xgb.astype(float))))
    if k_tab is not None:
        print(f"  {'TabPFN NOC':<22}" + "".join(f"{x:>7.3f}" for x in downstream_em(k_tab.astype(float))))

    (ROOT/"results"/"noc_tabular").mkdir(parents=True, exist_ok=True)
    json.dump({k: [float(x) for x in v] for k, v in results.items()},
              open(ROOT/"results"/"noc_tabular"/"metrics.json", "w"), indent=2)
    print("\nSaved -> results/noc_tabular/metrics.json")


if __name__ == "__main__":
    main()

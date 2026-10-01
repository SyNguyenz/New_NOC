"""Pool the label-budget (371) fine-tune runs of the 10-fold campaign.

report_folds.py pools the zero-shot report of every fold (fold<F>_seed42.json). The fine-tune runs live in
budget_fold<F>_d<D>.metrics.json (one per fold and budget draw, every head arm inside). This script reads them
and prints, per arm:
  - one row per fold (mean over draws)
  - mean ± sd over folds
  - the POOLED strict macro F1 (confusion matrices summed over folds, per draw, then averaged over draws)
  - the deepNoC-protocol macro F1 (mean; that protocol stores no confusion matrix)
--exclude_selection drops fold 0, the fold the fine-tune recipe was chosen on (hypothesis T5).

usage:  python report_budget.py send_all/ [--exclude_selection] [--out report_budget.md]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--exclude_selection", action="store_true", help="leave out fold 0 (the selection fold)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

files = []
for d in a.dirs:
    files += glob.glob(os.path.join(d, "**", "budget_fold*_d*.metrics.json"), recursive=True)
rx = re.compile(r"budget_fold(\d+)_d(\d+)\.metrics\.json$")
runs = defaultdict(dict)                       # fold -> draw -> count_arms
for f in sorted(files):
    m = rx.search(f)
    fo, dr = int(m.group(1)), int(m.group(2))
    if a.exclude_selection and fo == 0:
        continue
    runs[fo][dr] = json.load(open(f))["count_arms"]
assert runs, f"no budget_fold*_d*.metrics.json under {a.dirs}"
arms = sorted(set.intersection(*[set(c) for fo in runs.values() for c in fo.values()]))
folds = sorted(runs)
draws = sorted(set.intersection(*[set(v) for v in runs.values()]))


def f1_from_conf(C):
    C = np.asarray(C, float); tp = np.diag(C)
    p = tp / np.maximum(C.sum(0), 1); r = tp / np.maximum(C.sum(1), 1)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return f.mean(), f


L = [f"# Fine-tune 371 nhan — {len(folds)} fold {folds}, {len(draws)} lan rut {draws}"
     + (" (bo fold 0 = fold chon)" if a.exclude_selection else ""), ""]
L += ["| cach | strict mean ± sd (fold) | strict GOP | deepNoC protocol mean ± sd | F1 NOC1..5 gop |",
      "|---|---|---|---|---|"]
per_fold = {}
for arm in arms:
    s = np.array([np.mean([runs[fo][d][arm]["macro_f1"] for d in draws]) for fo in folds])
    dn = np.array([np.mean([(runs[fo][d][arm].get("deepnoc") or {}).get("macro_f1", np.nan) for d in draws])
                   for fo in folds])
    pooled, f5 = [], []
    for d in draws:
        C = sum(np.asarray(runs[fo][d][arm]["confusion_rows_true_cols_pred"], float) for fo in folds)
        m_, f_ = f1_from_conf(C); pooled.append(m_); f5.append(f_)
    per_fold[arm] = s
    L.append(f"| {arm} | {s.mean():.4f} ± {s.std(ddof=1) if len(s) > 1 else 0:.4f} | {np.mean(pooled):.4f} "
             f"| {np.nanmean(dn):.4f} ± {np.nanstd(dn, ddof=1) if len(dn) > 1 else 0:.4f} | "
             + " ".join(f"{v:.3f}" for v in np.mean(f5, 0)) + " |")
L += ["", "Moc deepNoC: macro F1 0.9097 (371 nhan, F1 cua Table 2).", "",
      "| fold | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
for i, fo in enumerate(folds):
    L.append(f"| {fo} | " + " | ".join(f"{per_fold[arm][i]:.4f}" for arm in arms) + " |")
txt = "\n".join(L)
print(txt)
if a.out:
    open(a.out, "w").write(txt + "\n")
    print(f"\n-> {a.out}")

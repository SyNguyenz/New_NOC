"""Pool per-fold NoC + ID reports into one report.

Every run of train_set_transformer.py ends by writing reports/fold<F>_seed<S>.json
(schema noc_id_fold_report/v1): the count (NoC) and the contributor ID of that archive fold, from the
same run and the same decoder. This script reads any number of them and writes one report with
  1. provenance and consistency: same code, count head and recipe in every fold, which folds are missing
  2. one row per fold
  3. mean ± sd over folds AND pooled numbers. The folds hold different numbers of profiles per NoC,
     so pooled NoC metrics sum the confusion matrices and pooled EM weights each NoC by its n.

usage:  python report_folds.py reports/ [more dirs or files] [--out report_folds.md] [--csv folds.csv]
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

SCHEMA = "noc_id_fold_report/v1"
NOCS = ("1", "2", "3", "4", "5")


def load(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    reps = []
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        if r.get("schema") != SCHEMA:
            print(f"skip {f}: schema {r.get('schema')!r}")
            continue
        r["_file"] = f
        reps.append(r)
    return sorted(reps, key=lambda r: (99 if r.get("fold") is None else r["fold"], r.get("seed", 0)))


def fmt(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def mean_sd(vals, d=3):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0:
        return "—"
    return f"{v.mean():.{d}f}" if len(v) == 1 else f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


def f1_from_cm(cm):
    out = {}
    for i, k in enumerate(NOCS):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        out[k] = {"f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
                  "recall": tp / (tp + fn) if (tp + fn) else None,
                  "precision": tp / (tp + fp) if (tp + fp) else None,
                  "n": int(cm[i].sum())}
    return out


def pooled_em(reps, key):
    """EM pooled over folds: each NoC rate weighted by that fold's n for the NoC."""
    num = {k: 0.0 for k in NOCS}; den = {k: 0 for k in NOCS}
    for r in reps:
        for k in NOCS:
            rate = r["id"][key].get(k); n = r["data"]["per_noc_n"].get(k, 0)
            if rate is not None and n:
                num[k] += rate * n; den[k] += n
    per = {k: (num[k] / den[k] if den[k] else None) for k in NOCS}
    tot = sum(den.values())
    return (sum(num.values()) / tot if tot else None), per


def checks_ok(r):
    c = r["checks"]
    ok = bool(c.get("params_unchanged"))
    ok &= c.get("split_audit_shared_combos") in (0, None)
    if r["noc"]["source"] == "noc_lora":
        ok &= bool(c.get("lora_folds_combination_disjoint"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="report JSON files or directories holding them")
    ap.add_argument("--out", default="report_folds.md", help="markdown output")
    ap.add_argument("--csv", default=None, help="optional per-fold CSV")
    args = ap.parse_args()

    reps = load(args.paths)
    if not reps:
        sys.exit("no reports found")
    L = []
    say = L.append

    # 1. provenance + consistency
    say("# NoC + ID — gộp theo fold\n")
    recipe = lambda r: (r.get("code"), r["noc"]["source"],
                        json.dumps({k: (r["noc"].get("lora") or {}).get(k) for k in ("kfold", "steps_per_fit", "rank")}),
                        r.get("protocol"))
    recipes = sorted({recipe(r) for r in reps})
    n_folds = max((r.get("n_folds") or 10) for r in reps)
    have = sorted({r["fold"] for r in reps if r.get("fold") is not None})
    missing = [f for f in range(n_folds) if f not in have]
    say(f"- Số báo cáo: **{len(reps)}** | fold có: {have} | fold thiếu: {missing if missing else 'không'}")
    say(f"- Công thức (code, nguồn đếm, LoRA kfold/steps/rank, protocol): "
        + ("**giống nhau ở mọi fold**" if len(recipes) == 1 else f"**KHÁC NHAU giữa các fold** — {recipes}"))
    for rc in recipes:
        say(f"  - code `{rc[0]}` · count `{rc[1]}` · LoRA {rc[2]} · protocol `{rc[3]}`")
    dup = [f for f in have if sum(1 for r in reps if r.get("fold") == f) > 1]
    if dup:
        say(f"- Cảnh báo: nhiều báo cáo cho cùng fold {dup} (nhiều seed) — mỗi báo cáo tính là một dòng")
    bad = [r["fold"] for r in reps if not checks_ok(r)]
    say(f"- Kiểm tra (params_unchanged, split audit 0 combo trùng, LoRA tách tổ hợp): "
        + ("**đạt ở mọi fold**" if not bad else f"**KHÔNG đạt ở fold {bad}**"))
    say("")

    # 2. per-fold rows
    say("## Từng fold\n")
    head = (["fold", "unknown donors", "n (N1/N2/N3/N4/N5)", "NoC acc", "NoC macroF1"]
            + [f"F1 N{k}" for k in NOCS] + ["oracle EM", "EM@k", "donor F1@k", "reject AUROC", "checks"])
    say("| " + " | ".join(head) + " |")
    say("|" + "---|" * len(head))
    rows = []
    for r in reps:
        n = r["data"]["per_noc_n"]
        row = [str(r.get("fold")), ",".join(str(d) for d in (r.get("unknown_donors") or [])),
               f"{r['data']['eval_n']} ({'/'.join(str(n.get(k, 0)) for k in NOCS)})",
               fmt(r["noc"]["accuracy"]), fmt(r["noc"]["macro_f1"])]
        row += [fmt(r["noc"]["per_noc"][k]["f1"]) for k in NOCS]
        row += [fmt(r["id"]["oracle_em"]["overall"]), fmt(r["id"]["em_at_pred_k"]["overall"]),
                fmt(r["id"]["donor_at_pred_k"]["overall"]["f1"]), fmt(r["id"]["reject_auroc"]),
                "OK" if checks_ok(r) else "LỖI"]
        say("| " + " | ".join(row) + " |")
        rows.append(dict(zip(head, row)))
    say("")

    # 3. summary: mean ± sd over folds and pooled
    cm = np.sum([np.array(r["noc"]["confusion_rows_true_cols_pred"]) for r in reps], axis=0)
    pf = f1_from_cm(cm)
    pooled_macro = np.mean([pf[k]["f1"] for k in NOCS if pf[k]["f1"] is not None])
    em_o, em_o_per = pooled_em(reps, "oracle_em")
    em_k, em_k_per = pooled_em(reps, "em_at_pred_k")
    say("## Tổng hợp\n")
    say("| chỉ số | trung bình ± sd qua fold | gộp (cộng dồn hồ sơ) |")
    say("|---|---|---|")
    say(f"| NoC accuracy | {mean_sd([r['noc']['accuracy'] for r in reps])} | {np.trace(cm) / cm.sum():.3f} |")
    say(f"| NoC macro F1 | {mean_sd([r['noc']['macro_f1'] for r in reps])} | {pooled_macro:.3f} |")
    for k in NOCS:
        say(f"| NoC F1 N{k} (n gộp {pf[k]['n']}) | {mean_sd([r['noc']['per_noc'][k]['f1'] for r in reps])} "
            f"| {fmt(pf[k]['f1'])} (P {fmt(pf[k]['precision'])} · R {fmt(pf[k]['recall'])}) |")
    say(f"| ID oracle EM | {mean_sd([r['id']['oracle_em']['overall'] for r in reps])} | {fmt(em_o)} |")
    say(f"| ID EM@k (k = số người dự đoán) | {mean_sd([r['id']['em_at_pred_k']['overall'] for r in reps])} | {fmt(em_k)} |")
    for k in NOCS:
        say(f"| ID EM@k N{k} | {mean_sd([r['id']['em_at_pred_k'][k] for r in reps])} | {fmt(em_k_per[k])} |")
    say(f"| donor precision @k | {mean_sd([r['id']['donor_at_pred_k']['overall']['precision'] for r in reps])} | — |")
    say(f"| donor recall @k | {mean_sd([r['id']['donor_at_pred_k']['overall']['recall'] for r in reps])} | — |")
    say(f"| donor F1 @k | {mean_sd([r['id']['donor_at_pred_k']['overall']['f1'] for r in reps])} | — |")
    say(f"| reject AUROC | {mean_sd([r['id']['reject_auroc'] for r in reps])} | — |")
    say("")
    say("## Ma trận nhầm lẫn NoC gộp (hàng = thật, cột = dự đoán)\n")
    say("| thật \\ dự đoán | " + " | ".join(f"N{k}" for k in NOCS) + " |")
    say("|" + "---|" * 6)
    for i, k in enumerate(NOCS):
        say(f"| N{k} | " + " | ".join(str(int(x)) for x in cm[i]) + " |")
    say("")
    say("Nguồn: " + ", ".join(os.path.basename(r["_file"]) for r in reps))

    text = "\n".join(L)
    print(text)
    with open(args.out, "w") as fh:
        fh.write(text + "\n")
    print(f"\n-> {args.out}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=head)
            w.writeheader(); w.writerows(rows)
        print(f"-> {args.csv}")


if __name__ == "__main__":
    main()

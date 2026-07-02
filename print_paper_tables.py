"""
Print all results as paper-ready tables (markdown).
Reads from results/ folder — run after ablation + multi-seed complete.

Usage: python print_paper_tables.py [--save]
  --save : also write tables to reports/paper_tables.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_json(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def fmt(v, decimals=4):
    if v is None: return "N/A"
    return f"{float(v):.{decimals}f}"


def print_table(headers, rows, caption=""):
    if caption:
        print(f"\n**{caption}**\n")
    widths = [max(len(h), max(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    print(hdr); print(sep)
    for row in rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("# Paper Tables - PROVEDIt 45-class STR Contributor Identification")
    out(f"Generated from: {RESULTS}")

    # ── Table 1: Baseline comparison ──────────────────────────────────────
    out("\n## Table 1: Main results (test n=1,325, GF29cycles)")
    models = {
        "Set Transformer": RESULTS / "set_transformer" / "metrics.json",
        "XGBoost":         RESULTS / "baseline_xgb" / "metrics.json",
        "MLP":             RESULTS / "mlp" / "metrics.json",
        "LR":              RESULTS / "baseline_lr" / "metrics.json",
        "CNN":             RESULTS / "cnn" / "metrics.json",
        "kNN":             RESULTS / "baseline_knn" / "metrics.json",
    }
    rows = []
    for name, path in models.items():
        d = load_json(path)
        if d is None: continue
        t = d.get("test", d)
        auc = d.get("reject_auroc")
        rows.append([
            name,
            fmt(t.get("macro_f1")),
            fmt(t.get("micro_f1")),
            fmt(t.get("exact_match")),
            fmt(t.get("hamming")),
            fmt(auc) if auc else "—",
        ])

    headers = ["Model", "Macro F1", "Micro F1", "Exact Match", "Hamming", "Reject AUROC"]
    out()
    w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-"*wi for wi in w) + " |"
    out("| " + " | ".join(h.ljust(wi) for h, wi in zip(headers, w)) + " |")
    out(sep)
    for row in rows:
        out("| " + " | ".join(str(c).ljust(wi) for c, wi in zip(row, w)) + " |")

    # ── Table 2: Ablation ────────────────────────────────────────────────
    out("\n## Table 2: Ablation study (60 epochs, same split)")
    ablation_variants = {
        "Full (ISAB+PMA, 3 heads)": "ablation_full",
        "Deep Sets (mean pool)":   "ablation_deep_sets",
        "MLP encoder (flat 590)":  "ablation_mlp_enc",
        "No NOC head":             "ablation_no_noc",
        "No reject head":          "ablation_no_reject",
    }
    a_rows = []
    for name, folder in ablation_variants.items():
        d = load_json(RESULTS / folder / "metrics.json")
        if d is None: a_rows.append([name, "—", "—", "—", "—", "—"]); continue
        t = d.get("test", d)
        auc = d.get("reject_auroc")
        a_rows.append([
            name,
            fmt(t.get("macro_f1")),
            fmt(t.get("micro_f1")),
            fmt(t.get("exact_match")),
            fmt(t.get("hamming")),
            fmt(auc) if auc else "—",
        ])

    out()
    w = [max(len(h), max(len(r[i]) for r in a_rows))
         for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-"*wi for wi in w) + " |"
    out("| " + " | ".join(h.ljust(wi) for h, wi in zip(headers, w)) + " |")
    out(sep)
    for row in a_rows:
        out("| " + " | ".join(str(c).ljust(wi) for c, wi in zip(row, w)) + " |")

    # ── Table 3: Open-set scoring ────────────────────────────────────────
    out("\n## Table 3: Open-set AUROC comparison (1,325 closed vs 1,366 open)")
    oss = load_json(RESULTS / "open_set_scoring.json")
    if oss:
        method_names = {
            "reject":  "Trained reject head (sigmoid)",
            "maha":    "Mahalanobis distance (z_mix)",
            "openmax": "Openmax (Weibull, closest centroid)",
            "msp":     "Max Sigmoid Probability (MSP)",
            "energy":  "Energy score (-logsumexp)",
        }
        out()
        out("| Method                              | AUROC  |")
        out("| ----------------------------------- | ------ |")
        for k, label in method_names.items():
            v = oss["auroc"].get(k)
            if v is not None:
                out(f"| {label:<35} | {v:.4f} |")

    # ── Table 4: Per-NOC exact match ─────────────────────────────────────
    out("\n## Table 4: Set Transformer - Exact Match by NOC")
    st_metrics = load_json(RESULTS / "set_transformer" / "metrics.json")
    if st_metrics:
        per_noc = st_metrics.get("per_noc", {})
        if per_noc:
            out()
            out("| NOC | Exact Match | n samples |")
            out("| --- | ----------- | --------- |")
            for k in sorted(per_noc.keys(), key=int):
                v = per_noc[k]
                em = v.get("em", v.get("exact_match", "—"))
                n  = v.get("n", "—")
                out(f"| {k}   | {fmt(em):<11} | {n:<9} |")

    # ── Table 5: NOC head accuracy ───────────────────────────────────────
    out("\n## Table 5: NOC head accuracy (vs deepNoC PROVEDIt baseline = 0.90)")
    analysis = load_json(RESULTS / "set_transformer" / "analysis.json")
    if analysis:
        noc_head = analysis.get("noc_head", {})
        out()
        out(f"Overall NOC accuracy: **{noc_head.get('overall_accuracy', '—'):.4f}**"
            " (deepNoC: 0.90)")
        out()
        out("| NOC | Accuracy | n |")
        out("| --- | -------- | - |")
        for k, v in sorted(noc_head.get("per_noc", {}).items(), key=lambda x: int(x[0])):
            out(f"| {k}   | {v['acc']:.4f}   | {v['n']} |")

    # ── Table 6: Calibration ─────────────────────────────────────────────
    out("\n## Table 6: Calibration (ECE)")
    if analysis:
        cal = analysis.get("calibration", {})
        out()
        out("| Head | ECE |")
        out("| ---- | --- |")
        rej = cal.get("reject_head", {})
        don = cal.get("donor_head_top1", {})
        out(f"| Reject head | {rej.get('ece', '—'):.4f} |")
        out(f"| Donor head (top-1) | {don.get('ece', '—'):.4f} |")

    # ── Table 7: Robustness ──────────────────────────────────────────────
    out("\n## Table 7: Robustness - Filtered vs UnFiltered (same test samples)")
    rob = load_json(RESULTS / "set_transformer" / "robustness.json")
    if rob:
        out()
        out("| Metric | Filtered (train+test) | UnFiltered (test) |")
        out("| ------ | --------------------- | ----------------- |")
        mf = rob.get("metrics_filtered_ref", {})
        mu = rob.get("metrics_unfiltered", {})
        for k in ["macro_f1", "micro_f1", "exact_match", "hamming"]:
            out(f"| {k:<12} | {fmt(mf.get(k)):<21} | {fmt(mu.get(k)):<17} |")
        auc_f = rob.get("metrics_filtered_ref", {}).get("reject_auroc", "1.0000")
        auc_u = rob.get("reject_auroc_unfiltered")
        out(f"| reject_auroc | {fmt(auc_f):<21} | {fmt(auc_u):<17} |")

    # ── Table 8: Stratified ──────────────────────────────────────────────
    out("\n## Table 8: Stratified accuracy by template DNA")
    strat = load_json(RESULTS / "set_transformer" / "stratified.json")
    if strat:
        out()
        out("| Template DNA | n | Exact Match | Macro F1 |")
        out("| ------------ | - | ----------- | -------- |")
        for row in strat.get("by_template", []):
            if row["n"] == 0: continue
            out(f"| {row['stratum']:<20} | {row['n']:<5} | {fmt(row['exact_match'])} | {fmt(row['macro_f1'])} |")

    # ── Multi-seed (if available) ────────────────────────────────────────
    ms = load_json(RESULTS / "multi_seed" / "summary.json")
    if ms:
        out("\n## Table 9: Multi-seed confidence intervals")
        out()
        out("| Metric | Mean ± Std |")
        out("| ------ | ---------- |")
        for k, v in ms.get("stats", {}).items():
            out(f"| {k} | {v['mean']:.4f} ± {v['std']:.4f} |")

    # ── Architecture sweep (if available) ───────────────────────────────
    for sweep_axis, sweep_label in [
        ("m_inducing", "Table 10a: Sweep — m_inducing (inducing points)"),
        ("n_isab",     "Table 10b: Sweep — n_isab (ISAB stack depth)"),
        ("n_heads",    "Table 10c: Sweep — n_heads (attention heads)"),
    ]:
        sw = load_json(RESULTS / f"sweep_{sweep_axis}.json")
        if sw:
            out(f"\n## {sweep_label}")
            out()
            out("| Config | Macro F1 | Exact Match | Reject AUROC |")
            out("| ------ | -------- | ----------- | ------------ |")
            for r in sw:
                t = r.get("test", r)
                auc = r.get("reject_auroc")
                out(f"| {r['sweep_label']:<20} | {fmt(t.get('macro_f1'))} | "
                    f"{fmt(t.get('exact_match'))} | {fmt(auc) if auc else '—'} |")

    # ── Attention ────────────────────────────────────────────────────────
    attn = load_json(RESULTS / "set_transformer" / "attn_locus_importance.json")
    if attn:
        out("\n## Attention: Top-10 loci by PMA attention weight")
        out()
        out("| Rank | Locus | Mean Attention |")
        out("| ---- | ----- | -------------- |")
        for i, (locus, imp) in enumerate(
            zip(attn["ranked_loci"][:10], attn["ranked_importance"][:10]), 1
        ):
            out(f"| {i:<4} | {locus:<15} | {imp:.4f}         |")

    if args.save:
        out_path = ROOT / "reports" / "paper_tables.md"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

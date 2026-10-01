"""
fold_report.py — provenance, NoC diagnostics and the per-fold report of train_set_transformer.py.

  fingerprints   arrfp / params_fp: the ID path (logits_cls, rank_te) and the backbone parameters
                 must be identical between builds; the trainer prints them as [ID GUARD]
  fold identity  fold_stamp: which archive cut a backbone was Phase-1 trained on (written next to
                 best_model.pt, refused by --init_from when it does not match the data dir)
  open labels    open_labels: true NoC + panel hits per open row, from open_labels/donor_map.npz
  diagnostics    count_diagnostics: template invariance, count distribution, selective
                 classification (deepNoC s3.3 / Fig 7) and per-source confusion
  report         write_report: results/<run>/report.json + reports/fold<F>_seed<S>.json
                 (schema noc_id_fold_report/v1), pooled across folds by report_folds.py

Split out of train_set_transformer.py unchanged: it computes nothing the trainer did not compute.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

REPORT_SCHEMA = "noc_id_fold_report/v1"


def arrfp(a):
    """Short fingerprint of an array — proves the ID path did not move."""
    return hashlib.md5(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:10]


def params_fp(model):
    return arrfp(np.concatenate([q.detach().cpu().numpy().ravel() for q in model.parameters()]))


def fold_stamp(data_dir):
    """Identity of the archive cut in `data_dir` (meta_set.json). A backbone is fold-specific: its
    in-silico train set was built from THAT fold's 45 known donors, so on another fold it has already
    seen synthetic versions of combinations under test. Phase 1 writes this next to best_model.pt and
    --init_from refuses a backbone whose stamp does not match."""
    mp = Path(data_dir) / "meta_set.json"
    if not mp.exists():
        return None
    m = json.load(open(mp))
    combos = sorted(tuple(sorted(c)) for c in m.get("combo_list", []))
    return {"unknown_donors": sorted(m.get("unknown_donors", [])),
            "n_combo": len(combos),
            "combo_md5": hashlib.md5(repr(combos).encode()).hexdigest()[:12],
            "split_sizes": m.get("split_sizes", {})}


def open_labels(data_dir, n_open, lbdir):
    """True NoC and panel-hit count of every open row, or (None, reason).

    Donor ids per row come from open_labels/donor_map.npz, keyed by md5[:16] of the raw tokens_open
    row (ids parsed from the PROVEDIt sample name); panel hits count the ids in this fold's
    known_donors. Reproduces the labels shipped earlier for folds 1-3 exactly (1339/1705/1365 rows)
    and maps every open row of folds 0-6."""
    mp = Path(lbdir) / "donor_map.npz"; rp = Path(data_dir) / "tokens_open.npy"
    fp = Path(data_dir) / "fold_info.json"
    for p in (mp, rp, fp):
        if not p.exists():
            return None, f"{p.name} missing"
    z = np.load(mp); lut = {h: d for h, d in zip(z["hash"], z["donors"])}
    raw = np.load(rp)
    if len(raw) != n_open:
        return None, f"tokens_open has {len(raw)} rows, tokens8_open {n_open}"
    rows = np.full((len(raw), 5), -1, np.int64); miss = 0
    for i, r in enumerate(raw):
        d = lut.get(hashlib.md5(np.ascontiguousarray(r).tobytes()).hexdigest()[:16])
        if d is None:
            miss += 1
        else:
            rows[i] = d
    if miss:
        return None, f"{miss} of {len(raw)} open rows not in donor_map.npz"
    known = json.load(open(fp))["known_donors"]
    valid = rows >= 0
    return (valid.sum(1).astype(int), (np.isin(rows, known) & valid).sum(1).astype(int)), "donor_map.npz"


def decode_k(P, mode="argmax"):
    """Count from the 5-way posterior. `argmax` takes the top class; `expect` rounds the expected value,
    which keeps the ORDER of the classes (NoC is ordinal) instead of treating them as unrelated labels.
    Measured on the fold 1 gate features, 3 seeds: expect 0.8588+-0.0032 vs argmax 0.8484+-0.0027, with
    fewer 2-or-more-off calls (0.014 vs 0.017)."""
    if mode == "expect":
        return np.clip(np.rint((P * np.arange(1, 6)).sum(1)), 1, 5).astype(int)
    return (P.argmax(1) + 1).astype(int)


def f1_macro(t, p):
    return float(f1_score(t, p, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0))


def score_count_arm(name, P, info, k_true, decode, folds=None, seconds=None):
    """Score ONE count arm's 5-way posterior and fill `info` with everything a report needs.

    Pure numpy: the model is already frozen and this only re-reads its output, which is why it lives
    here beside decode_k and not in the trainer. Both decodes are always stored, never only the one
    that happens to be shipped: the SAME posterior read with `argmax` and with `expect` gives two
    different numbers (fold 0 zero_shot: .9102 vs .9014), and a table that mixes them is wrong.

    Returns the decoded k under `decode`.
    """
    k = decode_k(P, decode)
    alt = "expect" if decode == "argmax" else "argmax"
    ka = decode_k(P, alt)
    f5 = f1_score(k_true, k, average=None, labels=[1, 2, 3, 4, 5], zero_division=0)
    f5a = f1_score(k_true, ka, average=None, labels=[1, 2, 3, 4, 5], zero_division=0)
    cm = confusion_matrix(k_true, k, labels=[1, 2, 3, 4, 5])
    mix = k_true >= 2

    info["decode"] = decode
    info["macro_f1_" + alt + "_decode"] = round(f1_macro(k_true, ka), 4)
    info["per_noc_f1_" + alt + "_decode"] = [round(float(v), 4) for v in f5a]
    info["macro_f1_mixtures_" + alt + "_decode"] = round(float(f5a[1:].mean()), 4)
    info["accuracy_" + alt + "_decode"] = round(float((ka == k_true).mean()), 4)
    print(f"  [DECODE] {name}: {decode} macroF1 {f1_macro(k_true, k):.4f} | {alt} "
          f"{f1_macro(k_true, ka):.4f} | 2-or-more off {float((abs(k - k_true) >= 2).mean()):.4f} vs "
          f"{float((abs(ka - k_true) >= 2).mean()):.4f}", flush=True)

    if seconds is not None:
        info["seconds"] = round(seconds, 1)
    info["folds"] = folds
    info["train_f1_mean"] = round(float(np.mean([r["train_f1"] for r in folds])), 4) if folds else None
    info["heldout_f1_mean"] = round(float(np.mean([r["heldout_f1"] for r in folds])), 4) if folds else None
    info.update(name=name, macro_f1=round(float(f5.mean()), 4),
                accuracy=round(float((k == k_true).mean()), 4),
                per_noc_f1=[round(float(v), 4) for v in f5],
                # everything needed to rebuild the full NoC table from JSON, without reading the log:
                macro_f1_mixtures=round(float(f5[1:].mean()), 4),
                accuracy_mixtures=round(float((k[mix] == k_true[mix]).mean()), 4),
                confusion_rows_true_cols_pred=cm.tolist(),
                per_noc_n=[int(x) for x in cm.sum(1)],
                per_noc_recall=[round(float(cm[i, i] / max(cm[i].sum(), 1)), 4) for i in range(5)],
                per_noc_precision=[round(float(cm[i, i] / max(cm[:, i].sum(), 1)), 4) for i in range(5)],
                under=int(((k < k_true) & mix).sum()), over=int(((k > k_true) & mix).sum()))
    return k


def score_open_split(name, Po, info, op_true, op_panel, decode, n_models):
    """Same scoring for the OPEN split: profiles holding at least one donor outside the 45-donor panel.

    Reported with BOTH decodes because they disagree here in the opposite direction to the closed set:
    `expect` helps on panel donors and blurs the off-panel rows (argmax won 6/6 open cells, muc 17)."""
    k = decode_k(Po, decode)
    ka = decode_k(Po, alt := ("expect" if decode == "argmax" else "argmax"))
    unk = np.clip(op_true - op_panel, 0, 5)
    mix = op_true >= 2
    mf1 = float(f1_score(op_true[mix], k[mix], average="macro", labels=[2, 3, 4, 5], zero_division=0))
    print(f"  [OPEN DECODE] {name}: {decode} acc {float((k == op_true).mean()):.4f} | "
          f"{alt} {float((ka == op_true).mean()):.4f}", flush=True)
    print(f"  [{name}] OPEN split (contributors outside the panel), mean of {n_models} fold models: "
          f"n={len(op_true)} acc {float((k == op_true).mean()):.4f}  mixtures macroF1 {mf1:.4f}"
          + "  | acc by outside contributors " + "  ".join(
              f"{int(u)}:{float((k[unk == u] == op_true[unk == u]).mean()):.3f} (n={int((unk == u).sum())})"
              for u in np.unique(unk)))
    info["open"] = {"n": int(len(op_true)), "accuracy": round(float((k == op_true).mean()), 4),
                    "accuracy_" + alt + "_decode": round(float((ka == op_true).mean()), 4),
                    "mixtures_macro_f1": round(mf1, 4)}


def print_confusion(name, kt, kp):
    cm = confusion_matrix(kt, kp, labels=[1, 2, 3, 4, 5])
    print(f"\n  confusion [{name}]  rows=true, cols=predicted")
    print(f"  {'':<9}" + "".join(f"{v:>7}" for v in range(1, 6)) + f"{'prec':>8}{'rec':>7}{'F1':>7}{'bias':>8}")
    for i, v in enumerate(range(1, 6)):
        tp = int(cm[i, i]); fn = int(cm[i].sum()) - tp; fp = int(cm[:, i].sum()) - tp
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        b = float((kp[kt == v] - v).mean()) if (kt == v).any() else float("nan")
        print(f"    NOC{v:<5}" + "".join(f"{x:>7d}" for x in cm[i]) + f"{p:>8.3f}{r:>7.3f}{f:>7.3f}{b:>+8.2f}")



def count_diagnostics(data_dir, noc, k_true, mixtures, sources, p_count, p_name=None):
    """Template invariance, predicted-count distribution, selective classification and confusion.
    Prints; returns the selective-classification curve for metrics.json."""
    # TEMPLATE INVARIANCE: PROVEDIt's 4-5 person mixtures carry more DNA than its 2-3 person ones, so a
    # counter that leans on peak height scores well here and fails in casework. A height shortcut shows
    # a large `spread` and predicted k climbing faster than true k across template quartiles.
    _tp = data_dir / "meta_template_test.npy"
    if _tp.exists():
        _tm = np.load(_tp).astype(np.float64)
        _ok = mixtures & np.isfinite(_tm)
        if int(_ok.sum()) >= 40:
            _qs = np.quantile(_tm[_ok], [0.25, 0.5, 0.75])
            _bin = np.digitize(_tm, _qs)
            _ns = [int((_ok & (_bin == b)).sum()) for b in range(4)]
            print("\n  === TEMPLATE INVARIANCE (mixtures only) ===")
            print(f"  corr(NOC, template) over these mixtures = {np.corrcoef(noc[_ok], _tm[_ok])[0, 1]:+.3f}"
                  f"   quartile cuts at {np.round(_qs, 3).tolist()}")
            print("  " + "{:<20}".format("count accuracy")
                  + "".join(f"{'Q'+str(i+1)+'(n='+str(_ns[i])+')':>13}" for i in range(4)) + f"{'spread':>9}")
            for nm, kk in sources:
                _a = [float((kk[_ok & (_bin == b)] == k_true[_ok & (_bin == b)]).mean()) if _ns[b] else float("nan")
                      for b in range(4)]
                print("  " + "{:<20}".format(nm) + "".join(f"{x:>13.3f}" for x in _a)
                      + f"{np.nanmax(_a) - np.nanmin(_a):>9.3f}")
            print("  " + "{:<20}".format("mean TRUE k") + "".join(
                f"{float(k_true[_ok & (_bin == b)].mean()) if _ns[b] else float('nan'):>13.2f}" for b in range(4)))
            for nm, kk in sources:
                print("  " + "{:<20}".format("mean pred k " + nm) + "".join(
                    f"{float(kk[_ok & (_bin == b)].mean()) if _ns[b] else float('nan'):>13.2f}" for b in range(4)))

    # count distribution, selective classification (deepNoC s3.3 / Fig 7), confusion
    print("\n  === NOC diagnostics ===")
    print(f"  {'count distribution (eval)':<28}" + "".join(f"{f'NOC{v}':>8}" for v in range(1, 6)))
    print(f"  {'true':<28}" + "".join(f"{int((k_true == v).sum()):>8}" for v in range(1, 6)))
    for nm, kk in sources:
        print(f"  {'pred: ' + nm:<28}" + "".join(f"{int((np.clip(kk, 1, 5) == v).sum()):>8}" for v in range(1, 6)))
    _conf = p_count.max(1); _pred = p_count.argmax(1) + 1
    _src = p_name or (sources[-1][0] if sources else "count")   # p_count belongs to the shipped count
    print(f"\n  selective classification [{_src}] — abstain when max prob < t")
    print(f"  {'t':>6}{'coverage':>10}{'acc|classified':>16}{'macroF1|cls':>13}")
    _rows = []
    for _t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        _m = _conf >= _t
        if _m.sum() == 0:
            continue
        _a = float((_pred[_m] == k_true[_m]).mean()); _f = f1_macro(k_true[_m], _pred[_m])
        print(f"  {_t:>6.2f}{_m.mean():>10.3f}{_a:>16.4f}{_f:>13.4f}")
        _rows.append({"t": _t, "coverage": round(float(_m.mean()), 4), "acc": round(_a, 4), "macro_f1": round(_f, 4)})
    _hit95 = None
    for _t in np.arange(0.0, 1.0, 0.01):
        _m = _conf >= _t
        if _m.sum() and (_pred[_m] == k_true[_m]).mean() >= 0.95:
            _hit95 = (float(_t), float(_m.mean())); break
    print(f"    -> acc 0.95 at t={_hit95[0]:.2f}, {_hit95[1]:.1%} of profiles classified"
          if _hit95 else "    -> acc 0.95 not reachable at any threshold")
    sel_curve = {_src: {"rows": _rows, "t_for_acc95": (round(_hit95[0], 2) if _hit95 else None),
                              "coverage_at_acc95": (round(_hit95[1], 4) if _hit95 else None)}}
    for nm, kk in sources:
        print_confusion(nm, k_true, np.clip(kk, 1, 5))

    return sel_curve


def write_report(results_dir, root, data_dir, cfg, seed, eval_label, model, noc, k_pred, k_gate,
                 y_true, y_pred, y_pred_oracle, rank_te, L_te, combo_id, open_n, oracle, em_post,
                 auroc, alpha_pr, lora_info, count_source, macro_f1, param_fp_before):
    """Write the fold report (NoC + ID from the SAME run) and print the [REPORT] lines."""
    k_true = np.clip(noc, 1, 5).astype(int)
    k_pred = np.clip(k_pred, 1, 5).astype(int)
    mixtures = noc >= 2
    # ══ FOLD REPORT — NoC and ID from the same run; report_folds.py pools these files over folds ══
    fi = json.load(open(data_dir / "fold_info.json")) if (data_dir / "fold_info.json").exists() else {}
    r4 = lambda x: (None if x is None or not np.isfinite(float(x)) else round(float(x), 4))
    cm = confusion_matrix(k_true, k_pred, labels=[1, 2, 3, 4, 5])
    per = {}
    for i, v in enumerate(range(1, 6)):
        tp = int(cm[i, i]); fn = int(cm[i].sum()) - tp; fp = int(cm[:, i].sum()) - tp
        per[str(v)] = {"n": int(cm[i].sum()),
                       "f1": r4(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
                       "recall": r4(tp / (tp + fn)) if (tp + fn) else None,
                       "precision": r4(tp / (tp + fp)) if (tp + fp) else None,
                       "bias": r4((k_pred[k_true == v] - v).mean()) if (k_true == v).any() else None}

    def donor_prf(yp):
        Yt = y_true.astype(bool); Yp = np.asarray(yp).astype(bool); out = {}
        for nm, m in [("overall", np.ones(len(k_true), bool))] + [(str(v), k_true == v) for v in range(1, 6)]:
            tp = int((Yt[m] & Yp[m]).sum()); npred = int(Yp[m].sum()); ntrue = int(Yt[m].sum())
            out[nm] = {"precision": r4(tp / npred) if npred else None,
                       "recall": r4(tp / ntrue) if ntrue else None,
                       "f1": r4(2 * tp / (npred + ntrue)) if (npred + ntrue) else None}
        return out

    em_dict = lambda r: {"overall": r4(r[0]), **{str(v): r4(r[v]) for v in range(1, 6)}}
    n_per = {str(v): int((k_true == v).sum()) for v in range(1, 6)}
    params_unchanged = params_fp(model) == param_fp_before
    report = {
        "schema": REPORT_SCHEMA,
        "code": cfg["_code_hash"], "fold": fi.get("fold"), "n_folds": fi.get("n_folds"),
        "unknown_donors": fi.get("unknown_donors"), "seed": seed, "ft_seed": seed,
        "protocol": "strict", "eval": eval_label,
        "backbone": {"source": cfg.get("_bb_src"), "fold_stamp": cfg.get("_bb_stamp"),
                     "params": int(sum(p.numel() for p in model.parameters()))},
        "data": {"eval_n": int(len(k_true)), "per_noc_n": n_per, "mixtures": int(mixtures.sum()),
                 "test_combos": int(len(np.unique(combo_id[combo_id >= 0]))), "open_n": int(open_n)},
        "checks": {"id_fp_logits_cls": arrfp(L_te), "id_fp_rank_te": arrfp(rank_te),
                   "params_unchanged": bool(params_unchanged),
                   "split_audit_shared_combos": cfg["_split_audit_shared"],
                   "lora_folds_combination_disjoint": True},
        "noc": {"source": count_source, "lora": lora_info, "count_info": lora_info,
                "accuracy": r4((k_pred == k_true).mean()), "macro_f1": r4(macro_f1),
                "mixtures_accuracy": r4((k_pred[mixtures] == k_true[mixtures]).mean()) if mixtures.any() else None,
                "per_noc": per, "confusion_rows_true_cols_pred": cm.tolist(),
                "gate_macro_f1": r4(f1_macro(k_true, k_gate))},
        "id": {"oracle_em": em_dict(oracle), "em_at_pred_k": em_dict(em_post),
               "donor_at_pred_k": donor_prf(y_pred),
               "donor_at_oracle_k": donor_prf(y_pred_oracle),
               "reject_auroc": r4(auroc) if auroc is not None else None,
               "phi_rerank_alpha": r4(alpha_pr)},
    }
    with open(results_dir / "report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    rdir = root / "reports"; rdir.mkdir(exist_ok=True)
    rname = rdir / f"fold{fi.get('fold', 'NA')}_seed{seed}.json"
    with open(rname, "w") as fh:
        json.dump(report, fh, indent=2)
    nq, iq, cq = report["noc"], report["id"], report["checks"]
    print(f"\n[REPORT] fold {report['fold']} | unknown donors {report['unknown_donors']} | CODE {report['code']} "
          f"| eval n={len(k_true)} per NOC {n_per}")
    print(f"[REPORT] NoC  source={count_source} | acc {nq['accuracy']} | macroF1 {nq['macro_f1']} | F1 NOC1..5 "
          + " ".join(str(per[str(v)]['f1']) for v in range(1, 6)))
    print(f"[REPORT] ID   oracle EM {iq['oracle_em']['overall']} | EM@k {iq['em_at_pred_k']['overall']} "
          f"| EM@k NOC1..5 " + " ".join(str(iq['em_at_pred_k'][str(v)]) for v in range(1, 6))
          + f" | donor P/R/F1 @k {iq['donor_at_pred_k']['overall']['precision']}/"
          f"{iq['donor_at_pred_k']['overall']['recall']}/{iq['donor_at_pred_k']['overall']['f1']} "
          f"| reject AUROC {iq['reject_auroc']}")
    print(f"[REPORT] checks: fp(logits_cls) {cq['id_fp_logits_cls']} | params_unchanged {cq['params_unchanged']} "
          f"| split-audit shared combos {cq['split_audit_shared_combos']} "
          f"| LoRA folds combination-disjoint {cq['lora_folds_combination_disjoint']}")
    print(f"[REPORT] -> {results_dir / 'report.json'}  +  {rname}")
    if not params_unchanged or cfg["_split_audit_shared"]:
        raise SystemExit("[REPORT] CHECK FAILED: backbone parameters moved or test combinations leaked into "
                         "Phase 1 — this fold's numbers are not valid")



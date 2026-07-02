"""
eval_cross_inc3.py — zero-shot OOD eval for the Increment-3 arms whose models carry
extra modules that eval_crossfolder.build_model() does not construct:
  - repA  geno_query   : registers donor_geno / donor_geno_mask buffers + geno_proj,
                         and _encode_geno() is USED in the forward (anchors donor queries).
  - repB  donor_contrast: adds an inference-discarded proj_peak head.

Non-invasive: reuses eval_crossfolder's load_tag / forward_all / _agg and the metric
helpers from train_set_transformer; only the model build differs. Same metrics/JSON
layout as eval_crossfolder so results are directly comparable to the base/sparse runs.

Usage:
  python eval_cross_inc3.py --tag idplus28_rd14 --arms inc3_repA_genoq,inc3_repB_donorcon --seeds 42,43,44
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import eval_crossfolder as ecf
from train_set_transformer import (
    SetTransformerMixture, per_noc_em, topk_decode, full_report, DEVICE,
)

ROOT = Path(__file__).resolve().parent
CROSS = ROOT / "data_cross"
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def build_model_inc3(cfg: dict, state: dict) -> SetTransformerMixture:
    """Same as ecf.build_model but also wires geno_query / donor_contrast. For geno_query
    we pass zero buffers of the checkpoint's shape; load_state_dict restores the real
    (GF-space) reference-genotype anchors."""
    geno_query = bool(cfg.get("geno_query"))
    donor_contrast = bool(cfg.get("donor_contrast"))
    extra = {}
    if geno_query:
        dg, dgm = state["donor_geno"], state["donor_geno_mask"]
        extra["donor_geno"] = torch.zeros_like(dg).float()
        extra["donor_geno_mask"] = torch.zeros_like(dgm).bool()
    m = SetTransformerMixture(
        n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16), d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4), n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
        n_classes=cfg.get("n_classes", 45), n_noc=cfg.get("n_noc", 6), dropout=cfg.get("dropout", 0.1),
        cls_decoder=cfg.get("cls_decoder", "pooled"), decoder_source=cfg.get("decoder_source", "encoded"),
        n_token_feats=cfg.get("n_token_feats", 8), encoder=cfg.get("encoder", "isab"),
        dec_layers=cfg.get("dec_layers", 2), num_embed=cfg.get("num_embed", "raw"),
        n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
        periodic_sigma=cfg.get("periodic_sigma", 1.0), aux_heads=cfg.get("aux_heads", False),
        noc_contrast=cfg.get("noc_contrast", False),
        noc_detach=(cfg.get("noc_contrast_mode", "shared") == "detach"),
        d_proj=cfg.get("d_proj", 64), sparse_attn=cfg.get("sparse_attn", False),
        geno_query=geno_query, donor_contrast=donor_contrast, **extra,
    ).to(DEVICE)
    return m


def eval_one(ckpt_dir: Path, tag: str, summary: dict):
    cfg = json.load(open(ckpt_dir / "metrics.json"))["config"]
    n_tok = cfg.get("n_token_feats", 8)
    state = torch.load(ckpt_dir / "best_model.pt", map_location=DEVICE, weights_only=True)
    model = build_model_inc3(cfg, state)
    model.load_state_dict(state)  # strict — proves geno buffers / heads all matched

    d = ecf.load_tag(tag, n_tok)
    P, C, R = ecf.forward_all(model, d["tokens"], d["mask"])
    k_card = C.argmax(1) + 1
    noc = d["noc"]
    res = {"arm_seed": ckpt_dir.name, "tag": tag, "n_token_feats": n_tok}

    res["card_noc_acc_all"] = float((np.clip(k_card, 1, 5) == np.clip(noc, 1, 5)).mean())

    if summary["id_measurable"] and d["closed"].any():
        ci = d["closed"]
        yC, PC, nC = d["y"][ci], P[ci], noc[ci]
        kC = k_card[ci]
        oracle = per_noc_em(yC, topk_decode(PC, nC), nC)
        joint = per_noc_em(yC, topk_decode(PC, kC), nC)
        rep = full_report(yC, topk_decode(PC, kC), nC,
                          f"OOD {tag} — {ckpt_dir.name} (joint-card, closed n={ci.sum()})")
        print(f"  {'decode':<12}{'all':>7}{'N1':>7}{'N2':>7}{'N3':>7}{'N4':>7}{'N5':>7}")
        for nm, r in [("oracle", oracle), ("joint", joint)]:
            print(f"  {nm:<12}" + "".join(f"{x:>7.3f}" for x in r))
        res["id_em"] = float(rep["exact_match"])
        res["oracle"] = {str(j): (None if np.isnan(oracle[j]) else round(float(oracle[j]), 4)) for j in range(6)}
        res["joint"] = {str(j): (None if np.isnan(joint[j]) else round(float(joint[j]), 4)) for j in range(6)}

    if summary["id_measurable"] and d["closed"].any() and (~d["closed"]).any():
        lab = np.concatenate([np.zeros(d["closed"].sum()), np.ones((~d["closed"]).sum())])
        sc = np.concatenate([R[d["closed"]], R[~d["closed"]]])
        try:
            res["reject_auroc_internal"] = float(roc_auc_score(lab, sc))
        except Exception:
            pass
    gf_tok = DATA / f"tokens{n_tok}_test.npy"
    if gf_tok.exists():
        gtok = np.load(gf_tok).astype(np.float32); gmask = np.load(DATA / "mask_test.npy")
        _, _, Rg = ecf.forward_all(model, gtok, gmask)
        lab = np.concatenate([np.zeros(len(Rg)), np.ones(len(R))])
        sc = np.concatenate([Rg, R])
        try:
            res["reject_auroc_vs_gf"] = float(roc_auc_score(lab, sc))
        except Exception:
            pass

    json.dump(res, open(ckpt_dir / f"metrics_cross_{tag}.json", "w"), indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()

    summary = json.load(open(CROSS / f"summary_{args.tag}.json"))
    print(f"\n=== OOD {args.tag}  panel={summary['panel']}  id_measurable={summary['id_measurable']}"
          f"  n={summary['n_samples']} (closed={summary['n_closed']})  noc={summary['noc_dist']} ===")

    seeds = [s for s in args.seeds.split(",") if s]
    for arm in args.arms.split(","):
        rows = []
        for s in seeds:
            ck = RESULTS / f"{arm}_seed{s}"
            if not (ck / "best_model.pt").exists() or not (ck / "metrics.json").exists():
                print(f"  [skip] {ck.name} (missing checkpoint/metrics)")
                continue
            print(f"\n--- {ck.name} ---")
            rows.append(eval_one(ck, args.tag, summary))
        if not rows:
            continue
        print(f"\n>>> AGG {arm}  (n={len(rows)} seeds)")
        na = ecf._agg([r["card_noc_acc_all"] for r in rows])
        print(f"    NOC count acc (all)   : {na[0]:.3f}±{na[1]:.3f} [{na[2]:.3f},{na[3]:.3f}]")
        if summary["id_measurable"]:
            ie = ecf._agg([r.get("id_em") for r in rows])
            if ie: print(f"    ID EM (closed)        : {ie[0]:.3f}±{ie[1]:.3f} [{ie[2]:.3f},{ie[3]:.3f}]")
            for lab in ("oracle", "joint"):
                for j in range(1, 6):
                    g = ecf._agg([r.get(lab, {}).get(str(j)) for r in rows])
                    if g: print(f"      {lab} N{j}            : {g[0]:.3f}±{g[1]:.3f} [{g[2]:.3f},{g[3]:.3f}]")
        ri = ecf._agg([r.get("reject_auroc_internal") for r in rows])
        if ri: print(f"    reject AUROC internal : {ri[0]:.3f}±{ri[1]:.3f}")
        rg = ecf._agg([r.get("reject_auroc_vs_gf") for r in rows])
        if rg: print(f"    reject AUROC vs GF    : {rg[0]:.3f}±{rg[1]:.3f}")


if __name__ == "__main__":
    main()

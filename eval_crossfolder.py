"""
eval_crossfolder.py — ZERO-SHOT OOD evaluation of frozen GF29 checkpoints on an
external kit folder prepared by prepare_crossfolder.py.

For each checkpoint it rebuilds the model from its saved config and load_state_dict's
best_model.pt — which RESTORES the train-time per-feature standardisation buffers
(feat_mean/feat_std). So the OOD tokens are normalised with TRAIN statistics: the
shift the model sees is the real domain gap, not a re-fit artefact. No training.

Reports (per checkpoint, then mean ± 95% CI across seeds):
  - ID  : per-donor Exact-Match + per-NOC EM + oracle ceiling   (closed samples; rd14 only)
  - NOC : card-head count accuracy (donor-agnostic; ALL samples, incl. rd12)
  - REJECT : folder-internal closed-vs-open AUROC (rd14) and OOD-vs-GF-test AUROC

Usage:
  python eval_crossfolder.py --tag idplus28_rd14 --arms inc2_2b_pe_s3,inc2_2d_sparse --seeds 42,43,44
  python eval_crossfolder.py --tag idplus29_rd12 --arms inc2_2b_pe_s3 --seeds 42,43,44
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from train_set_transformer import (
    SetTransformerMixture, posthoc_cardinality, per_noc_em, topk_decode, full_report, DEVICE,
)

ROOT = Path(__file__).resolve().parent
CROSS = ROOT / "data_cross"
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def build_model(cfg: dict) -> SetTransformerMixture:
    n_tok = cfg.get("n_token_feats", 3)
    m = SetTransformerMixture(
        n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16), d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4), n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
        n_classes=cfg.get("n_classes", 45), n_noc=cfg.get("n_noc", 6), dropout=cfg.get("dropout", 0.1),
        cls_decoder=cfg.get("cls_decoder", "pooled"), decoder_source=cfg.get("decoder_source", "encoded"),
        n_token_feats=n_tok, encoder=cfg.get("encoder", "isab"), dec_layers=cfg.get("dec_layers", 2),
        num_embed=cfg.get("num_embed", "raw"), n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
        periodic_sigma=cfg.get("periodic_sigma", 1.0), aux_heads=cfg.get("aux_heads", False),
        noc_contrast=cfg.get("noc_contrast", False),
        noc_detach=(cfg.get("noc_contrast_mode", "shared") == "detach"),
        d_proj=cfg.get("d_proj", 64), sparse_attn=cfg.get("sparse_attn", False),
    ).to(DEVICE)
    return m


def forward_all(model, tokens, mask, bs=256):
    """Return P(N,45) sigmoid cls probs, card logits(N,n_noc), reject score(N,)."""
    P, C, R = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tokens), bs):
            tk = torch.from_numpy(tokens[i:i+bs]).to(DEVICE)
            mk = torch.from_numpy(mask[i:i+bs]).to(DEVICE)
            out = model(tk, mk)
            P.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
            C.append(out["logits_card"].cpu().numpy())
            R.append(torch.sigmoid(out["logit_reject"]).cpu().numpy().ravel())
    return np.concatenate(P), np.concatenate(C), np.concatenate(R)


def load_tag(tag: str, n_tok: int):
    pref = f"tokens{n_tok}"
    d = {
        "tokens": np.load(CROSS / f"{pref}_{tag}.npy").astype(np.float32),
        "mask":   np.load(CROSS / f"mask_{tag}.npy"),
        "y":      np.load(CROSS / f"y_{tag}_set.npy").astype(np.float32),
        "noc":    np.load(CROSS / f"noc_{tag}.npy").astype(np.int64),
        "closed": np.load(CROSS / f"is_closed_{tag}.npy"),
        "cond":   np.load(CROSS / f"condition_{tag}.npy"),
    }
    return d


def eval_one(ckpt_dir: Path, tag: str, summary: dict):
    cfg = json.load(open(ckpt_dir / "metrics.json"))["config"]
    n_tok = cfg.get("n_token_feats", 8)
    model = build_model(cfg)
    state = torch.load(ckpt_dir / "best_model.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)

    d = load_tag(tag, n_tok)
    P, C, R = forward_all(model, d["tokens"], d["mask"])
    k_card = C.argmax(1) + 1
    noc = d["noc"]

    res = {"arm_seed": ckpt_dir.name, "tag": tag, "n_token_feats": n_tok}

    # NOC count transfer (donor-agnostic; ALL samples) ---------------------------
    res["card_noc_acc_all"] = float((np.clip(k_card, 1, 5) == np.clip(noc, 1, 5)).mean())
    res["noc_count_per_noc"] = {int(k): float((np.clip(k_card[noc == k], 1, 5) == k).mean())
                                for k in range(1, 6) if (noc == k).any()}

    # ID + per-NOC EM (closed samples; rd14 only) --------------------------------
    if summary["id_measurable"] and d["closed"].any():
        ci = d["closed"]
        yC, PC, nC = d["y"][ci], P[ci], noc[ci]
        kC = k_card[ci]
        oracle = per_noc_em(yC, topk_decode(PC, nC), nC)
        joint  = per_noc_em(yC, topk_decode(PC, kC), nC)
        rep = full_report(yC, topk_decode(PC, kC), nC,
                          f"OOD {tag} — {ckpt_dir.name} (joint-card decode, closed n={ci.sum()})")
        print(f"  {'decode':<12}{'all':>7}{'N1':>7}{'N2':>7}{'N3':>7}{'N4':>7}{'N5':>7}")
        for nm, r in [("oracle", oracle), ("joint", joint)]:
            print(f"  {nm:<12}" + "".join(f"{x:>7.3f}" for x in r))
        res["id_em"] = float(rep["exact_match"])
        res["oracle"] = {str(j): (None if np.isnan(oracle[j]) else round(float(oracle[j]), 4)) for j in range(6)}
        res["joint"]  = {str(j): (None if np.isnan(joint[j]) else round(float(joint[j]), 4)) for j in range(6)}

    # Reject AUROC ---------------------------------------------------------------
    # (a) folder-internal: closed (0) vs folder-open unknown-donor (1)  — rd14 only
    if summary["id_measurable"] and d["closed"].any() and (~d["closed"]).any():
        lab = np.concatenate([np.zeros(d["closed"].sum()), np.ones((~d["closed"]).sum())])
        sc  = np.concatenate([R[d["closed"]], R[~d["closed"]]])
        try:
            res["reject_auroc_internal"] = float(roc_auc_score(lab, sc))
        except Exception:
            pass
    # (b) OOD detection: GF in-silico closed test (0) vs this folder (1)
    gf_tok = DATA / f"tokens{n_tok}_test.npy"
    if gf_tok.exists():
        gtok = np.load(gf_tok).astype(np.float32); gmask = np.load(DATA / "mask_test.npy")
        _, _, Rg = forward_all(model, gtok, gmask)
        lab = np.concatenate([np.zeros(len(Rg)), np.ones(len(R))])
        sc  = np.concatenate([Rg, R])
        try:
            res["reject_auroc_vs_gf"] = float(roc_auc_score(lab, sc))
        except Exception:
            pass

    json.dump(res, open(ckpt_dir / f"metrics_cross_{tag}.json", "w"), indent=2)
    return res


def _agg(vals):
    a = np.array([v for v in vals if v is not None], float)
    if len(a) == 0:
        return None
    m = a.mean(); ci = (1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return (m, ci, a.min(), a.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", required=True, help="comma list, e.g. inc2_2b_pe_s3,inc2_2d_sparse")
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
        na = _agg([r["card_noc_acc_all"] for r in rows])
        print(f"    NOC count acc (all)   : {na[0]:.3f}±{na[1]:.3f} [{na[2]:.3f},{na[3]:.3f}]")
        if summary["id_measurable"]:
            ie = _agg([r.get("id_em") for r in rows])
            if ie: print(f"    ID EM (closed)        : {ie[0]:.3f}±{ie[1]:.3f} [{ie[2]:.3f},{ie[3]:.3f}]")
            for lab in ("oracle", "joint"):
                for j in range(1, 6):
                    g = _agg([r.get(lab, {}).get(str(j)) for r in rows])
                    if g: print(f"      {lab} N{j}            : {g[0]:.3f}±{g[1]:.3f} [{g[2]:.3f},{g[3]:.3f}]")
        ri = _agg([r.get("reject_auroc_internal") for r in rows])
        if ri: print(f"    reject AUROC internal : {ri[0]:.3f}±{ri[1]:.3f}")
        rg = _agg([r.get("reject_auroc_vs_gf") for r in rows])
        if rg: print(f"    reject AUROC vs GF    : {rg[0]:.3f}±{rg[1]:.3f}")


if __name__ == "__main__":
    main()

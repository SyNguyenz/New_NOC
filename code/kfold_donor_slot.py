"""kfold_donor_slot.py — GroupKFold(5) for the Donor-Slot (inc22) SetTransformer.

"Nguyên's part", run as 5-fold GroupKFold grouped by donor-combo (like the notebook),
using the repo's inc22 `SetTransformerMixture` (`code/models/set_transformer.py`) and the
pretrained `Donor-Slot_Set_Transformer.pt` checkpoint as initialisation.

Task: contributor identification = which of 45 known donors are present in a mixture
(the model's `logits_cls` head, targets = `y_*_set.npy`, 45-dim multi-hot). Reported
per fold: donor-set exact-match (EM, oracle top-true-k decode) overall + per NOC, plus
micro/macro-F1 on the multi-label presence.

Data (built beforehand by `data/prepare_data_set.py` + `features/enrich.py`):
  tokens8_{train,val,test}.npy  (N,160,8)   mask_{...}.npy (N,160) bool
  y_{...}_set.npy (N,45) float   noc_{...}.npy (N,)   meta_sample_names_{...}.json
The three closed splits are concatenated into one pool; GroupKFold(5) then re-splits it
by donor-combo (parsed from the sample-file names). `donor_geno`/`donor_geno_mask`/
`owner_lut` are read from the checkpoint (baked buffers).

IMPORTANT LEAKAGE CAVEAT: with --init pretrained (default), the checkpoint was trained on
part of this same real GF pool, so finetune+eval on GroupKFold folds is optimistic. Use
--init scratch for a leak-free cross-validation number.

Usage:
  python code/kfold_donor_slot.py --ckpt /path/Donor-Slot_Set_Transformer.pt \
      [--data-dir code/data --init pretrained --n-folds 5 --seed 42 --epochs 40 \
       --out results/kfold_donor_slot]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
from data.prepare_data_set import parse_donors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# inc22 model config (matches the Donor-Slot checkpoint; see train_set_transformer.py CFG)
MODEL_CFG = dict(
    n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32,
    n_classes=45, dropout=0.1, n_token_feats=8, n_freq=8, d_num_emb=8,
    periodic_sigma=0.3, n_slot_iters=3, ot_eps=0.05, ot_iters=5, gumbel_temp=1.0,
)
SPLITS = ("train", "val", "test")


class AsymmetricLoss(nn.Module):
    """ASL (Ben-Baruch 2020) on the multi-label donor logits — verbatim from train_set_transformer."""

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg; self.gamma_pos = gamma_pos; self.clip = clip; self.eps = eps

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        pt = xs_pos * targets + xs_neg * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        loss *= (1 - pt) ** gamma
        return -loss.mean()


class PoolDataset(Dataset):
    def __init__(self, tokens, mask, y, noc):
        self.tokens = torch.from_numpy(tokens.astype(np.float32))
        self.mask = torch.from_numpy(mask)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.noc = torch.from_numpy(noc.astype(np.int64))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.mask[i], self.y[i], self.noc[i]


def load_pool(data_dir: Path):
    """Concat the closed splits into one pool + build a donor-combo group id per sample."""
    toks, masks, ys, nocs, groups = [], [], [], [], []
    for sp in SPLITS:
        tp = data_dir / f"tokens8_{sp}.npy"
        if not tp.is_file():
            raise SystemExit(f"Missing {tp}. Build data first: prepare_data_set.py + enrich.py")
        toks.append(np.load(tp))
        masks.append(np.load(data_dir / f"mask_{sp}.npy"))
        ys.append(np.load(data_dir / f"y_{sp}_set.npy"))
        nocs.append(np.load(data_dir / f"noc_{sp}.npy"))
        names = json.loads((data_dir / f"meta_sample_names_{sp}.json").read_text())
        for nm in names:
            groups.append("-".join(str(d) for d in sorted(parse_donors(str(nm)))))
    tokens = np.concatenate(toks); mask = np.concatenate(masks)
    y = np.concatenate(ys); noc = np.concatenate(nocs); groups = np.array(groups)
    if not (len(tokens) == len(mask) == len(y) == len(noc) == len(groups)):
        raise SystemExit("Pool array length mismatch — data build is inconsistent.")
    return tokens, mask, y, noc, groups


def build_model(sd, init: str, train_tokens=None, train_mask=None):
    """Construct the repo SetTransformerMixture. donor_geno/owner_lut always come from the
    checkpoint (reference data, not learned). init='pretrained' loads the weights; 'scratch'
    keeps random init and recomputes feature standardisation from the fold's train peaks."""
    model = SetTransformerMixture(
        donor_geno=sd["donor_geno"], donor_geno_mask=sd["donor_geno_mask"],
        owner_lut=sd["owner_lut"], **MODEL_CFG,
    ).to(DEVICE)
    if init == "pretrained":
        model.load_state_dict(sd, strict=False)      # keeps ckpt feat_mean/std too
    else:  # scratch: recompute per-feature standardisation from this fold's train peaks
        tk = train_tokens; mk = train_mask.astype(bool)
        num = tk[:, :, 1:MODEL_CFG["n_token_feats"]][mk]
        model.feat_mean.copy_(torch.tensor(num.mean(0), dtype=torch.float32, device=DEVICE))
        model.feat_std.copy_(torch.tensor(num.std(0) + 1e-6, dtype=torch.float32, device=DEVICE))
    return model


@torch.no_grad()
def infer_probs(model, loader):
    model.eval(); P, Y, NOC = [], [], []
    for tok, msk, y, noc in loader:
        P.append(torch.sigmoid(model(tok.to(DEVICE), msk.to(DEVICE))["logits_cls"]).cpu().numpy())
        Y.append(y.numpy()); NOC.append(noc.numpy())
    return np.concatenate(P), np.concatenate(Y), np.concatenate(NOC)


def topk_decode(probs, k_arr):
    yp = np.zeros_like(probs, dtype=int)
    for i in range(len(probs)):
        k = int(max(1, min(5, round(k_arr[i]))))
        yp[i, np.argsort(probs[i])[::-1][:k]] = 1
    return yp


def per_noc_em(y_true, y_pred, noc):
    e = np.all(y_true == y_pred, axis=1)
    out = {"overall": round(float(e.mean()), 4)}
    for j in range(1, 6):
        m = noc == j
        out[str(j)] = round(float(e[m].mean()), 4) if m.any() else None
    return out


def evaluate(model, loader):
    """Oracle top-true-k donor-set decode → EM overall + per NOC, plus micro/macro-F1."""
    probs, y_true, noc = infer_probs(model, loader)
    y_pred = topk_decode(probs, noc)                       # oracle k = true NOC
    em = per_noc_em(y_true.astype(int), y_pred, noc)
    micro = float(f1_score(y_true.astype(int), y_pred, average="micro", zero_division=0))
    macro = float(f1_score(y_true.astype(int), y_pred, average="macro", zero_division=0))
    return em, micro, macro


def run_fold(cfg, sd, tr_idx, va_idx, tokens, mask, y, noc):
    tr_dl = DataLoader(PoolDataset(tokens[tr_idx], mask[tr_idx], y[tr_idx], noc[tr_idx]),
                       batch_size=cfg.batch_size, shuffle=True)
    va_dl = DataLoader(PoolDataset(tokens[va_idx], mask[va_idx], y[va_idx], noc[va_idx]),
                       batch_size=512, shuffle=False)
    model = build_model(sd, cfg.init, tokens[tr_idx], mask[tr_idx])
    criterion = AsymmetricLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    best_micro, best_state, wait = -1.0, None, 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for tok, msk, yb, _ in tr_dl:
            logits = model(tok.to(DEVICE), msk.to(DEVICE))["logits_cls"]
            loss = criterion(logits, yb.to(DEVICE))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        _, micro, _ = evaluate(model, va_dl)
        if micro > best_micro:
            best_micro, wait = micro, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep % 10 == 0 or ep == 1:
            print(f"    ep{ep:3d}  val_micro_f1={micro:.4f}  best={best_micro:.4f}")
        if wait >= cfg.patience:
            print(f"    early stop @ ep{ep}  best={best_micro:.4f}"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    em, micro, macro = evaluate(model, va_dl)
    return {"em": em, "micro_f1": round(micro, 4), "macro_f1": round(macro, 4)}


def run_kfold(cfg, sd, tokens, mask, y, noc, groups):
    gkf = GroupKFold(n_splits=cfg.n_folds)
    per_fold = []
    print(f"\nGroupKFold({cfg.n_folds})  init={cfg.init}  device={DEVICE}  "
          f"pool={len(tokens)}  groups={len(np.unique(groups))}")
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(tokens, y, groups), 1):
        t0 = time.time()
        torch.manual_seed(cfg.seed + fold); np.random.seed(cfg.seed + fold)
        print(f"\n  Fold {fold}/{cfg.n_folds}  train={len(tr_idx)}  val={len(va_idx)}  "
              f"val_groups={len(np.unique(groups[va_idx]))}")
        r = run_fold(cfg, sd, tr_idx, va_idx, tokens, mask, y, noc)
        r["fold"] = fold
        per_fold.append(r)
        print(f"  Fold {fold}: EM={r['em']['overall']:.4f}  micro-F1={r['micro_f1']:.4f}  "
              f"macro-F1={r['macro_f1']:.4f}  ({time.time()-t0:.0f}s)  per-NOC EM={r['em']}")
    ems = [r["em"]["overall"] for r in per_fold]
    micros = [r["micro_f1"] for r in per_fold]
    macros = [r["macro_f1"] for r in per_fold]
    results = {
        "model": "Donor-Slot SetTransformer (inc22)",
        "init": cfg.init,
        "task": "contributor identification (45-donor presence, logits_cls)",
        "per_fold": per_fold,
        "em_mean": round(float(np.mean(ems)), 4), "em_std": round(float(np.std(ems)), 4),
        "micro_f1_mean": round(float(np.mean(micros)), 4), "micro_f1_std": round(float(np.std(micros)), 4),
        "macro_f1_mean": round(float(np.mean(macros)), 4), "macro_f1_std": round(float(np.std(macros)), 4),
        "config": {"n_folds": cfg.n_folds, "seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr},
        "leakage_caveat": (
            "init=pretrained: checkpoint was trained on part of this real GF pool; "
            "finetune+eval on GroupKFold folds is optimistic. Use --init scratch for a "
            "leak-free number." if cfg.init == "pretrained" else
            "init=scratch: leak-free — each fold trains from random init on its own train split."
        ),
    }
    print(f"\n{'='*64}")
    print(f"Donor-Slot SetTransformer — GroupKFold({cfg.n_folds}) — init={cfg.init}")
    print(f"  Donor-set EM : {results['em_mean']:.4f} ± {results['em_std']:.4f}")
    print(f"  Micro-F1     : {results['micro_f1_mean']:.4f} ± {results['micro_f1_std']:.4f}")
    print(f"  Macro-F1     : {results['macro_f1_mean']:.4f} ± {results['macro_f1_std']:.4f}")
    print('='*64)
    return results


def resolve_config(argv=None):
    p = argparse.ArgumentParser(description="GroupKFold(5) Donor-Slot SetTransformer.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", default=str(ROOT.parent / "results" / "kfold_donor_slot"))
    return p.parse_args(argv)


def main(argv=None):
    cfg = resolve_config(argv)
    ckpt = Path(cfg.ckpt); data_dir = Path(cfg.data_dir); out = Path(cfg.out)
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    try:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SystemExit(f"Failed to load checkpoint {ckpt}: {exc}")
    for k in ("donor_geno", "donor_geno_mask", "owner_lut"):
        if k not in sd:
            raise SystemExit(f"Checkpoint missing baked buffer '{k}' — not a Donor-Slot inc22 ckpt.")
    tokens, mask, y, noc, groups = load_pool(data_dir)
    results = run_kfold(cfg, sd, tokens, mask, y, noc, groups)
    out.mkdir(parents=True, exist_ok=True)
    tag = cfg.init
    (out / f"kfold_metrics_{tag}.json").write_text(json.dumps(results, indent=2))
    (out / f"per_fold_scores_{tag}.json").write_text(json.dumps(
        {f"DonorSlot_SetTransformer_{tag}": [r["em"]["overall"] for r in results["per_fold"]]}, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

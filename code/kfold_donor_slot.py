"""kfold_donor_slot.py — GroupKFold(5) for the Donor-Slot (inc22) SetTransformer.

"Nguyên's part", run as 5-fold GroupKFold grouped by donor-combo (like the notebook),
using the repo's inc22 `SetTransformerMixture` (`code/models/set_transformer.py`) with the
pretrained `Donor-Slot_Set_Transformer.pt` checkpoint available as initialisation.

Two tasks (--task):
  count  (default, well-posed) — predict NOC = number of contributors (target_noc, 1..5)
         via the `logits_card` head, trained with class-weighted cross-entropy. This is the
         notebook cell-14 target; metric = Macro-F1 + Micro-F1 over the 5 count classes.
  donor  — contributor identification = which of 45 known donors are present (`logits_cls`,
         targets `y_*_set.npy`), oracle top-true-k decode. NOTE: ill-posed under donor-combo
         CV (each donor's single-source samples are held entirely in one fold); kept for the
         record. See docs/superpowers/2026-07-29-kfold-donor-slot-results.md.

Data (built beforehand by `data/prepare_data_set.py` + `features/enrich.py`):
  tokens8_{train,val,test}.npy (N,160,8)  mask_{...}.npy (N,160) bool
  y_{...}_set.npy (N,45)  noc_{...}.npy (N,)  meta_sample_names_{...}.json
The three closed splits are concatenated into one pool; GroupKFold(5) re-splits it by
donor-combo (parsed from the sample-file names). donor_geno/donor_geno_mask/owner_lut are
read from the checkpoint (baked buffers).

LEAKAGE CAVEAT: --init pretrained uses a checkpoint trained on part of this same GF pool, so
finetune+eval on the folds is optimistic. --init scratch (default) is leak-free.

Usage:
  python code/kfold_donor_slot.py --ckpt /path/Donor-Slot_Set_Transformer.pt \
      [--data-dir code/data --task count --init scratch --n-folds 5 --seed 42 \
       --epochs 60 --patience 12 --out results/kfold_donor_slot]
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

MODEL_CFG = dict(
    n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32,
    n_classes=45, dropout=0.1, n_token_feats=8, n_freq=8, d_num_emb=8,
    periodic_sigma=0.3, n_slot_iters=3, ot_eps=0.05, ot_iters=5, gumbel_temp=1.0,
)
SPLITS = ("train", "val", "test")


class AsymmetricLoss(nn.Module):
    """ASL on the multi-label donor logits — verbatim from train_set_transformer."""

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg; self.gamma_pos = gamma_pos; self.clip = clip; self.eps = eps

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits); xs_neg = 1.0 - xs_pos
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los = targets * torch.log(xs_pos.clamp(min=self.eps)) + \
            (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        pt = xs_pos * targets + xs_neg * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        return -(los * (1 - pt) ** gamma).mean()


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
    """Concat the closed splits into one pool + a donor-combo group id per sample."""
    toks, masks, ys, nocs, groups = [], [], [], [], []
    for sp in SPLITS:
        tp = data_dir / f"tokens8_{sp}.npy"
        if not tp.is_file():
            raise SystemExit(f"Missing {tp}. Build data first: prepare_data_set.py + enrich.py")
        toks.append(np.load(tp)); masks.append(np.load(data_dir / f"mask_{sp}.npy"))
        ys.append(np.load(data_dir / f"y_{sp}_set.npy")); nocs.append(np.load(data_dir / f"noc_{sp}.npy"))
        for nm in json.loads((data_dir / f"meta_sample_names_{sp}.json").read_text()):
            groups.append("-".join(str(d) for d in sorted(parse_donors(str(nm)))))
    tokens = np.concatenate(toks); mask = np.concatenate(masks)
    y = np.concatenate(ys); noc = np.concatenate(nocs); groups = np.array(groups)
    if not (len(tokens) == len(mask) == len(y) == len(noc) == len(groups)):
        raise SystemExit("Pool array length mismatch — data build is inconsistent.")
    return tokens, mask, y, noc, groups


def build_model(sd, init: str, train_tokens=None, train_mask=None):
    """Construct the repo SetTransformerMixture. donor_geno/owner_lut always come from the
    checkpoint. init='pretrained' loads weights; 'scratch' keeps random init and recomputes
    feature standardisation from this fold's train peaks."""
    model = SetTransformerMixture(donor_geno=sd["donor_geno"], donor_geno_mask=sd["donor_geno_mask"],
                                  owner_lut=sd["owner_lut"], **MODEL_CFG).to(DEVICE)
    if init == "pretrained":
        model.load_state_dict(sd, strict=False)
    else:
        mk = train_mask.astype(bool)
        num = train_tokens[:, :, 1:MODEL_CFG["n_token_feats"]][mk]
        model.feat_mean.copy_(torch.tensor(num.mean(0), dtype=torch.float32, device=DEVICE))
        model.feat_std.copy_(torch.tensor(num.std(0) + 1e-6, dtype=torch.float32, device=DEVICE))
    return model


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


@torch.no_grad()
def evaluate(model, loader, task):
    """Returns a metrics dict. count: macro/micro-F1 + acc over NOC 1..5.
    donor: donor-set EM (oracle top-true-k) + micro/macro-F1."""
    model.eval(); Lc, Pd, Y, NOC = [], [], [], []
    for tok, msk, y, noc in loader:
        out = model(tok.to(DEVICE), msk.to(DEVICE))
        Lc.append(out["logits_card"].cpu().numpy())
        Pd.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
        Y.append(y.numpy()); NOC.append(noc.numpy())
    Y = np.concatenate(Y).astype(int); NOC = np.concatenate(NOC)
    if task == "count":
        pred = np.concatenate(Lc).argmax(1) + 1                       # 1..5
        labels = [1, 2, 3, 4, 5]
        return {"macro_f1": round(float(f1_score(NOC, pred, average="macro", labels=labels, zero_division=0)), 4),
                "micro_f1": round(float(f1_score(NOC, pred, average="micro", labels=labels, zero_division=0)), 4),
                "acc": round(float((pred == NOC).mean()), 4),
                "per_class_f1": {str(k): round(float(v), 4) for k, v in
                                 zip(labels, f1_score(NOC, pred, average=None, labels=labels, zero_division=0))}}
    probs = np.concatenate(Pd); yp = topk_decode(probs, NOC)
    return {"em": per_noc_em(Y, yp, NOC),
            "micro_f1": round(float(f1_score(Y, yp, average="micro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(Y, yp, average="macro", zero_division=0)), 4)}


def run_fold(cfg, sd, tr_idx, va_idx, tokens, mask, y, noc):
    tr_dl = DataLoader(PoolDataset(tokens[tr_idx], mask[tr_idx], y[tr_idx], noc[tr_idx]),
                       batch_size=cfg.batch_size, shuffle=True)
    va_dl = DataLoader(PoolDataset(tokens[va_idx], mask[va_idx], y[va_idx], noc[va_idx]),
                       batch_size=512, shuffle=False)
    model = build_model(sd, cfg.init, tokens[tr_idx], mask[tr_idx])
    if cfg.task == "count":
        w = 1.0 / np.clip(np.bincount(noc[tr_idx] - 1, minlength=5).astype(float), 1, None)
        cw = torch.tensor(w / w.mean(), dtype=torch.float32, device=DEVICE)

        def loss_fn(out, yb, nb):
            return F.cross_entropy(out["logits_card"], nb - 1, weight=cw)
        sel = "macro_f1"
    else:
        asl = AsymmetricLoss()

        def loss_fn(out, yb, nb):
            return asl(out["logits_cls"], yb)
        sel = "micro_f1"

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    best, best_state, wait = -1.0, None, 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for tok, msk, yb, nb in tr_dl:
            out = model(tok.to(DEVICE), msk.to(DEVICE))
            loss = loss_fn(out, yb.to(DEVICE), nb.to(DEVICE))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        m = evaluate(model, va_dl, cfg.task)[sel]
        if m > best:
            best, wait = m, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep % 10 == 0 or ep == 1:
            print(f"    ep{ep:3d}  val_{sel}={m:.4f}  best={best:.4f}")
        if wait >= cfg.patience:
            print(f"    early stop @ ep{ep}  best={best:.4f}"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, va_dl, cfg.task)


def run_kfold(cfg, sd, tokens, mask, y, noc, groups):
    gkf = GroupKFold(n_splits=cfg.n_folds)
    per_fold = []
    print(f"\nGroupKFold({cfg.n_folds})  task={cfg.task}  init={cfg.init}  device={DEVICE}  "
          f"pool={len(tokens)}  groups={len(np.unique(groups))}")
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(tokens, y, groups), 1):
        t0 = time.time(); torch.manual_seed(cfg.seed + fold); np.random.seed(cfg.seed + fold)
        print(f"\n  Fold {fold}/{cfg.n_folds}  train={len(tr_idx)}  val={len(va_idx)}  "
              f"val_groups={len(np.unique(groups[va_idx]))}")
        r = run_fold(cfg, sd, tr_idx, va_idx, tokens, mask, y, noc); r["fold"] = fold
        per_fold.append(r)
        extra = f"acc={r['acc']:.4f}" if cfg.task == "count" else f"EM={r['em']['overall']:.4f}"
        print(f"  Fold {fold}: macro-F1={r['macro_f1']:.4f}  micro-F1={r['micro_f1']:.4f}  "
              f"{extra}  ({time.time()-t0:.0f}s)")
    def ms(key):
        v = [r[key] for r in per_fold]
        return round(float(np.mean(v)), 4), round(float(np.std(v)), 4)
    macro_m, macro_s = ms("macro_f1"); micro_m, micro_s = ms("micro_f1")
    results = {
        "model": "Donor-Slot SetTransformer (inc22)", "task": cfg.task, "init": cfg.init,
        "per_fold": per_fold,
        "macro_f1_mean": macro_m, "macro_f1_std": macro_s,
        "micro_f1_mean": micro_m, "micro_f1_std": micro_s,
        "config": {"n_folds": cfg.n_folds, "seed": cfg.seed, "epochs": cfg.epochs, "lr": cfg.lr},
    }
    if cfg.task == "count":
        results["acc_mean"], results["acc_std"] = ms("acc")
    if cfg.init == "pretrained":
        results["leakage_caveat"] = ("checkpoint trained on part of this GF pool; "
                                     "finetune+eval on the folds is optimistic — use --init scratch.")
    print(f"\n{'='*64}\nDonor-Slot SetTransformer — GroupKFold({cfg.n_folds}) — "
          f"task={cfg.task} init={cfg.init}")
    print(f"  Macro-F1 : {macro_m:.4f} ± {macro_s:.4f}")
    print(f"  Micro-F1 : {micro_m:.4f} ± {micro_s:.4f}")
    if cfg.task == "count":
        print(f"  Accuracy : {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
    print('='*64)
    return results


def resolve_config(argv=None):
    p = argparse.ArgumentParser(description="GroupKFold(5) Donor-Slot SetTransformer.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--task", choices=["count", "donor"], default="count")
    p.add_argument("--init", choices=["pretrained", "scratch"], default="scratch")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
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
    tag = f"{cfg.task}_{cfg.init}"
    (out / f"kfold_metrics_{tag}.json").write_text(json.dumps(results, indent=2))
    scores = [r["acc"] if cfg.task == "count" else r["em"]["overall"] for r in results["per_fold"]]
    (out / f"per_fold_scores_{tag}.json").write_text(json.dumps(
        {f"DonorSlot_{tag}": scores}, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

"""
train_inc6_sam.py — Increment 6 4a (SEPARATE CODE; does not touch the monolithic train loop).

Sharpness-Aware Minimization (Foret et al. 2021, arXiv 2010.01412) on the pe_s3+sparse base. SAM seeks
FLAT minima (min over an rho-ball), which generalize better OOD / combinatorially — the no-train probe
found the NOVEL-N5 minimum is 2.1x sharper than the SEEN one (feasibility_inc6b.py), so flat-seeking has
headroom on the N5 wall. Two forward-backward passes/step: (1) grad at w, ascend w += rho*g/||g||, (2)
grad at w+e, restore w, step with the perturbed-point grad.

Self-contained (isolated from the 20+ existing arms): core losses = cls (ASL) + cost-sensitive NOC CE +
Kendall-weighted aux (attr+phi). The reject/open-set head is OMITTED — it reads its own pool and never
touches the per-donor cls ranking, so it is irrelevant to the DEV N5 ID-oracle this arm is judged on.
measure_insilico_oracle.py is auto-invoked at the end (writes the generalization block).

Usage: python train_inc6_sam.py [--seed 42] [--out_subdir inc6_sam] [--epochs N] [--sam_rho 0.05]
Smoke: SAM_SMOKE=1 STR_DATA_DIR=data_smoke6 python train_inc6_sam.py --n_token_feats 3 --epochs 1
"""
import os, sys, json, argparse, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_set_transformer import (ClosedSetDataset, set_seed, AsymmetricLoss, cardinality_target,
                                   evaluate_oracle_em, DEVICE, DATA_DIR, ROOT)
from models.set_transformer import SetTransformerMixture


def base_cfg(n_tok=8):
    return dict(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32,
                n_classes=45, n_noc=6, dropout=0.1, cls_decoder="per_donor", decoder_source="encoded",
                n_token_feats=n_tok, encoder="isab++", dec_layers=2, num_embed="periodic", n_freq=8,
                d_num_emb=8, periodic_sigma=0.3, aux_heads=True, sparse_attn=True)


def train_sam(seed, out_subdir, n_tok, epochs, rho, lr=6e-4, bs=256, beta=0.3, card_lam=0.02):
    set_seed(seed)
    tp = f"tokens{n_tok}" if n_tok > 3 else "tokens"
    tr = ClosedSetDataset("train", tp)
    sel_split = "dev" if (DATA_DIR / f"{tp}_dev.npy").exists() else "val"
    sel = ClosedSetDataset(sel_split, tp)
    trL = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0)
    selL = DataLoader(sel, batch_size=256, shuffle=False, num_workers=0)
    print(f"SAM (rho={rho}) | selection={sel_split} ({len(sel)})")

    cfg = base_cfg(n_tok)
    model = SetTransformerMixture(**cfg).to(DEVICE)
    if n_tok > 3:
        _tk = tr.tokens.numpy(); _mk = tr.mask.numpy().astype(bool); _num = _tk[:, :, 1:n_tok][_mk]
        model.feat_mean.copy_(torch.tensor(_num.mean(0), dtype=torch.float32, device=DEVICE))
        model.feat_std.copy_(torch.tensor(_num.std(0) + 1e-6, dtype=torch.float32, device=DEVICE))

    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    _nc = np.bincount(np.clip(np.load(DATA_DIR / "noc_train.npy"), 1, 5) - 1, minlength=5).astype(float)
    _w = 1.0 / np.clip(_nc, 1, None); _w = _w / _w.mean()
    card_w = torch.tensor(np.clip(_w, 0.5, 2.0), dtype=torch.float32, device=DEVICE)
    log_var_attr = torch.zeros((), device=DEVICE, requires_grad=True)
    log_var_phi = torch.zeros((), device=DEVICE, requires_grad=True)
    opt = torch.optim.AdamW(list(model.parameters()) + [log_var_attr, log_var_phi], lr=lr, weight_decay=1e-4)
    # match the base recipe: decay LR on the dev-oracle plateau (SAM without this trains at a fixed
    # high LR the whole run and never fine-converges — confounds the flat-minima comparison).
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5, min_lr=1e-6)

    def core_loss(tokens, mask, y, noc, attr, phi):
        out = model(tokens, mask)
        loss = asl(out["logits_cls"], y)
        card_tgt = cardinality_target(torch.sigmoid(out["logits_cls"]).detach(), y, card_lam)
        loss = loss + beta * F.cross_entropy(out["logits_card"], card_tgt, weight=card_w)
        if (attr >= 0).any():
            la = out["logits_attr"]; B_, S_, C_ = la.shape
            la_ = F.cross_entropy(la.reshape(B_ * S_, C_), attr.reshape(B_ * S_), ignore_index=-1)
            lp_ = F.l1_loss(out["phi"], phi)
            loss = loss + (torch.exp(-log_var_attr) * la_ + log_var_attr
                           + torch.exp(-log_var_phi) * lp_ + log_var_phi)
        return loss

    params = [p for p in model.parameters() if p.requires_grad]
    best, best_state, patience, bad = -1.0, None, 25, 0
    EP = epochs or 150
    print(f"Training up to {EP} epochs (SAM 2-step) ...")
    for ep in range(1, EP + 1):
        model.train()
        for tokens, mask, y, noc, attr, phi in trL:
            tokens, mask, y = tokens.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            noc, attr, phi = noc.to(DEVICE), attr.to(DEVICE), phi.to(DEVICE)
            # SAM step 1: gradient at w
            opt.zero_grad(); core_loss(tokens, mask, y, noc, attr, phi).backward()
            with torch.no_grad():
                gnorm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None)) + 1e-12
                ew = []
                for p in params:
                    e = (rho / gnorm) * p.grad if p.grad is not None else None
                    ew.append(e)
                    if e is not None:
                        p.add_(e)
            # SAM step 2: gradient at w+e, then restore and step
            opt.zero_grad(); core_loss(tokens, mask, y, noc, attr, phi).backward()
            with torch.no_grad():
                for p, e in zip(params, ew):
                    if e is not None:
                        p.sub_(e)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        em, _ = evaluate_oracle_em(model, selL)
        sched.step(em)
        if em > best:
            best, best_state, bad = em, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep}  {sel_split}_oracleEM={em:.4f}  best={best:.4f}")
        if bad >= patience:
            print(f"  early stop ep{ep} (best {best:.4f})"); break
    model.load_state_dict(best_state)

    out_dir = ROOT / "results" / out_subdir; out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    json.dump({"model": "inc6_sam", "config": {**cfg, "seed": seed, "out_subdir": out_subdir,
              "sam_rho": rho}}, open(out_dir / "metrics.json", "w"), indent=2)
    print(f"wrote {out_dir/'metrics.json'}")
    if not os.environ.get("SAM_SMOKE"):
        subprocess.run([sys.executable, "measure_insilico_oracle.py", str(out_dir), str(DATA_DIR)],
                       cwd=str(ROOT), check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_subdir", type=str, default="inc6_sam")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n_token_feats", type=int, default=8)
    ap.add_argument("--sam_rho", type=float, default=0.05)
    a = ap.parse_args()
    train_sam(a.seed, a.out_subdir, a.n_token_feats, a.epochs, a.sam_rho)

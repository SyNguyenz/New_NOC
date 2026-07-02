"""
train_p5_noc_intrinsic.py — P5 (SEPARATE CODE; does not modify existing files).

NOC INTRINSIC head: a CORN ordinal count read from a DEDICATED encoder pool (own PMA over H), NOT
from the ID-profile bottleneck `CardinalityHead(sigmoid(logits_cls).detach())`. F30 measured that the
deployed (ID-profile) count COMBO-OVERFITS (~30pp train→dev gap) because it inherits the ID profile's
overfit. P5 tests whether an intrinsic count (own pool, combo-invariant aggregate features) generalizes
better. It trains BOTH heads (intrinsic CORN + the ID-profile card) so their dev-vs-train count gap is
compared in ONE run.

Reuses (imports, no edits): the data/loss/eval harness from train_set_transformer.py and the building
blocks PMA + SetTransformerMixture from models.set_transformer + CORN from models.ordinal.
Base = adopted pe_s3 + sparse (tok8 · periodic σ0.3 · isab++ · per_donor · aux · sparse_attn).

Usage: python train_p5_noc_intrinsic.py [--seed 42] [--out_subdir inc4_p5_noc_intrinsic] [--epochs N]
Smoke:  P5_SMOKE=1 python train_p5_noc_intrinsic.py
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_set_transformer import (ClosedSetDataset, AsymmetricLoss, evaluate_oracle_em,
                                    per_noc_em, topk_decode, set_seed, DEVICE, DATA_DIR)
from models.set_transformer import SetTransformerMixture, PMA
from models.ordinal import corn_loss, corn_probs
from torch.utils.data import DataLoader

N_NOC = 5


def base_kwargs(d_model=128, n_heads=4, n_isab=2, m_ind=32, dropout=0.1):
    return dict(n_loci=24, d_locus=16, d_model=d_model, n_heads=n_heads, n_isab=n_isab,
                m_inducing=m_ind, n_classes=45, n_noc=6, dropout=dropout, cls_decoder="per_donor",
                decoder_source="encoded", n_token_feats=8, encoder="isab++", dec_layers=2,
                num_embed="periodic", n_freq=8, d_num_emb=8, periodic_sigma=0.3,
                aux_heads=True, sparse_attn=True)


class P5Model(nn.Module):
    """Adopted base (reused) + an INTRINSIC CORN NOC head on its own PMA pool over H."""
    def __init__(self, d_model=128, n_heads=4, dropout=0.1, **bk):
        super().__init__()
        self.base = SetTransformerMixture(**base_kwargs(d_model=d_model, n_heads=n_heads, dropout=dropout, **bk))
        self.noc_pool = PMA(d_model, n_heads, 1, dropout)        # dedicated count pool (NOT ID profile)
        self.corn = nn.Linear(d_model, N_NOC - 1)               # CORN: 5 ranks -> 4 conditional logits

    def forward(self, tokens, mask):
        x0, H, pad = self.base._encode_set(tokens, mask)
        z = self.base.pma(H, pad_mask=pad).squeeze(1)
        out = {"logits_cls": self.base.cls_decoder_module(H, pad_mask=pad)}
        out["logits_card"] = self.base.cardinality_head(torch.sigmoid(out["logits_cls"]).detach())  # ID-profile count
        if getattr(self.base, "aux_heads", False):
            out["logits_attr"] = self.base.attr_head(H)
            out["phi"] = F.softplus(self.base.phi_head(z))
        out["logits_corn"] = self.corn(self.noc_pool(H, pad_mask=pad).squeeze(1))                    # intrinsic count
        return out


def _smoke():
    torch.manual_seed(0)
    m = P5Model(d_model=32, n_heads=2, n_isab=1, m_ind=8).to(DEVICE)
    tok = torch.rand(8, 40, 8, device=DEVICE); msk = torch.ones(8, 40, dtype=torch.bool, device=DEVICE)
    out = m(tok, msk)
    y = (torch.rand(8, 45) > 0.9).float().to(DEVICE); noc = torch.randint(1, 6, (8,), device=DEVICE)
    loss = AsymmetricLoss()(out["logits_cls"], y) + corn_loss(out["logits_corn"], noc, N_NOC) \
        + F.cross_entropy(out["logits_card"], noc - 1)
    loss.backward()
    assert out["logits_corn"].shape == (8, 4) and out["logits_card"].shape == (8, 5)
    print(f"P5 SMOKE OK  loss={loss.item():.3f}  corn{tuple(out['logits_corn'].shape)} card{tuple(out['logits_card'].shape)}")


def count_gen(model, arrs_by_split):
    """per-NOC count accuracy for BOTH heads on each split → the F30 comparison."""
    res = {}
    for name, (tok, msk, noc) in arrs_by_split.items():
        kc, kk = [], []
        with torch.no_grad():
            for i in range(0, len(tok), 256):
                o = model(torch.from_numpy(tok[i:i+256]).to(DEVICE), torch.from_numpy(msk[i:i+256]).to(DEVICE))
                kc.append(corn_probs(o["logits_corn"], N_NOC).argmax(1).cpu().numpy() + 1)
                kk.append(o["logits_card"].argmax(1).cpu().numpy() + 1)
        kc = np.concatenate(kc); kk = np.concatenate(kk); nocc = np.clip(noc, 1, 5)
        res[name] = {"intrinsic": {int(k): float((kc[nocc == k] == k).mean()) for k in range(1, 6) if (nocc == k).any()},
                     "id_profile": {int(k): float((kk[nocc == k] == k).mean()) for k in range(1, 6) if (nocc == k).any()}}
    return res


def train_p5(seed, out_subdir, epochs):
    set_seed(seed)
    tp = "tokens8"
    tr = ClosedSetDataset("train", tp); va = ClosedSetDataset("test", tp)
    dev = ClosedSetDataset("dev", tp) if (DATA_DIR / f"{tp}_dev.npy").exists() else va
    trL = DataLoader(tr, batch_size=256, shuffle=True, num_workers=0)
    devL = DataLoader(dev, batch_size=256, shuffle=False)
    model = P5Model().to(DEVICE)
    # per-feature standardization from train valid peaks (as train_set_transformer does for enriched tokens)
    _tk = tr.tokens.numpy(); _mk = tr.mask.numpy().astype(bool); _num = _tk[:, :, 1:8][_mk]
    model.base.feat_mean.copy_(torch.tensor(_num.mean(0), dtype=torch.float32, device=DEVICE))
    model.base.feat_std.copy_(torch.tensor(_num.std(0) + 1e-6, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=6e-4, weight_decay=1e-4)
    asl = AsymmetricLoss()
    best, best_state = -1, None
    EP = epochs or 150
    for ep in range(1, EP + 1):
        model.train()
        for tok, msk, y, noc, attr, phi in trL:
            tok, msk, y, noc = tok.to(DEVICE), msk.to(DEVICE), y.float().to(DEVICE), noc.to(DEVICE)
            attr, phi = attr.to(DEVICE), phi.to(DEVICE)
            o = model(tok, msk)
            loss = asl(o["logits_cls"], y)
            loss = loss + F.cross_entropy(o["logits_card"], noc.clamp(1, 5) - 1)
            loss = loss + 0.3 * corn_loss(o["logits_corn"], noc.clamp(1, 5), N_NOC)
            if "logits_attr" in o:
                la = F.cross_entropy(o["logits_attr"].reshape(-1, o["logits_attr"].size(-1)),
                                     attr.reshape(-1), ignore_index=-1)
                lp = F.mse_loss(o["phi"], phi)
                loss = loss + 0.3 * la + 0.1 * lp
            opt.zero_grad(); loss.backward(); opt.step()
        _, macrec = evaluate_oracle_em(model, devL)
        if macrec > best:
            best, best_state = macrec, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0:
            print(f"  ep{ep} dev_macro_recall={macrec:.4f} best={best:.4f}")
    model.load_state_dict(best_state)

    # ── eval: per-NOC oracle (ID) on real test + count generalization for BOTH heads ──
    out_dir = Path("results") / out_subdir; out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    te = va

    @torch.no_grad()
    def probs(ds):
        P = []
        for i in range(0, len(ds.tokens), 256):
            o = model(ds.tokens[i:i+256].to(DEVICE), ds.mask[i:i+256].to(DEVICE))
            P.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
        return np.concatenate(P)
    P_te = probs(te); noc_te = te.noc.numpy(); y_te = te.y.numpy()
    oracle = per_noc_em(y_te, topk_decode(P_te, noc_te), noc_te)

    def splarr(ds, n=4000):
        idx = np.random.default_rng(1).choice(len(ds.tokens), size=min(n, len(ds.tokens)), replace=False)
        return ds.tokens.numpy()[idx], ds.mask.numpy()[idx], ds.noc.numpy()[idx]
    gen = count_gen(model, {
        "dev": (dev.tokens.numpy(), dev.mask.numpy(), dev.noc.numpy()),
        "train": splarr(tr), "test": (te.tokens.numpy(), te.mask.numpy(), te.noc.numpy())})

    metrics = {"model": "p5_noc_intrinsic", "config": {**base_kwargs(), "seed": seed, "out_subdir": out_subdir,
               "noc_intrinsic": True}, "oracle_em": round(float(oracle[0]), 4),
               "per_noc_oracle": {str(k): round(float(v), 4) for k, v in zip(range(1, 6), oracle[1:])},
               "count_generalization": gen}
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)
    print("\n== P5 count generalization (intrinsic vs id_profile), N5 ==")
    for sp in ("train", "dev", "test"):
        print(f"  {sp:>5}: intrinsic N5={gen[sp]['intrinsic'].get(5)}  id_profile N5={gen[sp]['id_profile'].get(5)}")
    print(f"oracle_em={metrics['oracle_em']}  per_noc_oracle={metrics['per_noc_oracle']}")
    print(f"wrote {out_dir/'metrics.json'}")


if __name__ == "__main__":
    if os.environ.get("P5_SMOKE"):
        _smoke(); sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_subdir", type=str, default="inc4_p5_noc_intrinsic")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    train_p5(a.seed, a.out_subdir, a.epochs)

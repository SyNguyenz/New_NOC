"""
train_inc6_meta.py — Increment 6 B (SEPARATE CODE; does not modify existing files).

META-LEARNING root lever for the COMBINATORIAL-GENERALIZATION wall (F29/F30): train N5 oracle ~1.0
-> held-out-combo DEV ~0.65. The decoder MEMORIZES training donor-combos and fails to COMPOSE known
donors into a novel combo. Peeling / residual-aug only chip the symptom; this attacks the mechanism.

METHOD (screening proxy of Lake 2019 "Compositional generalization through meta seq2seq learning",
arXiv 1906.05381): REPTILE (Nichol et al. 2018, arXiv 1803.02999) episodic fine-tune of an already-
trained base. Each EPISODE picks a target donor d and fast-adapts (k inner SGD steps) on a mini-batch
of samples that all contain d but in VARYING co-contributor contexts; the Reptile outer step moves the
base weights toward the adapted weights. Repeating this over donors pressures the model to encode each
donor as a context-invariant, *composable* unit rather than a per-combo template.

HONEST SCOPE (C8-style): this is the screening proxy, NOT full meta-learning. Reptile averages adapted
weights; it lacks the explicit query objective of FOMAML/MAML. If the DEV N5 oracle moves vs the base,
that is the signal to invest in the strong FOMAML version (support/query-disjoint combos). If it does
not move, meta-finetune-lite is insufficient and the lever is the heavier episodic re-train. Judge on
the DEV per-NOC oracle (measure_insilico_oracle.py, auto-invoked at the end), NOT aggregate EM.

Usage:
  python train_inc6_meta.py --init_from inc5_res_rand1_seed42 --out_subdir inc6_meta --seed 42
Smoke (tok3, fresh init, tiny):
  META_SMOKE=1 STR_DATA_DIR=data_smoke6 python train_inc6_meta.py --n_token_feats 3 --meta_iters 4 \
      --out_subdir smoke_meta
"""
import os, sys, json, argparse, copy, subprocess
from pathlib import Path
import numpy as np
import torch

from train_set_transformer import ClosedSetDataset, set_seed, AsymmetricLoss, DEVICE, DATA_DIR, ROOT
from models.set_transformer import SetTransformerMixture


def base_cfg(n_tok=8):
    # MUST mirror the inc5/base training flags so the checkpoint state_dict loads and
    # measure_insilico_oracle rebuilds the identical architecture from this config.
    return dict(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32,
                n_classes=45, n_noc=6, dropout=0.1, cls_decoder="per_donor", decoder_source="encoded",
                n_token_feats=n_tok, encoder="isab++", dec_layers=2, num_embed="periodic", n_freq=8,
                d_num_emb=8, periodic_sigma=0.3, aux_heads=True, sparse_attn=True)


def build(cfg):
    return SetTransformerMixture(**cfg).to(DEVICE)


def meta_finetune(seed, out_subdir, init_from, n_tok, meta_iters, inner_steps, inner_lr, eps, pool):
    set_seed(seed)
    tp = f"tokens{n_tok}" if n_tok > 3 else "tokens"
    tr = ClosedSetDataset("train", tp)
    tokens, mask, y = tr.tokens, tr.mask, tr.y.float()
    cfg = base_cfg(n_tok)
    model = build(cfg)

    # set per-feature standardization buffers from train (for fresh init; a checkpoint overwrites them)
    if n_tok > 3:
        _tk = tr.tokens.numpy(); _mk = tr.mask.numpy().astype(bool); _num = _tk[:, :, 1:n_tok][_mk]
        model.feat_mean.copy_(torch.tensor(_num.mean(0), dtype=torch.float32, device=DEVICE))
        model.feat_std.copy_(torch.tensor(_num.std(0) + 1e-6, dtype=torch.float32, device=DEVICE))

    if init_from:
        ckpt = ROOT / "results" / init_from / "best_model.pt"
        assert ckpt.exists(), f"--init_from checkpoint not found: {ckpt}"
        sd = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
        model.load_state_dict(sd, strict=False)
        print(f"meta: warm-start from {ckpt}")
    else:
        print("meta: FRESH init (no --init_from) — mechanics test / smoke only")

    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    yb = (y.numpy() > 0.5)
    donor_idx = {d: np.where(yb[:, d])[0] for d in range(45)}
    donors_present = [d for d in range(45) if len(donor_idx[d]) >= max(2, pool)]
    assert donors_present, "no donor has enough samples to form an episode"
    rng = np.random.default_rng(seed)

    print(f"meta-finetune: {meta_iters} episodes × {inner_steps} inner-steps "
          f"(lr={inner_lr}, eps={eps}, pool={pool}, donors={len(donors_present)})")
    model.train()
    for it in range(1, meta_iters + 1):
        d = donors_present[int(rng.integers(len(donors_present)))]
        pool_idx = donor_idx[d]
        theta0 = copy.deepcopy(model.state_dict())                 # snapshot for the Reptile step
        opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
        last = 0.0
        for _ in range(inner_steps):
            bi = rng.choice(pool_idx, size=min(pool, len(pool_idx)), replace=False)
            tb, mb, yy = tokens[bi].to(DEVICE), mask[bi].to(DEVICE), y[bi].to(DEVICE)
            loss = asl(model(tb, mb)["logits_cls"], yy)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            last = loss.item()
        # Reptile outer update: θ ← θ0 + eps·(θ_adapted − θ0)  (float params only; buffers kept as-is)
        adapted = model.state_dict()
        merged = {k: (theta0[k] + eps * (adapted[k] - theta0[k])
                      if torch.is_floating_point(adapted[k]) else adapted[k]) for k in adapted}
        model.load_state_dict(merged)
        if it % max(1, meta_iters // 10) == 0:
            print(f"  meta-iter {it}/{meta_iters}  (donor {d}, last inner loss {last:.3f})")

    out_dir = ROOT / "results" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    metrics = {"model": "inc6_meta_reptile",
               "config": {**cfg, "seed": seed, "out_subdir": out_subdir, "init_from": init_from,
                          "meta_iters": meta_iters, "inner_steps": inner_steps, "inner_lr": inner_lr,
                          "eps": eps, "pool": pool}}
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)
    print(f"wrote {out_dir/'metrics.json'}")

    # self-measure DEV/train/real per-NOC oracle (like P5/P6 do) unless smoking
    if not os.environ.get("META_SMOKE"):
        subprocess.run([sys.executable, "measure_insilico_oracle.py", str(out_dir), str(DATA_DIR)],
                       cwd=str(ROOT), check=False)


def fomaml_finetune(seed, out_subdir, init_from, n_tok, meta_iters, inner_steps, inner_lr, meta_lr,
                    support, query):
    """Inc6 4c+: FIRST-ORDER MAML (Finn 2017 / Nichol 2018) — the STRONGER meta variant of B. Each
    episode adapts on a SUPPORT set of donor d's samples, then takes the OUTER step along the gradient
    of the QUERY loss (held-out, donor-context-DISJOINT samples of the same donor) evaluated at the
    adapted weights. Unlike Reptile (weight-averaging) this has the explicit query objective = "after
    seeing donor d in some contexts, identify it in a NOVEL context" → directly the composition skill."""
    set_seed(seed)
    tp = f"tokens{n_tok}" if n_tok > 3 else "tokens"
    tr = ClosedSetDataset("train", tp)
    tokens, mask, y = tr.tokens, tr.mask, tr.y.float()
    cfg = base_cfg(n_tok)
    model = build(cfg)
    if n_tok > 3:
        _tk = tr.tokens.numpy(); _mk = tr.mask.numpy().astype(bool); _num = _tk[:, :, 1:n_tok][_mk]
        model.feat_mean.copy_(torch.tensor(_num.mean(0), dtype=torch.float32, device=DEVICE))
        model.feat_std.copy_(torch.tensor(_num.std(0) + 1e-6, dtype=torch.float32, device=DEVICE))
    if init_from:
        ckpt = ROOT / "results" / init_from / "best_model.pt"
        assert ckpt.exists(), f"--init_from checkpoint not found: {ckpt}"
        sd = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
        model.load_state_dict(sd, strict=False); print(f"fomaml: warm-start from {ckpt}")
    else:
        print("fomaml: FRESH init (smoke only)")

    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    yb = (y.numpy() > 0.5)
    donor_idx = {d: np.where(yb[:, d])[0] for d in range(45)}
    need = support + query
    donors_present = [d for d in range(45) if len(donor_idx[d]) >= need]
    assert donors_present, f"no donor has >= {need} samples for support+query"
    rng = np.random.default_rng(seed)
    meta_opt = torch.optim.AdamW(model.parameters(), lr=meta_lr, weight_decay=0.0)
    print(f"FOMAML: {meta_iters} episodes | inner {inner_steps}@lr{inner_lr} | meta_lr {meta_lr} | "
          f"support {support}/query {query} | donors {len(donors_present)}")

    def batch(bi):
        return tokens[bi].to(DEVICE), mask[bi].to(DEVICE), y[bi].to(DEVICE)

    model.train()
    for it in range(1, meta_iters + 1):
        d = donors_present[int(rng.integers(len(donors_present)))]
        pool = rng.permutation(donor_idx[d])
        sup, qry = pool[:support], pool[support:support + query]      # context-disjoint
        theta0 = {k: v.clone() for k, v in model.state_dict().items()}
        inner = torch.optim.SGD(model.parameters(), lr=inner_lr)
        for _ in range(inner_steps):
            tb, mb, yy = batch(rng.choice(sup, size=min(support, len(sup)), replace=False))
            inner.zero_grad(); asl(model(tb, mb)["logits_cls"], yy).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); inner.step()
        # outer: query gradient at the adapted weights (first-order)
        tb, mb, yy = batch(qry)
        meta_opt.zero_grad(); lq = asl(model(tb, mb)["logits_cls"], yy); lq.backward()
        model.load_state_dict(theta0)                                 # restore θ0 (grads kept)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); meta_opt.step()
        if it % max(1, meta_iters // 10) == 0:
            print(f"  fomaml-iter {it}/{meta_iters}  (donor {d}, query loss {lq.item():.3f})")

    out_dir = ROOT / "results" / out_subdir; out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    json.dump({"model": "inc6_fomaml", "config": {**cfg, "seed": seed, "out_subdir": out_subdir,
              "init_from": init_from, "meta_iters": meta_iters, "inner_steps": inner_steps,
              "inner_lr": inner_lr, "meta_lr": meta_lr, "support": support, "query": query}},
              open(out_dir / "metrics.json", "w"), indent=2)
    print(f"wrote {out_dir/'metrics.json'}")
    if not os.environ.get("META_SMOKE"):
        subprocess.run([sys.executable, "measure_insilico_oracle.py", str(out_dir), str(DATA_DIR)],
                       cwd=str(ROOT), check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_subdir", type=str, default="inc6_meta")
    ap.add_argument("--init_from", type=str, default=None,
                    help="results subdir of a trained base to warm-start (e.g. inc5_res_rand1_seed42)")
    ap.add_argument("--n_token_feats", type=int, default=8)
    ap.add_argument("--meta_iters", type=int, default=400, help="number of Reptile episodes")
    ap.add_argument("--epochs", type=int, default=None, help="alias that overrides --meta_iters (runner --epochs)")
    ap.add_argument("--inner_steps", type=int, default=5)
    ap.add_argument("--inner_lr", type=float, default=1e-4)
    ap.add_argument("--eps", type=float, default=0.1, help="Reptile outer step size")
    ap.add_argument("--pool", type=int, default=16, help="episode mini-batch size (samples per inner step)")
    ap.add_argument("--fomaml", action="store_true", help="4c+: first-order MAML (query objective) instead of Reptile")
    ap.add_argument("--meta_lr", type=float, default=1e-4, help="FOMAML outer step size")
    ap.add_argument("--support", type=int, default=32, help="FOMAML support set size")
    ap.add_argument("--query", type=int, default=32, help="FOMAML query set size")
    a = ap.parse_args()
    iters = a.epochs if a.epochs else a.meta_iters
    if a.fomaml:
        fomaml_finetune(a.seed, a.out_subdir, a.init_from, a.n_token_feats, iters,
                        a.inner_steps, a.inner_lr, a.meta_lr, a.support, a.query)
    else:
        meta_finetune(a.seed, a.out_subdir, a.init_from, a.n_token_feats, iters,
                      a.inner_steps, a.inner_lr, a.eps, a.pool)

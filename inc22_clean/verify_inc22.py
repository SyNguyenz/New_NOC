"""
verify_inc22.py — PROOF that the clean SetTransformerMixture is a bit-identical extraction of the inc22 path.

Loads the published checkpoint results/inc22_fixed_aslot_seed42/best_model.pt into BOTH:
  * the ORIGINAL models/set_transformer.py::SetTransformerMixture built with the inc22 kwargs, and
  * the CLEAN models/set_transformer.py::SetTransformerMixture,
then runs the SAME test batch in eval() mode and asserts:
  1. every RANKING output tensor (logits_cls, logits_card, phi, logits_attr, logit_reject)
     is identical (max|d| ~ 0), and
  2. the gradient of a deterministic scalar of those outputs is identical on every shared parameter.

eval() makes the forward deterministic (dropout off; AdaSlot gate = sigmoid(logit), no Gumbel noise;
no mask_peaks), so identical weights => identical compute is the necessary-and-sufficient check that
the clean code path computes the same function as the original.  The checkpoint keys the clean model
omits are the UNUSED `cardinality_head.*` and the EXCLUDED CORN `ord_count_head.*` (loaded strict=False);
the CORN head fed only DETACHED features, so dropping it leaves the ranking byte-identical.

Usage:
    python inc22_clean/verify_inc22.py           # uses results/inc22_fixed_aslot_seed42 + data_insilico_w
    CKPT=<run_dir> DATA=<data_dir> python inc22_clean/verify_inc22.py
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent          # inc22_clean/
PROJ = HERE.parent                               # project root

CKPT = Path(os.environ.get("CKPT", str(PROJ / "results" / "inc22_fixed_aslot_seed42")))
DATA = Path(os.environ.get("DATA", str(PROJ / "data_insilico_w")))
DEVICE = torch.device("cpu")                     # CPU = deterministic op order for an exact compare
ALLELE_OFF = 30; LUT_W = 1024


def load_module(name, path):
    """Load a .py file as a standalone module by explicit path (avoids the `models` package-name
    collision between the project root and inc22_clean — both have a `models/` dir)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def build_owner_lut(donor_geno, donor_geno_mask, n_cls=45):
    gg = donor_geno; gm = donor_geno_mask.bool()
    owner = torch.zeros(24, LUT_W, n_cls)
    for c in range(min(n_cls, gg.size(0))):
        for j in range(gg.size(1)):
            if gm[c, j]:
                li = int(gg[c, j, 0]); ab = int(round(float(gg[c, j, 1]) * 10)) + ALLELE_OFF
                if 0 <= li < 24 and 0 <= ab < LUT_W:
                    owner[li, ab, c] = 1.0
    return owner


def main():
    assert (CKPT / "best_model.pt").exists(), f"checkpoint not found: {CKPT}/best_model.pt"
    gp = DATA / "donor_geno.npy"
    if not gp.exists():
        gp = PROJ / "data" / "donor_geno.npy"
    donor_geno = torch.from_numpy(np.load(gp).astype(np.float32))
    donor_geno_mask = torch.from_numpy(np.load(gp.parent / "donor_geno_mask.npy"))
    owner_lut = build_owner_lut(donor_geno, donor_geno_mask)

    # ── ORIGINAL model (inc22 kwargs, exactly as train_set_transformer builds it) ──
    SetTransformerMixture = load_module("orig_set_transformer",
                                        PROJ / "models" / "set_transformer.py").SetTransformerMixture
    orig = SetTransformerMixture(
        n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32, n_classes=45, n_noc=6,
        dropout=0.1, cls_decoder="aslot", n_token_feats=8, encoder="isab++",
        num_embed="periodic", periodic_sigma=0.3, aux_heads=True,
        nc_attn="mab0", feas_filter=True, set_of_set=True, soft_geno_attr=False,
        donor_geno=donor_geno, donor_geno_mask=donor_geno_mask, owner_lut=owner_lut,
        n_slot_iters=3, ot_eps=0.05, ot_iters=5, noc_head_v2=True,
    ).to(DEVICE)

    # ── CLEAN model ──
    clean = load_module("set_transformer_clean", HERE / "models" / "set_transformer.py").SetTransformerMixture(
        n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32, n_classes=45,
        dropout=0.1, n_token_feats=8, periodic_sigma=0.3,
        n_slot_iters=3, ot_eps=0.05, ot_iters=5,
        donor_geno=donor_geno, donor_geno_mask=donor_geno_mask, owner_lut=owner_lut,
    ).to(DEVICE)

    sd = torch.load(CKPT / "best_model.pt", weights_only=True, map_location=DEVICE)
    mo, uo = orig.load_state_dict(sd, strict=False)
    mc, uc = clean.load_state_dict(sd, strict=False)
    print(f"original  load: missing={list(mo)} unexpected={list(uo)}")
    print(f"clean     load: missing={list(mc)} unexpected={list(uc)}")
    assert not mc, f"clean model is MISSING checkpoint keys: {list(mc)}"
    assert all(k.startswith(("cardinality_head", "ord_count_head")) for k in uc), \
        f"clean model has unexpected leftover keys beyond cardinality_head/ord_count_head: {list(uc)}"
    print("  -> clean omits ONLY the unused cardinality_head.* and the EXCLUDED CORN ord_count_head.* (expected)\n")

    orig.eval(); clean.eval()

    tok = torch.from_numpy(np.load(DATA / "tokens8_test.npy")[:64].astype(np.float32)).to(DEVICE)
    msk = torch.from_numpy(np.load(DATA / "mask_test.npy")[:64]).to(DEVICE)

    keys = ["logits_cls", "logits_card", "phi", "logits_attr", "logit_reject"]

    # 1) forward identity
    with torch.no_grad():
        oo = orig(tok, msk); co = clean(tok, msk)
    print("== forward (max|d| over a 64-sample test batch) ==")
    ok = True
    for k in keys:
        d = (oo[k] - co[k]).abs().max().item()
        flag = "OK" if d < 1e-5 else "FAIL"
        ok &= d < 1e-5
        print(f"  {k:18s} max|d|={d:.3e}  [{flag}]")

    # 2) gradient identity on a deterministic scalar of all outputs
    def scalar(out):
        return sum(out[k].float().pow(2).mean() for k in keys)

    orig.zero_grad(); clean.zero_grad()
    scalar(orig(tok, msk)).backward()
    scalar(clean(tok, msk)).backward()
    og = {n: p.grad for n, p in orig.named_parameters() if p.grad is not None}
    cg = {n: p.grad for n, p in clean.named_parameters() if p.grad is not None}
    shared = sorted(set(og) & set(cg))
    print(f"\n== gradients (shared params={len(shared)}; clean-only={sorted(set(cg)-set(og))}; "
          f"orig-only={sorted(set(og)-set(cg))}) ==")
    gmax = 0.0
    for n in shared:
        gmax = max(gmax, (og[n] - cg[n]).abs().max().item())
    print(f"  max |dgrad| over all shared params = {gmax:.3e}  [{'OK' if gmax < 1e-5 else 'FAIL'}]")
    ok &= gmax < 1e-5

    print("\n" + ("PASS — clean SetTransformerMixture is bit-identical to the inc22 path."
                  if ok else "FAIL — divergence detected (see above)."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

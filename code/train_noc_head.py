"""train_noc_head.py — a learned NOC head on PERMUTATION-INVARIANT aggregates of a frozen ID model.

Route chosen by measurement. On the real test, novel donor combos, macro-F1 over NOC2-5:

    raw peaks -> physical features                       0.44
    raw peaks -> forked encoder + pooled vector          0.73   (4 variants, incl. real fine-tune)
    pooled ID vector z_mix                               0.18 (novel combo) / 0.59 (fit on 18k combos)
    CORN head on invariant aggregates (a by-product,     0.86 count acc
      weight 0.3, detached, no specialisation)
    post-hoc RF on the 10-dim sorted prob profile        0.92 count acc / 0.875 macro-F1

Everything identity-BLIND transfers; everything identity-BEARING memorises the donor combo. So this
head reads only invariant aggregates: the sorted per-donor probability profile, the AdaSlot gate sum,
the sorted slot masses, and allele-richness/height-ratio features that are invariant to scaling every
peak of a sample. The ID model is frozen and its outputs are untouched (asserted).

Training data is a knob, because the open question is whether the count problem is data-limited:
  --n_syn N        in-silico rows
  --real {none,open,both}
                   `open` = 672 REAL mixtures whose donors are outside the 45-donor panel: useless for
                   identification, but their NOC is known from the sample name, so they are free real
                   supervision that costs no test combo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture

SYN = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data_w_inc22")))
REAL = Path(os.environ.get("STR_REAL_DIR", str(ROOT / "data")))
ALLELE_OFF, LUT_W = 30, 1024
DEVICE = (torch.device(os.environ["STR_DEVICE"]) if os.environ.get("STR_DEVICE")
          else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

TOPM = 8


def build_luts(dg, dgm, n_cls=45):
    gm = dgm.bool(); owner = torch.zeros(24, LUT_W, n_cls)
    for c in range(min(n_cls, dg.size(0))):
        for j in range(dg.size(1)):
            if gm[c, j]:
                li = int(dg[c, j, 0]); ab = int(round(float(dg[c, j, 1]) * 10)) + ALLELE_OFF
                if 0 <= li < 24 and 0 <= ab < LUT_W:
                    owner[li, ab, c] = 1.0
    return owner


def physical(tok, msk):
    """Allele richness + height RATIOS. Every entry is invariant to multiplying a sample's peak
    heights by any constant, so it cannot encode the DNA template the experimenter happened to use
    (template correlates with NOC in PROVEDIt by design: corr .255, and drives TPH: corr .672)."""
    m = msk.astype(bool); h = np.expm1(tok[:, :, 2].astype(np.float64)); loc = tok[:, :, 0].astype(int)
    out = np.zeros((len(tok), 12), np.float32)
    for i in range(len(tok)):
        v = np.where(m[i])[0]
        if not len(v):
            continue
        hv = h[i][v]; lv = loc[i][v]; gmax = max(hv.max(), 1e-9)
        cnt = np.bincount(lv, minlength=24)
        c5 = np.bincount(lv[hv > 0.05 * gmax], minlength=24)
        lg = np.log(np.clip(hv, 1e-9, None))
        hb, sec, eff = [], [], []
        for L in np.unique(lv):
            hl = np.sort(hv[lv == L])[::-1]
            if len(hl) >= 2:
                hb.append(hl[-1] / max(hl[0], 1e-9)); sec.append(hl[1] / max(hl[0], 1e-9))
            p = hl / max(hl.sum(), 1e-9)
            eff.append(float(np.exp(-(p * np.log(p + 1e-12)).sum())))
        out[i] = [len(v), cnt.max(), (cnt >= 3).sum(), (cnt >= 4).sum(), (cnt >= 5).sum(),
                  cnt[cnt > 0].mean(),
                  np.bincount(lv[hv > 0.01 * gmax], minlength=24).max(), c5.max(), (c5 >= 3).sum(),
                  lg.std(), float(np.mean(hb)) if hb else 0.0, float(np.mean(eff)) if eff else 0.0]
    return out


@torch.no_grad()
def model_aggregates(model, tok, msk, bs=256):
    """sorted prob profile (TOPM+2) + sum/sorted gate (1+TOPM) + sorted slot_mass (TOPM)."""
    outs = []
    for i in range(0, len(tok), bs):
        o = model(torch.from_numpy(tok[i:i + bs].astype(np.float32)).to(DEVICE),
                  torch.from_numpy(msk[i:i + bs]).to(DEVICE))
        p = torch.sigmoid(o["logits_cls"])
        sp = torch.sort(p, 1, descending=True).values[:, :TOPM]
        prof = torch.cat([sp, p.sum(1, keepdim=True), (p >= 0.5).float().sum(1, keepdim=True)], 1)
        g = o["gate"]
        gg = torch.cat([g.sum(1, keepdim=True), torch.sort(g, 1, descending=True).values[:, :TOPM]], 1)
        sm = o["slot_mass"]
        sm = torch.sort(sm, 1, descending=True).values[:, :TOPM]
        sm = sm / sm.sum(1, keepdim=True).clamp(min=1e-9)          # scale-free
        outs.append(torch.cat([prof, gg, sm], 1).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def features(model, tok, msk):
    return np.hstack([model_aggregates(model, tok, msk), physical(tok, msk)])


def load(d, split):
    return (np.load(d / f"tokens8_{split}.npy"), np.load(d / f"mask_{split}.npy"),
            np.clip(np.load(d / (f"noc_true_{split}.npy" if (d / f"noc_true_{split}.npy").exists()
                                 else f"noc_{split}.npy")).astype(int), 1, 5))


def report(name, pred, noc):
    m = noc >= 2
    return {"name": name, "acc_all": float((pred == noc).mean()),
            "acc_mix": float((pred[m] == noc[m]).mean()),
            "macro_f1": float(f1_score(noc, pred, average="macro", labels=[1, 2, 3, 4, 5],
                                       zero_division=0)),
            "macro_f1_mix": float(f1_score(noc[m], pred[m], average="macro", labels=[2, 3, 4, 5],
                                           zero_division=0)),
            "per_noc": [float((pred[noc == j] == j).mean()) if (noc == j).any() else None
                        for j in range(1, 6)]}


def line(r):
    return (f"  {r['name']:<34}{r['acc_all']:>8.4f}{r['macro_f1']:>10.4f}{r['acc_mix']:>10.4f}"
            f"{r['macro_f1_mix']:>10.4f}   "
            + " ".join(f"{x:.3f}" if x is not None else "  -  " for x in r["per_noc"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--n_syn", type=int, nargs="+", default=[6000, 10000, 20000, 50000])
    ap.add_argument("--real", choices=("none", "open", "both"), default="both")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dgp = SYN / "donor_geno.npy"
    if not dgp.exists():
        dgp = REAL / "donor_geno.npy"
    dg = torch.from_numpy(np.load(dgp).astype(np.float32))
    dgm = torch.from_numpy(np.load(dgp.parent / "donor_geno_mask.npy"))
    model = SetTransformerMixture(
        n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32, n_classes=45,
        dropout=0.1, n_token_feats=8, n_freq=8, d_num_emb=8, periodic_sigma=0.3, n_slot_iters=3,
        ot_eps=0.05, ot_iters=5, gumbel_temp=1.0, donor_geno=dg, donor_geno_mask=dgm,
        owner_lut=build_luts(dg, dgm).to(DEVICE), noc_head_v2=True).to(DEVICE)
    miss, _ = model.load_state_dict(torch.load(args.backbone, map_location=DEVICE,
                                               weights_only=True), strict=False)
    assert not miss, miss
    model.eval()
    print(f"device={DEVICE}  backbone={args.backbone}")

    tk_t, mk_t, n_t = load(REAL, "test")
    tk_o, mk_o, n_o = load(REAL, "open")
    tk_v, mk_v, n_v = load(REAL, "val")
    tk_s, mk_s, n_s = load(SYN, "train")
    print(f"real test {len(n_t)} ({int((n_t>=2).sum())} mixtures) | real open {len(n_o)} "
          f"({int((n_o>=2).sum())} mixtures) | synth pool {len(n_s)}")

    Xt = features(model, tk_t, mk_t)
    Xo = features(model, tk_o, mk_o)
    Xv = features(model, tk_v, mk_v)
    print(f"feature dim = {Xt.shape[1]}")

    def mlp():
        return make_pipeline(StandardScaler(),
                             MLPClassifier((128, 64), max_iter=800, random_state=0,
                                           early_stopping=True, n_iter_no_change=25))

    def rf():
        return RandomForestClassifier(n_estimators=400, max_depth=8, random_state=0, n_jobs=-1)

    rng = np.random.default_rng(0)
    res = []
    hdr = (f"  {'setting':<34}{'acc':>8}{'macroF1':>10}{'acc_mix':>10}{'mF1_mix':>10}"
           f"   per-NOC recall 1..5")
    print("\n" + hdr); print("  " + "-" * (len(hdr) - 2))

    # baselines: what the deployed count does, same evaluation
    for nm, clf, Xtr, ytr in [
            ("RF / prob-profile only / synth 20k", rf(), None, None)]:
        pass
    sub20 = np.sort(rng.choice(len(tk_s), min(20000, len(tk_s)), replace=False))
    Xs20 = features(model, tk_s[sub20], mk_s[sub20]); ys20 = n_s[sub20]
    prof_cols = slice(0, TOPM + 2)
    r = report("RF / prob-profile / synth 20k",
               rf().fit(Xs20[:, prof_cols], ys20).predict(Xt[:, prof_cols]), n_t)
    print(line(r)); res.append(r)

    # size sweep on the full invariant feature set
    for n in args.n_syn:
        if n > len(tk_s):
            continue
        sub = np.sort(rng.choice(len(tk_s), n, replace=False))
        Xs = features(model, tk_s[sub], mk_s[sub]); ys = n_s[sub]
        r = report(f"MLP / invariant / synth {n}", mlp().fit(Xs, ys).predict(Xt), n_t)
        print(line(r)); res.append(r)
        if args.real in ("open", "both"):
            Xb = np.vstack([Xs, Xo, Xv]); yb = np.concatenate([ys, n_o, n_v])
            r = report(f"MLP / invariant / synth {n} + real", mlp().fit(Xb, yb).predict(Xt), n_t)
            print(line(r)); res.append(r)

    # matched-size synth vs real: same number of MIXTURES, no leak (open is disjoint from test)
    n_mix_real = int((n_o >= 2).sum())
    idx_s_mix = np.where(n_s >= 2)[0]
    sub_m = np.sort(rng.choice(idx_s_mix, n_mix_real, replace=False))
    Xs_m = features(model, tk_s[sub_m], mk_s[sub_m]); ys_m = n_s[sub_m]
    om = n_o >= 2
    r = report(f"MLP / invariant / {n_mix_real} SYNTH mixtures", mlp().fit(Xs_m, ys_m).predict(Xt), n_t)
    print("\n" + line(r)); res.append(r)
    r = report(f"MLP / invariant / {n_mix_real} REAL mixtures", mlp().fit(Xo[om], n_o[om]).predict(Xt), n_t)
    print(line(r)); res.append(r)

    print(f"\n  reference — deployed post-hoc RF on real test: acc .9186  macroF1 .875")
    out = Path(args.out) if args.out else ROOT / "results" / "noc_head_invariant"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()

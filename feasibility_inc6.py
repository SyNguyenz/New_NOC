"""
feasibility_inc6.py — NO-TRAIN feasibility probes for the 4 Increment-6 root levers.

Reads EXISTING checkpoints (no retrain) and prints a GO / NO-GO signal per lever, in the spirit of
probes_n5.py / quick_peel.py (the F29 no-train probes). ~minutes on local GPU. The point: decide which
of {D-mask, D-andmask, C-slot, B-meta} has headroom BEFORE paying a full training run.

Probes (judged against the DEV combo-disjoint wall, base N5 oracle ~0.65):
  andmask : cosine + sign-agreement of the cls-gradient between low-NOC and high-NOC environments.
            Conflict (low cosine / agreement≈50%) => spurious env-inconsistent directions exist for
            AND-mask to remove => GO. High agreement => AND-mask ≈ no-op => NO-GO.
  mask    : oracle drop under INFERENCE peak-dropout, SEEN (train-fit) vs NOVEL (dev). If SEEN N5 is
            template-propped (collapses under dropout) far more than NOVEL => the model leans on a
            memorized full-peak template => training-time masking has a crutch to break => GO.
  meta    : cross-combo std of a donor's predicted prob, NOVEL (dev) vs SEEN (train-fit). Donor conf
            swinging with co-contributor context on novel combos => combo-entangled rep => meta
            (context-invariance) has headroom => GO.
  slot    : on the P6 checkpoint, %∅ slots + rate the TRUE donor is the runner-up UNDER ∅ in collapsed
            slots. True donors suppressed just under ∅ => DETR eos_coef will surface them => GO.

Usage: python feasibility_inc6.py [inc5_res_rand1_seed42] [data_insilico_w] [inc4_p6_slot_seed42]
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.autograd as autograd

try:
    sys.stdout.reconfigure(encoding="utf-8")          # Windows console defaults to cp1252
except Exception:
    pass
RUN   = sys.argv[1] if len(sys.argv) > 1 else "inc5_res_rand1_seed42"
DATA  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_insilico_w")
P6RUN = sys.argv[3] if len(sys.argv) > 3 else "inc4_p6_slot_seed42"
ROOT  = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.set_transformer import SetTransformerMixture
from train_set_transformer import AsymmetricLoss

cfg = json.load(open(ROOT / "results" / RUN / "metrics.json"))["config"]
n_tok = cfg.get("n_token_feats", 8)
tp = f"tokens{n_tok}" if n_tok > 3 else "tokens"


def load_split(s):
    return (np.load(DATA / f"{tp}_{s}.npy").astype(np.float32), np.load(DATA / f"mask_{s}.npy"),
            np.load(DATA / f"y_{s}_set.npy"), np.load(DATA / f"noc_{s}.npy").astype(int))


def dev_mask_seed0(y, noc, combo_frac=0.15, noc1_frac=0.06, seed=0):
    rng = np.random.default_rng(seed); noc = np.clip(noc.astype(int), 1, 5); N = len(noc); m = np.zeros(N, bool)
    for k in [2, 3, 4, 5]:
        idx = np.where(noc == k)[0]; combos = {}
        for i in idx:
            combos.setdefault(tuple(np.where(y[i] == 1)[0].tolist()), []).append(i)
        uniq = list(combos); rng.shuffle(uniq)
        for c in uniq[:max(1, int(round(len(uniq) * combo_frac)))]:
            m[combos[c]] = True
    idx1 = np.where(noc == 1)[0]
    m[rng.choice(idx1, size=int(round(len(idx1) * noc1_frac)), replace=False)] = True
    return m


full = load_split("train")
dm = dev_mask_seed0(full[2], full[3])
DEV = tuple(a[dm] for a in full); TRAINFIT = tuple(a[~dm] for a in full)
print(f"feasibility probes | run={RUN} data={DATA} device={DEVICE}")
print(f"DEV (novel combos) {dm.sum()}  |  TRAIN-fit (seen) {(~dm).sum()}\n")

model = SetTransformerMixture(
    n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16), d_model=cfg.get("d_model", 128),
    n_heads=cfg.get("n_heads", 4), n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
    n_classes=cfg.get("n_classes", 45), n_noc=cfg.get("n_noc", 6), dropout=cfg.get("dropout", 0.1),
    cls_decoder=cfg.get("cls_decoder", "per_donor"), decoder_source=cfg.get("decoder_source", "encoded"),
    n_token_feats=n_tok, encoder=cfg.get("encoder", "isab++"), dec_layers=cfg.get("dec_layers", 2),
    num_embed=cfg.get("num_embed", "periodic"), n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
    periodic_sigma=cfg.get("periodic_sigma", 0.3), aux_heads=cfg.get("aux_heads", False),
    sparse_attn=cfg.get("sparse_attn", False),
).to(DEVICE)
sd = torch.load(ROOT / "results" / RUN / "best_model.pt", map_location=DEVICE, weights_only=False)
sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
model.load_state_dict(sd, strict=False)
model.eval()
verdicts = {}


@torch.no_grad()
def probs(tok, msk, bs=256):
    P = []
    for i in range(0, len(tok), bs):
        o = model(torch.from_numpy(tok[i:i+bs]).to(DEVICE), torch.from_numpy(msk[i:i+bs]).to(DEVICE))
        P.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
    return np.concatenate(P)


def oracle_k(P, y, noc, k):
    sel = np.clip(noc, 1, 5) == k
    if not sel.any():
        return float("nan")
    yy, PP = y[sel], P[sel]; em = []
    for j in range(len(PP)):
        top = np.argsort(PP[j])[::-1][:k]; pr = np.zeros(P.shape[1], int); pr[top] = 1
        em.append((pr == yy[j]).all())
    return float(np.mean(em))


# ── PROBE 1: AND-mask gradient-conflict ──────────────────────────────────────────────────
print("== PROBE andmask (cls-grad conflict low-NOC vs high-NOC) ==")
asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
params = [p for p in model.parameters() if p.requires_grad]

def env_grad(env_mask, n_take=512, bs=128):
    # accumulate the cls-loss gradient over an environment in small batches (4GB-safe)
    idx = np.where(env_mask)[0]
    idx = np.random.default_rng(0).permutation(idx)[:n_take]
    acc = [torch.zeros_like(p) for p in params]; nb = 0
    for i in range(0, len(idx), bs):
        bi = idx[i:i+bs]
        tk = torch.from_numpy(TRAINFIT[0][bi]).to(DEVICE); mk = torch.from_numpy(TRAINFIT[1][bi]).to(DEVICE)
        yb = torch.from_numpy(TRAINFIT[2][bi]).float().to(DEVICE)
        g = autograd.grad(asl(model(tk, mk)["logits_cls"], yb), params,
                          retain_graph=False, allow_unused=True)
        for k in range(len(params)):
            if g[k] is not None:
                acc[k] += g[k].detach()
        nb += 1
        del g; torch.cuda.empty_cache()
    return torch.cat([(c / nb).reshape(-1) for c in acc])

ncl = np.clip(TRAINFIT[3], 1, 5)
a = env_grad(ncl <= 3); b = env_grad(ncl >= 4)
cos = (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()
nz = (a.abs() > 1e-9) & (b.abs() > 1e-9)
agree = ((a.sign() == b.sign())[nz].float().mean()).item()
go = (cos < 0.3) or (agree < 0.6)
verdicts["andmask"] = ("GO" if go else "no-go",
                       f"cosine={cos:.3f} sign-agree={agree:.1%} (GO if cos<0.3 or agree<60%)")
print(f"  low↔high NOC grad: cosine={cos:.3f}  sign-agreement={agree:.1%}  -> {verdicts['andmask'][0]}\n")


# ── PROBE 2: peak-dropout sensitivity, SEEN vs NOVEL ─────────────────────────────────────
print("== PROBE mask (inference peak-dropout sensitivity, SEEN vs NOVEL) ==")
def oracle_drop(arrs, k, p):
    tok, msk, y, noc = arrs
    sel = np.clip(noc, 1, 5) == k
    tok, msk, y = tok[sel], msk[sel].copy(), y[sel]
    if p > 0:
        rng = np.random.default_rng(1); mb = msk.astype(bool)
        drop = (rng.random(mb.shape) < p) & mb
        kept = mb & ~drop; enough = kept.sum(1, keepdims=True) >= 8
        msk = np.where(enough, kept, mb)
    P = probs(tok, msk); em = []
    for j in range(len(P)):
        top = np.argsort(P[j])[::-1][:k]; pr = np.zeros(P.shape[1], int); pr[top] = 1
        em.append((pr == y[j]).all())
    return float(np.mean(em)) if em else float("nan")

go_any = False
for k in (4, 5):
    s0, s2 = oracle_drop(TRAINFIT, k, 0.0), oracle_drop(TRAINFIT, k, 0.2)
    n0, n2 = oracle_drop(DEV, k, 0.0), oracle_drop(DEV, k, 0.2)
    ds, dn = s0 - s2, n0 - n2
    flag = ds - dn > 0.10
    go_any = go_any or flag
    print(f"  N{k}: SEEN {s0:.3f}->{s2:.3f} (Δ{ds:+.3f})  NOVEL {n0:.3f}->{n2:.3f} (Δ{dn:+.3f})  "
          f"seen-extra-drop={ds-dn:+.3f}{'  *' if flag else ''}")
verdicts["mask"] = ("GO" if go_any else "no-go", "GO if SEEN N4/N5 drops >0.10 more than NOVEL (template-propped)")
print(f"  -> {verdicts['mask'][0]}\n")


# ── PROBE 3: meta combo-entanglement (cross-combo donor-prob std) ─────────────────────────
print("== PROBE meta (cross-combo donor-prob std, NOVEL vs SEEN) ==")
def donor_std(arrs, kmin=2):
    tok, msk, y, noc = arrs
    hi = np.clip(noc, 1, 5) >= kmin
    tok, msk, y = tok[hi], msk[hi], y[hi]
    P = probs(tok, msk); yb = y > 0.5; stds = []
    for d in range(45):
        rows = np.where(yb[:, d])[0]
        if len(rows) >= 8:
            stds.append(P[rows, d].std())
    return float(np.mean(stds)) if stds else float("nan"), len(stds)
seen_std, ns = donor_std(TRAINFIT); novel_std, nn_ = donor_std(DEV)
ratio = novel_std / (seen_std + 1e-9)
go = ratio > 1.5
verdicts["meta"] = ("GO" if go else "no-go", f"novel/seen std ratio={ratio:.2f} (GO if >1.5)")
print(f"  donor-prob std  SEEN={seen_std:.4f} (n={ns})  NOVEL={novel_std:.4f} (n={nn_})  ratio={ratio:.2f}  "
      f"-> {verdicts['meta'][0]}\n")


# ── PROBE 4: slot ∅-collapse headroom (needs the P6 checkpoint) ───────────────────────────
print("== PROBE slot (P6 ∅-collapse headroom) ==")
p6_ckpt = ROOT / "results" / P6RUN / "best_model.pt"
if p6_ckpt.exists():
    from train_p6_slot import P6Model
    p6 = P6Model().to(DEVICE)
    p6.load_state_dict(torch.load(p6_ckpt, map_location=DEVICE, weights_only=False), strict=False)
    p6.eval()
    def slot_probe(arrs, k):
        tok, msk, y, noc = arrs
        sel = np.clip(noc, 1, 5) == k
        tok, msk, y = tok[sel], msk[sel], y[sel]
        nullf, run_true = [], []
        for i in range(0, len(tok), 256):
            with torch.no_grad():
                ls = p6(torch.from_numpy(tok[i:i+256]).to(DEVICE),
                        torch.from_numpy(msk[i:i+256]).to(DEVICE))["logits_slot"].cpu().numpy()
            for j in range(len(ls)):
                am = ls[j].argmax(1); nullf.append((am == 45).mean())
                truth = set(np.where(y[i+j] > 0.5)[0].tolist())
                for s in range(ls.shape[1]):
                    if am[s] == 45:
                        run_true.append(int(ls[j, s, :45].argmax() in truth))
        return float(np.mean(nullf)), (float(np.mean(run_true)) if run_true else float("nan"))
    go_any = False
    for k in (4, 5):
        nf, rt = slot_probe(DEV, k)
        flag = (rt == rt) and rt > 0.30
        go_any = go_any or flag
        print(f"  N{k}: ∅-slots={nf:.1%}  true-donor-is-under-∅-runnerup={rt:.1%}{'  *' if flag else ''}")
    verdicts["slot"] = ("GO" if go_any else "no-go", "GO if >30% of collapsed slots hide a true donor just under ∅")
else:
    verdicts["slot"] = ("skip", f"no P6 checkpoint at {p6_ckpt}")
    print(f"  {verdicts['slot'][1]}")
print()


# ── verdict table ────────────────────────────────────────────────────────────────────────
print("================ FEASIBILITY VERDICTS (no-train) ================")
for lever in ("andmask", "mask", "meta", "slot"):
    v, why = verdicts[lever]
    print(f"  {lever:<8} {v:<5}  {why}")
print("\nReminder: a probe is a DIRECTION-FINDER, not proof. A 'GO' = worth a full single-seed run on")
print("the DEV N5 oracle; confirm any N4/N5 effect with 3-seed CIs (F14/F16 trap) before a finding.")

"""
feasibility_inc6b.py — NO-TRAIN feasibility probes for the BLIND-SPOT levers (the still-untried ⚪
branches of the intervention map): 4a SAM, 4b model-soup/SWA, 3d Spectral Decoupling, 5d test-time
adaptation. Same discipline as feasibility_inc6.py: read existing local checkpoints, print GO/NO-GO.

Probes (judged vs the DEV combo-disjoint N5 wall, base oracle ~0.65):
  sam   : loss-SHARPNESS (rise under an rho-norm gradient-ascent weight perturbation) on N5, SEEN
          (train-fit) vs NOVEL (dev). Novel much sharper => flat-minima (SAM) has headroom => GO.
  soup  : average the rand1 + randk weights (both seed-42 init, same arch => legitimately soup-able);
          DEV N5 oracle of the soup >= best individual (with N1/2/3 guard held) => GO.
  spec  : Spectral-Decoupling / gradient-starvation signature = logit magnitude + prediction peakiness,
          SEEN vs NOVEL. Seen far larger/peakier => model over-relies on few dominant features => GO.
  tta   : TENT-style test-time adapt (entropy-min on the test batch, NORM affines only, few steps);
          DEV N5 oracle rises => per-mixture adaptation has headroom => GO.

Usage: python feasibility_inc6b.py [inc5_res_rand1_seed42] [data_insilico_w] [inc5_res_randk_seed42]
"""
import os, sys, json, copy
from pathlib import Path
import numpy as np
import torch
import torch.autograd as autograd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
RUN   = sys.argv[1] if len(sys.argv) > 1 else "inc5_res_rand1_seed42"
DATA  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_insilico_w")
RUN2  = sys.argv[3] if len(sys.argv) > 3 else "inc5_res_randk_seed42"   # 2nd ckpt for the soup
ROOT  = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.set_transformer import SetTransformerMixture
from train_set_transformer import AsymmetricLoss

cfg = json.load(open(ROOT / "results" / RUN / "metrics.json"))["config"]
n_tok = cfg.get("n_token_feats", 8)
tp = f"tokens{n_tok}" if n_tok > 3 else "tokens"
asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)


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
print(f"feasibility probes (blind-spot) | run={RUN} data={DATA} device={DEVICE}")
print(f"DEV (novel) {dm.sum()}  |  TRAIN-fit (seen) {(~dm).sum()}\n")


def build():
    return SetTransformerMixture(
        n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16), d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4), n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
        n_classes=cfg.get("n_classes", 45), n_noc=cfg.get("n_noc", 6), dropout=cfg.get("dropout", 0.1),
        cls_decoder=cfg.get("cls_decoder", "per_donor"), decoder_source=cfg.get("decoder_source", "encoded"),
        n_token_feats=n_tok, encoder=cfg.get("encoder", "isab++"), dec_layers=cfg.get("dec_layers", 2),
        num_embed=cfg.get("num_embed", "periodic"), n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
        periodic_sigma=cfg.get("periodic_sigma", 0.3), aux_heads=cfg.get("aux_heads", False),
        sparse_attn=cfg.get("sparse_attn", False)).to(DEVICE)


def load_ckpt(m, run):
    sd = torch.load(ROOT / "results" / run / "best_model.pt", map_location=DEVICE, weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    m.load_state_dict(sd, strict=False); return sd


model = build(); load_ckpt(model, RUN); model.eval()
params = [p for p in model.parameters() if p.requires_grad]
verdicts = {}


def subset(arrs, k):
    sel = np.clip(arrs[3], 1, 5) == k
    return arrs[0][sel], arrs[1][sel], arrs[2][sel]


@torch.no_grad()
def mean_loss(arrs, k, bs=128):
    tok, msk, y = subset(arrs, k); tot = 0.0; n = 0
    for i in range(0, len(tok), bs):
        o = model(torch.from_numpy(tok[i:i+bs]).to(DEVICE), torch.from_numpy(msk[i:i+bs]).to(DEVICE))
        yb = torch.from_numpy(y[i:i+bs]).float().to(DEVICE)
        b = len(tok[i:i+bs]); tot += asl(o["logits_cls"], yb).item() * b; n += b
    return tot / max(n, 1)


def grad_dir(arrs, k, n_take=512, bs=128):
    tok, msk, y = subset(arrs, k)
    idx = np.random.default_rng(0).permutation(len(tok))[:n_take]
    acc = [torch.zeros_like(p) for p in params]; nb = 0
    for i in range(0, len(idx), bs):
        bi = idx[i:i+bs]
        tk = torch.from_numpy(tok[bi]).to(DEVICE); mk = torch.from_numpy(msk[bi]).to(DEVICE)
        yb = torch.from_numpy(y[bi]).float().to(DEVICE)
        g = autograd.grad(asl(model(tk, mk)["logits_cls"], yb), params, retain_graph=False, allow_unused=True)
        for j in range(len(params)):
            if g[j] is not None:
                acc[j] += g[j].detach()
        nb += 1; del g; torch.cuda.empty_cache()
    return [a / max(nb, 1) for a in acc]


@torch.no_grad()
def oracle_k(arrs, k, bs=256, mdl=None):
    mdl = mdl or model
    tok, msk, y = subset(arrs, k); em = []
    for i in range(0, len(tok), bs):
        P = torch.sigmoid(mdl(torch.from_numpy(tok[i:i+bs]).to(DEVICE),
                              torch.from_numpy(msk[i:i+bs]).to(DEVICE))["logits_cls"]).cpu().numpy()
        for j in range(len(P)):
            top = np.argsort(P[j])[::-1][:k]; pr = np.zeros(P.shape[1], int); pr[top] = 1
            em.append((pr == y[i+j]).all())
    return float(np.mean(em)) if em else float("nan")


# ── 4a SAM sharpness ─────────────────────────────────────────────────────────────────────
print("== PROBE sam (loss-sharpness under rho-perturbation, SEEN vs NOVEL N5) ==")
def sharpness(arrs, k, rho=0.05):
    g = grad_dir(arrs, k)
    gnorm = torch.sqrt(sum((gi ** 2).sum() for gi in g)) + 1e-12
    base = mean_loss(arrs, k)
    saved = [p.data.clone() for p in params]
    with torch.no_grad():
        for p, gi in zip(params, g):
            p.data.add_(gi, alpha=(rho / gnorm).item())
    pert = mean_loss(arrs, k)
    with torch.no_grad():
        for p, s in zip(params, saved):
            p.data.copy_(s)
    return base, pert - base
for k in (5,):
    sb, ss = sharpness(TRAINFIT, k); nb, nsh = sharpness(DEV, k)
    ratio = nsh / (ss + 1e-9)
    go = ratio > 1.5
    verdicts["sam"] = ("GO" if go else "no-go", f"novel/seen sharpness ratio={ratio:.2f} (GO if >1.5)")
    print(f"  N{k}: SEEN loss {sb:.3f} sharp {ss:+.3f} | NOVEL loss {nb:.3f} sharp {nsh:+.3f} | ratio {ratio:.2f} -> {verdicts['sam'][0]}\n")


# ── 4b model-soup / SWA ──────────────────────────────────────────────────────────────────
print("== PROBE soup (average rand1+randk weights, DEV N5 oracle vs best individual) ==")
sd1 = {k: v.clone() for k, v in model.state_dict().items()}
m2 = build(); sd2 = load_ckpt(m2, RUN2)
base_n5_1 = oracle_k(DEV, 5)
base_n5_2 = oracle_k(DEV, 5, mdl=m2)
soup = {}
for k in sd1:
    v1, v2 = sd1[k], sd2[k] if k in sd2 else sd1[k]
    soup[k] = ((v1.float() + v2.float()) / 2).to(v1.dtype) if torch.is_floating_point(v1) else v1
model.load_state_dict(soup, strict=False)
soup_n5 = oracle_k(DEV, 5); g1, g2, g3 = oracle_k(DEV, 1), oracle_k(DEV, 2), oracle_k(DEV, 3)
model.load_state_dict(sd1, strict=False)  # restore rand1
del m2; torch.cuda.empty_cache()
best_ind = max(base_n5_1, base_n5_2)
go = soup_n5 >= best_ind - 1e-9 and min(g1, g2, g3) >= 0.95
verdicts["soup"] = ("GO" if go else "no-go",
                    f"soup N5={soup_n5:.3f} vs best-indiv {best_ind:.3f} (guard N1/2/3={g1:.2f}/{g2:.2f}/{g3:.2f})")
print(f"  rand1 N5={base_n5_1:.3f} randk N5={base_n5_2:.3f} | SOUP N5={soup_n5:.3f}  guard {g1:.2f}/{g2:.2f}/{g3:.2f} -> {verdicts['soup'][0]}\n")


# ── 3d Spectral Decoupling signature ─────────────────────────────────────────────────────
print("== PROBE spec (logit magnitude + peakiness, SEEN vs NOVEL N5) ==")
@torch.no_grad()
def logit_stats(arrs, k, bs=256):
    tok, msk, _ = subset(arrs, k); norms = []; peaks = []
    for i in range(0, len(tok), bs):
        L = model(torch.from_numpy(tok[i:i+bs]).to(DEVICE),
                  torch.from_numpy(msk[i:i+bs]).to(DEVICE))["logits_cls"]
        norms.append(L.norm(dim=1).cpu().numpy())
        peaks.append(torch.sigmoid(L).max(dim=1).values.cpu().numpy())
    return float(np.concatenate(norms).mean()), float(np.concatenate(peaks).mean())
sn, sp = logit_stats(TRAINFIT, 5); nn_, np_ = logit_stats(DEV, 5)
rn = sn / (nn_ + 1e-9)
go = rn > 1.3
verdicts["spec"] = ("GO" if go else "no-go", f"seen/novel logit-norm ratio={rn:.2f} (GO if >1.3)")
print(f"  N5 logit-L2  SEEN={sn:.2f} (peak {sp:.2f})  NOVEL={nn_:.2f} (peak {np_:.2f})  norm-ratio={rn:.2f} -> {verdicts['spec'][0]}\n")


# ── 5d TENT-style test-time adaptation (norm-affine entropy-min) ──────────────────────────
print("== PROBE tta (TENT entropy-min on DEV N5, norm-affine params, DEV N5 oracle before/after) ==")
def tta_oracle(arrs, k, steps=10, lr=1e-3, bs=256):
    sd = {kk: v.clone() for kk, v in model.state_dict().items()}
    norm_params = [p for n, p in model.named_parameters() if "norm" in n.lower() and p.requires_grad]
    opt = torch.optim.SGD(norm_params, lr=lr)
    tok, msk, y = subset(arrs, k)
    model.train()
    for _ in range(steps):
        for i in range(0, len(tok), bs):
            tk = torch.from_numpy(tok[i:i+bs]).to(DEVICE); mk = torch.from_numpy(msk[i:i+bs]).to(DEVICE)
            L = model(tk, mk)["logits_cls"]; p = torch.sigmoid(L)
            ent = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean()
            opt.zero_grad(); ent.backward(); opt.step()
    model.eval()
    after = oracle_k(arrs, k)
    model.load_state_dict(sd, strict=False)
    return after, len(norm_params)
before = oracle_k(DEV, 5)
after, n_np = tta_oracle(DEV, 5)
go = after - before > 0.01
verdicts["tta"] = ("GO" if go else "no-go", f"DEV N5 {before:.3f}->{after:.3f} adapting {n_np} norm-params (GO if +>.01)")
print(f"  DEV N5 oracle  before={before:.3f}  after-TTA={after:.3f}  ({n_np} norm-affine params) -> {verdicts['tta'][0]}\n")


# ── verdicts ─────────────────────────────────────────────────────────────────────────────
print("================ BLIND-SPOT FEASIBILITY VERDICTS (no-train) ================")
for lever in ("sam", "soup", "spec", "tta"):
    v, why = verdicts[lever]
    print(f"  {lever:<6} {v:<5}  {why}")
print("\nNot probeable no-train (need (re)build/train): 3e masked-pretrain, 2f disentangled-latents, 4c FOMAML.")

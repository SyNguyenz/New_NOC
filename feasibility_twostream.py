"""
feasibility_twostream.py — NO-TRAIN probe for the CONTENT/CONTEXT separation direction (Russin 2019).

Hypothesis: the combo-overfit comes from the encoder's GLOBAL context-mixing (ISAB H) injecting
co-contributor (combo) information into every peak, which the decoder then memorizes. Russin's fix =
route the ID decision through a CONTEXT-INDEPENDENT content stream.

Decisive test (linear probe on FROZEN features of the best model; the big net is never trained):
fit a per-peak donor-attribution classifier on
  - x0  = PRE-ISAB projected tokens  (content: each peak from its own locus/allele/height, no cross-peak mix)
  - H   = POST-ISAB encoded tokens    (context: globally combo-mixed)
and compare the TRAIN-fit (seen combos) -> DEV (novel combos) generalization GAP at N4/N5.
  gap(H) >> gap(x0)  => global mixing is the combo-overfit source; content stream protects => two-stream GO.
  gap(H) ~= gap(x0)  => mixing is not the culprit; two-stream won't help -> look elsewhere.

Usage: python feasibility_twostream.py [inc6_maskp_seed42]
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
ROOT = Path(__file__).resolve().parent
DATA = Path("data_insilico_w")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN = sys.argv[1] if len(sys.argv) > 1 else "inc6_maskp_seed42"
from models.set_transformer import SetTransformerMixture


def dev_mask_seed0(y, noc, cf=0.15, nf=0.06, seed=0):
    rng = np.random.default_rng(seed); noc = np.clip(noc.astype(int), 1, 5); N = len(noc); m = np.zeros(N, bool)
    for k in [2, 3, 4, 5]:
        idx = np.where(noc == k)[0]; combos = {}
        for i in idx:
            combos.setdefault(tuple(np.where(y[i] == 1)[0].tolist()), []).append(i)
        uniq = list(combos); rng.shuffle(uniq)
        for c in uniq[:max(1, int(round(len(uniq) * cf)))]:
            m[combos[c]] = True
    idx1 = np.where(noc == 1)[0]
    m[rng.choice(idx1, size=int(round(len(idx1) * nf)), replace=False)] = True
    return m


cfg = json.load(open(ROOT / "results" / RUN / "metrics.json"))["config"]
n_tok = cfg.get("n_token_feats", 8)
y = np.load(DATA / "y_train_set.npy").astype(int); noc = np.clip(np.load(DATA / "noc_train.npy").astype(int), 1, 5)
tok = np.load(DATA / f"tokens{n_tok}_train.npy").astype(np.float32); msk = np.load(DATA / "mask_train.npy").astype(bool)
attr = np.load(DATA / "attr_train.npy").astype(np.int64)
dm = dev_mask_seed0(y, noc)

model = SetTransformerMixture(
    n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32, n_classes=45, n_noc=6,
    dropout=0.0, cls_decoder=cfg.get("cls_decoder", "per_donor"), decoder_source="encoded",
    n_token_feats=n_tok, encoder=cfg.get("encoder", "isab++"), dec_layers=cfg.get("dec_layers", 2),
    num_embed=cfg.get("num_embed", "periodic"), n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
    periodic_sigma=cfg.get("periodic_sigma", 0.3), aux_heads=cfg.get("aux_heads", False),
    sparse_attn=cfg.get("sparse_attn", False)).to(DEVICE)
sd = torch.load(ROOT / "results" / RUN / "best_model.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(sd, strict=False); model.eval()
print(f"frozen features from {RUN}\n")


@torch.no_grad()
def extract(idx):
    """-> X0, H, lab  (per labeled valid peak)"""
    X0, HH, LB = [], [], []
    for b in range(0, len(idx), 128):
        bi = idx[b:b+128]
        t = torch.from_numpy(tok[bi]).to(DEVICE); m = torch.from_numpy(msk[bi]).to(DEVICE)
        x0, H, _ = model._encode_set(t, m)
        a = attr[bi]
        valid = msk[bi] & (a >= 0)                       # labeled real peaks
        X0.append(x0.cpu().numpy()[valid]); HH.append(H.cpu().numpy()[valid]); LB.append(a[valid])
    return np.concatenate(X0), np.concatenate(HH), np.concatenate(LB)


def linear_probe(Xtr, ytr, Xev, yev, epochs=40):
    clf = nn.Linear(Xtr.shape[1], 45).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr).to(DEVICE); yt = torch.from_numpy(ytr).to(DEVICE)
    for _ in range(epochs):
        for b in range(0, len(Xt), 8192):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(clf(Xt[b:b+8192]), yt[b:b+8192])
            loss.backward(); opt.step()
    @torch.no_grad()
    def acc(X, yy):
        Xe = torch.from_numpy(X).to(DEVICE); pred = []
        for b in range(0, len(Xe), 16384):
            pred.append(clf(Xe[b:b+16384]).argmax(1).cpu().numpy())
        return float((np.concatenate(pred) == yy).mean())
    return acc(Xtr, ytr), acc(Xev, yev)


print(f"{'NOC':>4} {'feature':>8} {'train-fit':>10} {'dev(novel)':>11} {'gap':>7}")
for K in (4, 5):
    tr_idx = np.where((~dm) & (noc == K))[0]
    tr_idx = np.random.default_rng(0).permutation(tr_idx)[:3000]
    dv_idx = np.where(dm & (noc == K))[0]
    x0_tr, h_tr, lb_tr = extract(tr_idx)
    x0_dv, h_dv, lb_dv = extract(dv_idx)
    for name, Xtr, Xdv in (("x0(content)", x0_tr, x0_dv), ("H(context)", h_tr, h_dv)):
        a_tr, a_dv = linear_probe(Xtr, lb_tr, Xdv, lb_dv)
        print(f"{K:>4} {name:>11} {a_tr:>10.3f} {a_dv:>11.3f} {a_tr-a_dv:>7.3f}")
    print()
print("gap(H) >> gap(x0) => ISAB global-mixing memorizes combos; content stream generalizes => two-stream GO")

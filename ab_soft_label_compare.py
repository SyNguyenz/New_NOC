"""
ab_soft_label_compare.py
Compare attr head training strategies — same small model, same cls/noc/phi losses:

  Model A: hard CE on parent-donor labels  (current, the "wrong" baseline)
  Model B: soft CE with GT φ × Cn labels   (EuroForMix attribution formula)

Then for each model, evaluate phi quality from multiple sources:
  phi_head    — direct neural phi head
  EM-uniform  — symbolic EM (no neural attr), baseline oracle
  EM-attr     — EM seeded with model's raw attr logits as compatibility
  EM-attr+M   — same but attr logits are inference-masked (private-hard / shared-soft)

Key fix vs ab_phi_attr_train.py:
  - Old script trained with genotype mask at training (wrong: NaN gradient for stutter peaks)
  - New base (A): no mask at training, hard CE on parent-donor label
  - New soft (B): no mask at training, soft CE with GT phi × Cn (EuroForMix-consistent)
  - Copy number (Cn = 0/1/2) built from genotype accumulation (not binary)
"""
import os, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DA   = "data_insilico_w"
G    = "data/donor_geno.npy"
DEV  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_N = int(os.environ.get("TRAIN_N", "12000"))
EPOCHS  = int(os.environ.get("EPOCHS",  "30"))
SEED    = int(os.environ.get("SEED",    "42"))
C = 45; NITER = 5

# ── Genotype lookups ─────────────────────────────────────────────────────────
def ab(a):  return int(round(float(a) * 10))
def kk(l, a): return (int(round(float(l))), ab(a))

g  = np.load(G)
gm = np.load(G.replace(".npy", "_mask.npy")).astype(bool)

# CN[locus, allele_bin, donor] = copy number (0 het→1, homo→2)
CN = np.zeros((24, 1024, C), np.float32)
carr_raw = {}  # (locus,bin) → list with possible duplicates
for c in range(C):
    for j in range(g.shape[1]):
        if gm[c, j]:
            l_ = int(round(float(g[c, j, 0]))); b_ = ab(g[c, j, 1])
            if 0 <= l_ < 24 and 0 <= b_ < 1024:
                CN[l_, b_, c] += 1.0
                carr_raw.setdefault(kk(g[c,j,0], g[c,j,1]), []).append(c)
# Deduplicate carriers (homozygous donors appeared twice)
carr = {k: list(set(v)) for k, v in carr_raw.items()}
CN_t = torch.from_numpy(CN).to(DEV)

print(f"Genotype loaded: {C} donors, mean carriers/allele = "
      f"{np.mean([len(v) for v in carr.values()]):.2f}, "
      f"CN=2 entries = {int((CN == 2).sum())}")

# ── Data loading ─────────────────────────────────────────────────────────────
def load(sp):
    return (
        np.load(f"{DA}/tokens8_{sp}.npy").astype(np.float32),
        np.load(f"{DA}/mask_{sp}.npy").astype(bool),
        np.load(f"{DA}/y_{sp}_set.npy").astype(np.float32),
        np.clip(np.load(f"{DA}/noc_{sp}.npy"), 1, 5),
        np.load(f"{DA}/phi_{sp}.npy").astype(np.float32),   # GT mixture proportions (N, C)
        np.load(f"{DA}/attr_{sp}.npy").astype(np.int64),    # parent-donor labels (N, max_peaks)
    )

tk, mk, y, noc, phi, attr = load("train")

def devmask(y, noc, frac=0.15, n1=0.06, seed=0):
    rng = np.random.default_rng(seed); m = np.zeros(len(noc), bool)
    for k in [2, 3, 4, 5]:
        idx = np.where(noc == k)[0]; cm = {}
        for i in idx: cm.setdefault(tuple(np.where(y[i] == 1)[0]), []).append(i)
        u = list(cm); rng.shuffle(u)
        for cc in u[:max(1, int(round(len(u)*frac)))]: m[cm[cc]] = True
    i1 = np.where(noc == 1)[0]
    m[rng.choice(i1, size=int(round(len(i1)*n1)), replace=False)] = True
    return m

dm  = devmask(y, noc)
tri = np.where(~dm)[0]
rng = np.random.default_rng(0)
tri = rng.choice(tri, size=min(TRAIN_N, len(tri)), replace=False)

def sub(ix): return tuple(a[ix] for a in (tk, mk, y, noc, phi, attr))
TRN = sub(tri); VAL = load("val"); TST = load("test")

_v    = TRN[0][TRN[1]][:, 2:8]
FMEAN = torch.tensor(_v.mean(0)).to(DEV)
FSTD  = torch.tensor(_v.std(0).clip(1e-3)).to(DEV)

print(f"train={len(TRN[0])}  testN5={int((TST[3]==5).sum())}  valNOC={np.bincount(VAL[3])[1:]}")

# ── Soft label builder ────────────────────────────────────────────────────────
def build_soft_labels(xb: torch.Tensor, mb: torch.Tensor, phib: torch.Tensor) -> torch.Tensor:
    """
    EuroForMix-consistent per-peak soft attribution labels.

    soft_y[b,n,c] = phi[b,c] * Cn[c, locus, allele_bin]
                    / Σ_c'(phi[b,c'] * Cn[c', locus, allele_bin])

    Special cases:
      Infeasible (Σ=0): soft_y[b,n,C] = 1.0  (background)
      Padding (!mb):    soft_y[b,n,*] = 0.0  (ignored in loss)

    Stutter peaks: if stutter allele is in NO carrier genotype → infeasible → background.
    If stutter allele matches some carrier's real allele → attributed to that carrier(s)
    proportional to phi × Cn. This is more correct than "parent donor" hard labels.
    """
    li = xb[..., 0].long().clamp(0, 23)                      # (B, N)
    bi = (xb[..., 1] * 10).round().long().clamp(0, 1023)     # (B, N)
    cn = CN_t[li, bi]                                         # (B, N, C) copy numbers
    # Weight each donor's Cn by their mixture proportion
    phi_cn = phib.unsqueeze(1) * cn                          # (B, N, C)  broadcast phi over peaks
    total  = phi_cn.sum(-1, keepdim=True)                    # (B, N, 1)
    feasible = (total.squeeze(-1) > 1e-9) & mb               # (B, N)
    normalized = phi_cn / total.clamp(min=1e-9)               # (B, N, C)
    soft_y = torch.zeros(xb.size(0), xb.size(1), C + 1, device=DEV)
    # Feasible: distribute proportionally
    soft_y[..., :C] = normalized * feasible.unsqueeze(-1).float()
    # Infeasible (valid but no carrier): all to background
    soft_y[..., C]  = (~feasible & mb).float()
    # Padding: stays zero
    return soft_y

# ── Model (same architecture for A and B) ────────────────────────────────────
class MAB(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.att = nn.MultiheadAttention(d, h, batch_first=True)
        s.l1 = nn.LayerNorm(d); s.l2 = nn.LayerNorm(d)
        s.ff  = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
    def forward(s, q, k, kpm=None):
        a, _ = s.att(q, k, k, key_padding_mask=kpm, need_weights=False)
        x = s.l1(q + a); return s.l2(x + s.ff(x))

class Per(nn.Module):
    def __init__(s, d, nf=8, sg=0.3):
        super().__init__()
        s.c = nn.Parameter(torch.randn(nf) * sg); s.l = nn.Linear(2*nf, d)
    def forward(s, v):
        z = 2 * math.pi * s.c * v.unsqueeze(-1)
        return s.l(torch.cat([torch.sin(z), torch.cos(z)], -1))

class SmallModel(nn.Module):
    def __init__(s, d=64, h=4, nind=16):
        super().__init__()
        s.le  = nn.Embedding(26, d); s.pe = Per(d); s.cl = nn.Linear(6, d)
        s.I   = nn.Parameter(torch.randn(nind, d) * 0.5)
        s.mh  = MAB(d, h); s.mx = MAB(d, h)
        s.Q   = nn.Parameter(torch.empty(C, d)); nn.init.xavier_uniform_(s.Q)
        s.dec = MAB(d, h); s.dec2 = MAB(d, h)
        s.sw  = nn.Parameter(torch.empty(C, d)); nn.init.xavier_uniform_(s.sw)
        s.sb  = nn.Parameter(torch.full((C,), -2.0))
        s.card = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 5))
        s.attr = nn.Linear(d, C + 1)  # per-peak attribution logit
        s.phi  = nn.Linear(d, C)      # phi head (aux, evaluated separately)
    def forward(s, x, m):
        kpm = ~m
        e   = (s.le(x[..., 0].long().clamp(0, 25))
               + s.pe(x[..., 1])
               + s.cl((x[..., 2:8] - FMEAN) / FSTD))
        H   = s.mx(e, s.mh(s.I.unsqueeze(0).expand(x.size(0), -1, -1), e, kpm))
        pooled = (H * m.float().unsqueeze(-1)).sum(1) / m.float().sum(1, keepdim=True).clamp(min=1)
        Qb  = s.Q.unsqueeze(0).expand(x.size(0), -1, -1)
        dec = s.dec2(s.dec(Qb, H, kpm), H, kpm)
        cls = torch.einsum("bcd,cd->bc", dec, s.sw) + s.sb    # (B, C)
        return cls, s.card(pooled), s.attr(H), F.softplus(s.phi(pooled))

# ── Training ─────────────────────────────────────────────────────────────────
def train_model(label: str, seed: int = SEED):
    """label = 'hard' | 'soft'"""
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallModel().to(DEV)
    opt   = torch.optim.Adam(model.parameters(), lr=6e-4, weight_decay=1e-4)
    posw  = torch.full((C,), 8.0, device=DEV)
    gen   = torch.Generator().manual_seed(seed)

    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(TRN[0]), generator=gen).numpy()
        ep_loss = 0.0; n_batch = 0
        for b0 in range(0, len(idx), 128):
            bi_  = idx[b0:b0+128]
            xb   = torch.from_numpy(TRN[0][bi_]).to(DEV)
            mb   = torch.from_numpy(TRN[1][bi_]).to(DEV)
            yb   = torch.from_numpy(TRN[2][bi_]).to(DEV)
            nb   = torch.from_numpy((TRN[3][bi_]-1).astype(np.int64)).to(DEV)
            pb   = torch.from_numpy(TRN[4][bi_]).to(DEV)   # GT phi (B, C)
            ab_  = torch.from_numpy(TRN[5][bi_]).to(DEV)   # parent-donor labels

            lo, cd, al, ph = model(xb, mb)

            # Shared losses (cls, noc, phi)
            loss = (F.binary_cross_entropy_with_logits(lo, yb, pos_weight=posw)
                    + 0.3 * F.cross_entropy(cd, nb)
                    + 1.0 * F.mse_loss(ph, pb))

            # Attr loss: hard or soft
            if label == 'soft':
                soft_y    = build_soft_labels(xb, mb, pb)          # (B, N, C+1)
                log_p     = F.log_softmax(al, dim=-1)               # (B, N, C+1)
                loss_attr = -(soft_y * log_p).sum(-1)               # (B, N) per-peak CE
                # Mask padding and normalize
                n_valid   = mb.float().sum().clamp(min=1)
                loss_attr = (loss_attr * mb.float()).sum() / n_valid
            else:  # hard
                lbl = ab_.clone(); lbl[lbl < 0] = C; lbl[~mb] = -100
                loss_attr = F.cross_entropy(al.reshape(-1, C+1), lbl.reshape(-1), ignore_index=-100)

            loss = loss + 0.5 * loss_attr
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_batch += 1

        if (ep + 1) % 10 == 0:
            print(f"  [{label}] ep {ep+1}/{EPOCHS}  loss={ep_loss/n_batch:.3f}")

    model.eval()
    return model

# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer(model, S):
    x, m = S[0], S[1]
    Lout, PHout, ALout = [], [], []
    for i in range(0, len(x), 128):
        xb = torch.from_numpy(x[i:i+128]).to(DEV)
        mb = torch.from_numpy(m[i:i+128]).to(DEV)
        lo, _, al, ph = model(xb, mb)
        Lout.append(lo.cpu().numpy())
        PHout.append(ph.cpu().numpy())
        ALout.append(al.cpu().numpy())
    return np.concatenate(Lout), np.concatenate(PHout), np.concatenate(ALout)

def apply_geno_mask_np(AL: np.ndarray, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Inference-time genotype mask on attr logits (numpy).
    Private (1 carrier): hard assign → +1e9 / −1e9.
    Shared  (>1 carrier): soft mask → non-carriers → −1e9.
    Infeasible (0 carrier): all → −1e9.
    """
    AL = AL.copy()
    for i in range(len(x)):
        for k in np.where(m[i])[0]:
            l_ = int(round(float(x[i, k, 0])))
            b_ = int(round(float(x[i, k, 1]) * 10))
            if not (0 <= l_ < 24 and 0 <= b_ < 1024): continue
            cn = CN[l_, b_, :]            # (C,) copy numbers
            n_car = int((cn > 0).sum())
            if n_car == 0:
                AL[i, k, :C] = -1e9
            elif n_car == 1:
                c_idx = int(np.where(cn > 0)[0][0])
                AL[i, k, :C] = -1e9
                AL[i, k, c_idx] = 1e9
            else:
                AL[i, k, :C][cn == 0] = -1e9   # only donor cols, not bg
    return AL

# ── EM phi ────────────────────────────────────────────────────────────────────
def em_phi(tkd, mkd, AL=None, niter=NITER):
    """Compute EM phi. AL=None → uniform compat; else use AL[i,k,c] as log-compat."""
    N = len(tkd); P = np.zeros((N, C))
    for i in range(N):
        pk = [(k, kk(tkd[i,k,0], tkd[i,k,1]), np.expm1(tkd[i,k,2]))
              for k in np.where(mkd[i])[0] if kk(tkd[i,k,0], tkd[i,k,1]) in carr]
        if not pk: continue
        n = len(pk); h = np.array([p[2] for p in pk])
        S = np.full((n, C+1), -1e9)
        for r, (k, it, _) in enumerate(pk):
            for c in carr[it]: S[r, c] = 0.0 if AL is None else float(AL[i, k, c])
            S[r, C] = -2.0 if AL is None else float(AL[i, k, C])
        ph = np.ones(C+1) / (C+1)
        for _ in range(niter):
            z = S + np.log(ph + 1e-9); z -= z.max(1, keepdims=True)
            A = np.exp(z); A /= A.sum(1, keepdims=True)
            w  = (A[:, :C] * h[:, None]).sum(0); bg = (A[:, C] * h).sum()
            ph = np.concatenate([w, [bg]]) / max(w.sum() + bg, 1e-9)
        P[i] = ph[:C]
    return P

# ── Evaluation ────────────────────────────────────────────────────────────────
def auc(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    if not len(p) or not len(q): return float("nan")
    a = np.concatenate([p, q])
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); rk = ((cs - cnt + cs + 1) / 2.0)[inv]
    return (rk[:len(p)].sum() - len(p)*(len(p)+1)/2) / (len(p)*len(q))

def zsc(a): s = a.std(); return (a - a.mean()) / (s if s > 1e-9 else 1.0)

def evaluate(Lt, Lv, PHt, PHv, yt, yv, nt, nv, phit):
    """decoyAUC + within-phi corr + N5 oracle (alpha tuned on val)."""
    mt, dc, wc = [], [], []
    for i in np.where(nt == 5)[0]:
        top5 = set(np.argsort(Lt[i])[::-1][:5])
        miss = [c for c in np.where(yt[i])[0] if c not in top5]
        dec  = [c for c in top5 if not yt[i, c]]
        for c in miss: mt.append(PHt[i, c])
        for c in dec:  dc.append(PHt[i, c])
        cs = np.where(yt[i])[0]
        if len(cs) >= 2:
            a = PHt[i,cs] - PHt[i,cs].mean(); b = phit[i, cs] - phit[i, cs].mean()
            d = a.std() * b.std()
            wc.append((a*b).mean()/d if d > 1e-9 else 0)

    def orc(L, PH, y, no, al):
        sel = np.where(no == 5)[0]; hit = 0
        for i in sel:
            sc = zsc(L[i]) + al * zsc(np.log(PH[i] + 1e-6))
            top = np.argsort(sc)[::-1][:5]; pr = np.zeros(C, int); pr[top] = 1
            hit += int((pr == y[i]).all())
        return hit / max(1, len(sel))

    alphas = [0, 0.25, 0.5, 0.75, 1.0]; ba, bv = 0, -1
    for al in alphas:
        v = orc(Lv, PHv, yv, nv, al)
        if v > bv: bv, ba = v, al
    return auc(mt, dc), np.nanmean(wc), orc(Lt, PHt, yt, nt, 0), orc(Lt, PHt, yt, nt, ba), ba

# ── Main ──────────────────────────────────────────────────────────────────────
print("\n=== Training Model A: hard CE on parent-donor labels ===")
mA = train_model('hard')
print("\n=== Training Model B: soft CE with GT phi x Cn labels ===")
mB = train_model('soft')

print("\n=== Inference ===")
Lt_A, PH_A, AL_A = infer(mA, TST); Lv_A, PHv_A, ALv_A = infer(mA, VAL)
Lt_B, PH_B, AL_B = infer(mB, TST); Lv_B, PHv_B, ALv_B = infer(mB, VAL)

# Inference-time genotype masks
print("Applying genotype mask...")
AL_A_m  = apply_geno_mask_np(AL_A,  TST[0], TST[1])
ALv_A_m = apply_geno_mask_np(ALv_A, VAL[0], VAL[1])
AL_B_m  = apply_geno_mask_np(AL_B,  TST[0], TST[1])
ALv_B_m = apply_geno_mask_np(ALv_B, VAL[0], VAL[1])

# Normalize phi_head outputs
def norm_phi(P): return P / np.maximum(P.sum(1, keepdims=True), 1e-9)

print("Computing EM phi variants...")
PH_uni_t  = em_phi(TST[0], TST[1])
PH_uni_v  = em_phi(VAL[0], VAL[1])
PH_emA_t  = em_phi(TST[0], TST[1], AL_A);   PH_emA_v  = em_phi(VAL[0], VAL[1], ALv_A)
PH_emAm_t = em_phi(TST[0], TST[1], AL_A_m); PH_emAm_v = em_phi(VAL[0], VAL[1], ALv_A_m)
PH_emB_t  = em_phi(TST[0], TST[1], AL_B);   PH_emB_v  = em_phi(VAL[0], VAL[1], ALv_B)
PH_emBm_t = em_phi(TST[0], TST[1], AL_B_m); PH_emBm_v = em_phi(VAL[0], VAL[1], ALv_B_m)

yt = TST[2].astype(bool); nt = TST[3]; phit = TST[4]
yv = VAL[2].astype(bool); nv = VAL[3]

# ── Print results table ───────────────────────────────────────────────────────
sources = [
    # (name, Lt, Lv, PHt, PHv)
    ("EM-uniform [ref]",          Lt_A, Lv_A, PH_uni_t,  PH_uni_v),
    ("A: phi_head",               Lt_A, Lv_A, norm_phi(PH_A),  norm_phi(PHv_A)),
    ("A: EM-attr(hard CE)",       Lt_A, Lv_A, PH_emA_t,  PH_emA_v),
    ("A: EM-attr+mask(hard CE)",  Lt_A, Lv_A, PH_emAm_t, PH_emAm_v),
    ("B: phi_head",               Lt_B, Lv_B, norm_phi(PH_B),  norm_phi(PHv_B)),
    ("B: EM-attr(soft CE)",       Lt_B, Lv_B, PH_emB_t,  PH_emB_v),
    ("B: EM-attr+mask(soft CE)",  Lt_B, Lv_B, PH_emBm_t, PH_emBm_v),
]

hdr = f"{'source':<28} {'decoyAUC':>9} {'within_phi':>10} {'N5 model':>9} {'N5 rerank':>10} {'α*':>4}"
print(f"\n{'='*len(hdr)}\n  Small-scale attr comparison  (train={TRAIN_N} epochs={EPOCHS} seed={SEED})")
print(f"{'='*len(hdr)}\n{hdr}\n{'-'*len(hdr)}")
for nm, Lt, Lv, PHt, PHv in sources:
    da, wc, m0, mr, ba = evaluate(Lt, Lv, PHt, PHv, yt, yv, nt, nv, phit)
    print(f"  {nm:<26} {da:>9.3f} {wc:>8.3f} {m0:>9.3f} {mr:>10.3f} {ba:>4}")
print(f"{'='*len(hdr)}")
print("\nNote: decoyAUC = AUC(missed-true phi > decoy phi, N5 missed cases).")
print("      EM-uniform is model-independent — same phi for all rows using Lt_A for N5 model/rerank.")
print("      'bar to beat' = EM-uniform decoyAUC (should be ~0.968 at this scale).")
print("      Win condition: B:EM-attr(soft CE) decoyAUC > A:EM-attr(hard CE) decoyAUC.")

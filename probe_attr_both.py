"""(a) temperature-scale logits_attr  (b) neural attr vs symbolic soft-split, split by shared/private peak."""
import importlib.util, numpy as np, torch
from pathlib import Path

PROJ = Path("."); HERE = PROJ / "inc22_clean"
CKPT = PROJ / "results/inc22_fixed_aslot_seed42/Donor-Slot_Set_Transformer.pt"
DATA = PROJ / "data_insilico_w"; GENO = PROJ / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLELE_OFF, LUT_W = 30, 1024
K = 45

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def build_owner_lut(gg, gm, n_cls=45):
    gm = gm.bool(); owner = torch.zeros(24, LUT_W, n_cls)
    for c in range(min(n_cls, gg.size(0))):
        for j in range(gg.size(1)):
            if gm[c, j]:
                li = int(gg[c, j, 0]); ab = int(round(float(gg[c, j, 1]) * 10)) + ALLELE_OFF
                if 0 <= li < 24 and 0 <= ab < LUT_W: owner[li, ab, c] = 1.0
    return owner

donor_geno = torch.from_numpy(np.load(GENO / "donor_geno.npy").astype(np.float32))
donor_geno_mask = torch.from_numpy(np.load(GENO / "donor_geno_mask.npy"))
owner_lut = build_owner_lut(donor_geno, donor_geno_mask)

clean = load_module("st_clean", HERE / "models" / "set_transformer.py").SetTransformerMixture(
    n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32, n_classes=45,
    dropout=0.1, n_token_feats=8, periodic_sigma=0.3, n_slot_iters=3, ot_eps=0.05, ot_iters=5,
    donor_geno=donor_geno, donor_geno_mask=donor_geno_mask, owner_lut=owner_lut).to(DEVICE)
sd = torch.load(CKPT, weights_only=True, map_location=DEVICE)
mc, _ = clean.load_state_dict(sd, strict=False); assert not mc
clean.eval()

tok = np.load(DATA / "tokens8_test.npy").astype(np.float32)
msk = np.load(DATA / "mask_test.npy")
at  = np.load(DATA / "attr_test.npy"); noc = np.load(DATA / "noc_test.npy")
y_true = np.load(PROJ / "results/inc22_fixed_aslot_seed42/y_test_true.npy")   # (N,45) present set
g = donor_geno.numpy(); gm = donor_geno_mask.numpy()

# ---- forward: raw logits_attr (N,160,46) ----
logits = np.zeros((tok.shape[0], tok.shape[1], K + 1), np.float32)
with torch.no_grad():
    for i in range(0, tok.shape[0], 256):
        out = clean(torch.from_numpy(tok[i:i+256]).to(DEVICE), torch.from_numpy(msk[i:i+256]).to(DEVICE))
        logits[i:i+256] = out["logits_attr"].cpu().numpy()

real = at >= 0
true = at[real]
L = logits[real]                                   # (n_real, 46) raw logits

# ===== (a) TEMPERATURE — argmax is scale-invariant, so top-1 acc cannot change =====
print("=== (a) temperature scaling of logits_attr ===")
print("  T    top1_acc   mean_top1_prob   (argmax/top-1 is mathematically invariant to T)")
for T in [0.5, 1.0, 2.0, 4.0, 8.0]:
    p = torch.softmax(torch.from_numpy(L / T), -1).numpy()
    top1 = p.argmax(1)
    acc = (top1 == true).mean()
    print(f"  {T:>3}   {acc:.4f}     {np.sort(p,1)[:,-1].mean():.3f}")

# ===== build carrier map (locus, round(allele*10)) -> list of donors =====
carr = {}
for c in range(g.shape[0]):
    for j in range(g.shape[1]):
        if gm[c, j]:
            key = (int(round(g[c, j, 0])), int(round(g[c, j, 1] * 10)))
            carr.setdefault(key, [])
            if c not in carr[key]: carr[key].append(c)

# ===== (b) symbolic soft-split (uniform-compat height EM), per-peak owner =====
def symbolic_owner(i, n_iters=10):
    """returns dict peak_idx -> argmax symbolic owner among 45 panel carriers."""
    m = msk[i].astype(bool)
    idx, keys = [], []
    for p in np.where(m)[0]:
        key = (int(round(tok[i, p, 0])), int(round(tok[i, p, 1] * 10)))
        if key in carr: idx.append(p); keys.append(key)
    if not idx: return {}
    h = np.expm1(tok[i, idx, 2].astype(np.float64))
    n = len(idx); S = np.full((n, K + 1), -1e9)
    for r, key in enumerate(keys):
        for c in carr[key]: S[r, c] = 0.0
        S[r, K] = -2.0
    phi = np.ones(K + 1) / (K + 1)
    for _ in range(n_iters):
        z = S + np.log(phi + 1e-9); z -= z.max(1, keepdims=True)
        A = np.exp(z); A /= A.sum(1, keepdims=True)
        w = (A[:, :K] * h[:, None]).sum(0); bg = (A[:, K] * h).sum()
        phi = np.concatenate([w, [bg]]) / max(w.sum() + bg, 1e-9)
    return {idx[r]: int(A[r].argmax()) for r in range(n)}

# classify each real peak shared/private among the sample's TRUE present donors
neu_top1 = logits.argmax(-1)        # neural argmax owner per (sample,peak)
rows = []
for i in range(tok.shape[0]):
    pres = set(np.where(y_true[i] > 0.5)[0])
    sym = symbolic_owner(i)
    for p in np.where(real[i])[0]:
        o = int(at[i, p])
        key = (int(round(tok[i, p, 0])), int(round(tok[i, p, 1] * 10)))
        ncar = sum(1 for c in carr.get(key, []) if c in pres)   # true carriers of this allele
        shared = ncar >= 2
        rows.append((noc[i], shared, o, int(neu_top1[i, p]), sym.get(p, -1)))

rows = np.array(rows)
nocs, sh, ow, neu, sym = rows[:,0], rows[:,1].astype(bool), rows[:,2], rows[:,3], rows[:,4]
valid_sym = sym >= 0
def acc(mask_):
    m = mask_ & valid_sym
    return (neu[m]==ow[m]).mean(), (sym[m]==ow[m]).mean(), m.sum()

print("\n=== (b) per-peak owner accuracy: NEURAL argmax vs SYMBOLIC soft-split ===")
print("  group          n        neural   symbolic")
for name, mk in [("PRIVATE peaks", ~sh), ("SHARED peaks", sh), ("ALL", np.ones_like(sh))]:
    an, as_, n = acc(mk)
    print(f"  {name:<14} {n:>7,}   {an:.3f}    {as_:.3f}")
print("\n  SHARED peaks by NOC:")
print("  NOC     n       neural   symbolic")
for v in [2,3,4,5]:
    an, as_, n = acc(sh & (nocs==v))
    if n: print(f"   {v}    {n:>6,}    {an:.3f}    {as_:.3f}")

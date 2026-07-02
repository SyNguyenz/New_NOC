"""Forward inc22 checkpoint, dump SOFT logits_attr (B,N,46), measure peakedness + top-k recovery.
Answers: is attr_head a hard 1-peak-1-donor assign, or a soft rank over 45 donors (+bg)?"""
import importlib.util, numpy as np, torch
from pathlib import Path

PROJ = Path("."); HERE = PROJ / "inc22_clean"
CKPT = PROJ / "results/inc22_fixed_aslot_seed42/Donor-Slot_Set_Transformer.pt"
DATA = PROJ / "data_insilico_w"; GENO = PROJ / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLELE_OFF, LUT_W = 50, 1024

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
mc, uc = clean.load_state_dict(sd, strict=False)
assert not mc, f"missing keys: {list(mc)}"
clean.eval()

tok = torch.from_numpy(np.load(DATA / "tokens8_test.npy").astype(np.float32))
msk = torch.from_numpy(np.load(DATA / "mask_test.npy"))
at  = np.load(DATA / "attr_test.npy"); noc = np.load(DATA / "noc_test.npy")
K = 45

# forward in batches, collect softmax over 46 classes
probs = np.zeros((tok.shape[0], tok.shape[1], K + 1), np.float32)
with torch.no_grad():
    for i in range(0, tok.shape[0], 256):
        out = clean(tok[i:i+256].to(DEVICE), msk[i:i+256].to(DEVICE))
        probs[i:i+256] = torch.softmax(out["logits_attr"], -1).cpu().numpy()

real = at >= 0
P = probs[real]                       # (n_real, 46) soft distribution per peak
true = at[real]
order = np.argsort(-P, axis=1)        # rank donors by prob, desc
top1 = order[:, 0]
p_sorted = np.sort(P, axis=1)[:, ::-1]

# peakedness
ent = -(P * np.log(P + 1e-12)).sum(1)
print("=== SHAPE of attr_head output ===")
print(f"  logits_attr per peak = vector over {K+1} classes (45 donors + 1 background) -> SOFT rank, not a single index")
print()
print("=== how PEAKED is the soft distribution? (real peaks, n={:,}) ===".format(real.sum()))
print(f"  mean top-1 prob          = {p_sorted[:,0].mean():.3f}")
print(f"  mean top-2 prob          = {p_sorted[:,1].mean():.3f}")
print(f"  mean top-3 prob          = {p_sorted[:,2].mean():.3f}")
print(f"  mean entropy (max={np.log(K+1):.2f}) = {ent.mean():.3f}   ({ent.mean()/np.log(K+1)*100:.0f}% of uniform)")
print(f"  frac peaks with top-1 prob > 0.90 = {(p_sorted[:,0]>0.90).mean():.3f}")
print(f"  frac peaks with top-1 prob > 0.99 = {(p_sorted[:,0]>0.99).mean():.3f}")
print()
print("=== does the RANKING carry info beyond argmax? top-k recovery of true owner ===")
for k in [1, 2, 3, 5]:
    hit = (order[:, :k] == true[:, None]).any(1).mean()
    print(f"  top-{k} acc = {hit:.3f}")
print()
print("=== top-k by NOC ===")
print("  NOC   top1    top2    top3")
nr = noc[np.where(real)[0]]
for v in sorted(np.unique(noc)):
    m = nr == v
    if m.sum() == 0: continue
    t1 = (order[m, :1] == true[m, None]).any(1).mean()
    t2 = (order[m, :2] == true[m, None]).any(1).mean()
    t3 = (order[m, :3] == true[m, None]).any(1).mean()
    print(f"   {int(v)}    {t1:.3f}   {t2:.3f}   {t3:.3f}")

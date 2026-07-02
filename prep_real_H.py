"""Cache frozen-inc22 H (attr_head input) + cls_logits on REAL val/test, for the EMAttr experiment."""
import numpy as np, torch, importlib.util
from pathlib import Path
PROJ = Path("."); REAL = PROJ / "data"; CKPT = PROJ / "results/inc22_fixed_aslot_seed42/Donor-Slot_Set_Transformer.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"; ALLELE_OFF, LUT_W = 30, 1024
OUT = PROJ / "cache_real"; OUT.mkdir(exist_ok=True)
def lm(n, p): s = importlib.util.spec_from_file_location(n, str(p)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
stc = lm("stc", PROJ / "inc22_clean" / "models" / "set_transformer.py")
def owner_lut(gg, gm, n=45):
    gm = gm.bool(); o = torch.zeros(24, LUT_W, n)
    for c in range(min(n, gg.size(0))):
        for j in range(gg.size(1)):
            if gm[c, j]:
                li = int(gg[c, j, 0]); ab = int(round(float(gg[c, j, 1]) * 10)) + ALLELE_OFF
                if 0 <= li < 24 and 0 <= ab < LUT_W: o[li, ab, c] = 1.0
    return o
dg = torch.from_numpy(np.load(REAL/"donor_geno.npy").astype(np.float32)); dgm = torch.from_numpy(np.load(REAL/"donor_geno_mask.npy"))
clean = stc.SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32, n_classes=45,
    dropout=0.1, n_token_feats=8, periodic_sigma=0.3, n_slot_iters=3, ot_eps=0.05, ot_iters=5,
    donor_geno=dg, donor_geno_mask=dgm, owner_lut=owner_lut(dg, dgm)).to(DEV)
clean.load_state_dict(torch.load(CKPT, weights_only=True, map_location=DEV), strict=False); clean.eval()
for sp in ["val", "test"]:
    tok = np.load(REAL/f"tokens8_{sp}.npy").astype(np.float32); msk = np.load(REAL/f"mask_{sp}.npy")
    H = np.zeros((len(tok), tok.shape[1], 128), np.float32); L = np.zeros((len(tok), 45), np.float32)
    with torch.no_grad():
        for i in range(0, len(tok), 256):
            sl = slice(i, i+256); t = torch.from_numpy(tok[sl]).to(DEV); m = torch.from_numpy(msk[sl]).to(DEV)
            H[sl] = clean._encode_set(t, m)[1].cpu().numpy()
            L[sl] = clean(t, m)["logits_cls"].cpu().numpy()
    np.save(OUT/f"H_{sp}.npy", H); np.save(OUT/f"cls_{sp}.npy", L)
    print(f"{sp}: H {H.shape}  cls {L.shape}  saved")

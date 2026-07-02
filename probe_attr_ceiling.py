"""Can attr_head reach 0.9 owner-acc? Test the CEILING: freeze the inc22 encoder, train a FRESH head
(same Dropout+Linear arch as attr_head) on the HARD owner target (attr label) instead of the soft phi*CN
target. If owner-acc jumps toward ~0.9, the owner info IS in H (per F28's 0.97 linear-probe) and the
current attr_head underperforms only because of its soft-split target."""
import numpy as np, torch, importlib.util
from pathlib import Path
PROJ = Path("."); DATA = PROJ / "data_insilico_w"; GENO = PROJ / "data"
CKPT = PROJ / "results/inc22_fixed_aslot_seed42/Donor-Slot_Set_Transformer.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"; ALLELE_OFF, LUT_W, K = 30, 1024, 45
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
dg = torch.from_numpy(np.load(GENO/"donor_geno.npy").astype(np.float32)); dgm = torch.from_numpy(np.load(GENO/"donor_geno_mask.npy"))
clean = stc.SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32, n_classes=45,
    dropout=0.1, n_token_feats=8, periodic_sigma=0.3, n_slot_iters=3, ot_eps=0.05, ot_iters=5,
    donor_geno=dg, donor_geno_mask=dgm, owner_lut=owner_lut(dg, dgm)).to(DEV)
clean.load_state_dict(torch.load(CKPT, weights_only=True, map_location=DEV), strict=False)
for p in clean.parameters(): p.requires_grad_(False)
clean.eval()

def load(sp): return {k: np.load(DATA/f"{k}_{sp}.npy") for k in ["tokens8", "mask", "attr", "noc"]}
tr, te = load("train"), load("test")
y_te = np.load(PROJ/"results/inc22_fixed_aslot_seed42/y_test_true.npy")

head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(128, K + 1)).to(DEV)  # same arch as attr_head
opt = torch.optim.Adam(head.parameters(), lr=1e-3)
def H_of(tok, msk):                                    # the attr_head input H, frozen
    with torch.no_grad(): return clean._encode_set(tok, msk)[1]
def bat(d, bs, sh):
    n = len(d["tokens8"]); idx = np.random.permutation(n) if sh else np.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i+bs]
        yield (torch.from_numpy(d["tokens8"][j]).float().to(DEV), torch.from_numpy(d["mask"][j]).to(DEV),
               torch.from_numpy(d["attr"][j]).long().to(DEV))
for ep in range(6):
    head.train(); tot = 0; nb = 0
    for tok, msk, attr in bat(tr, 256, True):
        H = H_of(tok, msk); logits = head(H)                    # (B,N,K+1)
        tgt = attr.clone(); tgt[tgt < 0] = K                     # hard owner; -1 (no owner) -> bg class K
        valid = msk.bool()
        loss = torch.nn.functional.cross_entropy(logits[valid], tgt[valid])
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
    print(f"ep{ep} ce {tot/nb:.3f}")

# eval owner-acc (argmax over 46, real peaks) — same protocol as the 0.846 figure
head.eval(); N = len(te["tokens8"]); pred = np.zeros((N, te["tokens8"].shape[1]), np.int64)
with torch.no_grad():
    for i in range(0, N, 256):
        sl = slice(i, i+256)
        H = H_of(torch.from_numpy(te["tokens8"][sl]).float().to(DEV), torch.from_numpy(te["mask"][sl]).to(DEV))
        pred[sl] = head(H).argmax(-1).cpu().numpy()
attr = te["attr"]; noc = te["noc"]; g = dg.numpy(); gm = dgm.numpy()
carr = {}
for c in range(g.shape[0]):
    for j in range(g.shape[1]):
        if gm[c, j]:
            key = (int(round(g[c, j, 0])), int(round(g[c, j, 1]*10))); carr.setdefault(key, set()).add(c)
real = attr >= 0; rows = []
for i in range(N):
    pres = set(np.where(y_te[i] > 0.5)[0])
    for p in np.where(real[i])[0]:
        key = (int(round(te["tokens8"][i, p, 0])), int(round(te["tokens8"][i, p, 1]*10)))
        nc = sum(1 for c in carr.get(key, ()) if c in pres)
        rows.append((noc[i], nc >= 2, int(attr[i, p]), int(pred[i, p])))
rows = np.array(rows); ncs, sh, ow, pr = rows[:,0], rows[:,1].astype(bool), rows[:,2], rows[:,3]
def acc(m): return (pr[m] == ow[m]).mean(), m.sum()
print("\n=== owner-acc, FRESH head on frozen H trained on HARD owner ===")
print("            (ref: attr_head 0.846/0.546  symbolic 0.887/0.743)")
for nm, mk in [("PRIVATE", ~sh), ("SHARED", sh), ("ALL", np.ones_like(sh))]:
    a, n = acc(mk); print(f"  {nm:<8} {n:>8,}  {a:.3f}")
print("  SHARED by NOC:")
for v in [2,3,4,5]:
    a, n = acc(sh & (ncs == v))
    if n: print(f"   N{v}  {n:>6,}  {a:.3f}")

"""
Direct encoder-information probe (NOT a metric readout).

Question: does the inc7 encoder LOSE minor-donor information?
Method: load checkpoint, run forward, extract per-peak representations at TWO points:
    x0 = pre-encoder projected token (input side)
    H  = post-encoder ISAB/MassISAB token (what the decoder + attr head read)
Fit an INDEPENDENT linear probe  rep[peak] -> global donor id (45-way)  on TRAIN-combo
peaks, then test on REAL TEST (novel-combo) peaks, stratified by donor height-rank.

If the encoder washes minors:  probe-acc(H, minor) << probe-acc(x0, minor).
If minor info is preserved:     probe-acc(H, minor) ~= probe-acc(x0, minor), >> chance.
This isolates "info in the representation" from "the model's own decoder uses it",
so a weak/buggy lever cannot fake the answer.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture

DATA = Path("data_insilico_w")
RUN  = Path(sys.argv[1] if len(sys.argv) > 1 else "results/inc7_masspool_seed42")
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

cfg = json.load(open(RUN / "metrics.json"))["config"]
n_tok = cfg["n_token_feats"]

# ---- build model exactly as trainer, load weights (feat_mean/std restored from ckpt) ----
dg = dgm = None
if cfg.get("geno_query"):
    dg  = torch.from_numpy(np.load(DATA/"donor_geno.npy").astype(np.float32))
    dgm = torch.from_numpy(np.load(DATA/"donor_geno_mask.npy"))
model = SetTransformerMixture(
    n_loci=cfg.get("n_loci",24), d_locus=cfg.get("d_locus",16), d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4), n_isab=cfg.get("n_isab",2), m_inducing=cfg.get("m_inducing",32),
    n_classes=cfg.get("n_classes",45), n_noc=cfg.get("n_noc",6), dropout=cfg.get("dropout",0.1),
    cls_decoder=cfg.get("cls_decoder","pooled"), decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok, encoder=cfg.get("encoder","isab"), dec_layers=cfg.get("dec_layers",2),
    num_embed=cfg.get("num_embed","raw"), n_freq=cfg.get("n_freq",8), d_num_emb=cfg.get("d_num_emb",8),
    periodic_sigma=cfg.get("periodic_sigma",1.0), aux_heads=cfg.get("aux_heads",False),
    sparse_attn=cfg.get("sparse_attn",False), geno_query=cfg.get("geno_query",False),
    donor_geno=dg, donor_geno_mask=dgm, vib=cfg.get("vib",False), mass_pool=cfg.get("mass_pool",False),
).to(DEV)
sd = torch.load(RUN/"best_model.pt", map_location=DEV, weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded {RUN.name} | mass_pool={cfg.get('mass_pool')} | missing={len(missing)} unexpected={len(unexpected)}")
print(f"  sanity: MassMABpp scale present in ckpt = {'encoder.0.mab0.scale' in sd}  value={float(sd.get('encoder.0.mab0.scale', torch.tensor(float('nan')))):.3f}")
model.eval()

def load(split):
    tk = np.load(DATA/f"tokens8_test.npy") if split=="test" else np.load(DATA/f"tokens8_{split}.npy")
    mk = np.load(DATA/f"mask_{split}.npy").astype(bool)
    at = np.load(DATA/f"attr_{split}.npy")
    nc = np.load(DATA/f"noc_{split}.npy")
    return tk[:,:, :n_tok].astype(np.float32), mk, at, nc

@torch.no_grad()
def encode(tk, mk, idxs, bs=128):
    """Return per-peak x0, H, donor, height-rank, noc for the given sample indices."""
    X0=[]; HH=[]; DON=[]; RNK=[]; NOC=[]; SID=[]
    at_all = AT_CACHE;
    for s in range(0, len(idxs), bs):
        sel = idxs[s:s+bs]
        t = torch.from_numpy(tk[sel]).to(DEV)
        m = torch.from_numpy(mk[sel]).to(DEV)
        x0, H, _ = model._encode_set(t, m)
        x0=x0.cpu().numpy(); H=H.cpu().numpy()
        for j,gi in enumerate(sel):
            a = at_all[gi]; valid = np.where(a>=0)[0]
            if len(valid)==0: continue
            logh = tk[gi][:,2]  # feature idx 2 = log_h
            don = a[valid]
            # height rank per donor (0=major)
            dh = {int(d): float(np.exp(logh[a==d]).sum()) for d in np.unique(don)}
            order = sorted(dh, key=lambda d:-dh[d]); ro={d:r for r,d in enumerate(order)}
            X0.append(x0[j][valid]); HH.append(H[j][valid]); DON.append(don)
            RNK.append(np.array([ro[int(d)] for d in don])); NOC.append(np.full(len(valid), NOC_CACHE[gi]))
            SID.append(np.full(len(valid), gi))
    return (np.concatenate(X0), np.concatenate(HH), np.concatenate(DON).astype(int),
            np.concatenate(RNK).astype(int), np.concatenate(NOC).astype(int),
            np.concatenate(SID).astype(int))

# ---- gather peaks ----
tk_tr, mk_tr, at_tr, nc_tr = load("train")
tk_te, mk_te, at_te, nc_te = load("test")

# fit probe on a balanced subsample of TRAIN samples (all NOC)
rng = np.random.default_rng(0)
tr_idx = rng.choice(len(nc_tr), size=8000, replace=False)
fit_idx, evl_idx = tr_idx[:6000], tr_idx[6000:]   # SEEN-combo held-out for probe eval
AT_CACHE, NOC_CACHE = at_tr, nc_tr
x0_tr, H_tr, d_tr, r_tr, n_tr, s_tr = encode(tk_tr, mk_tr, fit_idx)
x0_ev, H_ev, d_ev, r_ev, n_ev, s_ev = encode(tk_tr, mk_tr, evl_idx)
AT_CACHE, NOC_CACHE = at_te, nc_te
x0_te, H_te, d_te, r_te, n_te, s_te = encode(tk_te, mk_te, np.arange(len(nc_te)))
print(f"train peaks {len(d_tr)} | test peaks {len(d_te)}")

# ---- linear probe (torch, GPU): rep -> 45-way donor ----
def fit_probe(Xtr, ytr, Xte, dim, epochs=60):
    clf = nn.Linear(dim, 45).to(DEV)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr).to(DEV); yt = torch.from_numpy(ytr).long().to(DEV)
    lossf = nn.CrossEntropyLoss()
    n=len(yt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, 8192):
            b = perm[s:s+8192]
            opt.zero_grad(); loss = lossf(clf(Xt[b]), yt[b]); loss.backward(); opt.step()
    with torch.no_grad():
        pr = clf(torch.from_numpy(Xte).to(DEV)).argmax(1).cpu().numpy()
    return pr

def fit_probe2(Xtr, ytr, Xte, dim, epochs=60):
    clf = nn.Linear(dim, 45).to(DEV)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr).to(DEV); yt = torch.from_numpy(ytr).long().to(DEV)
    lossf = nn.CrossEntropyLoss(); n=len(yt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for s in range(0,n,8192):
            b=perm[s:s+8192]; opt.zero_grad(); lossf(clf(Xt[b]),yt[b]).backward(); opt.step()
    with torch.no_grad():
        return [clf(torch.from_numpy(X).to(DEV)).argmax(1).cpu().numpy() for X in Xte]

pr_x0 = fit_probe(x0_tr, d_tr, x0_te, x0_tr.shape[1])
prH_list = fit_probe2(H_tr, d_tr, [H_te, H_ev], H_tr.shape[1])
pr_H, pr_Hev = prH_list[0], prH_list[1]

# SEEN vs NOVEL combo readability of the FAINTEST minors (rank>=3) from the SAME encoder H:
print("\n=== Encoder-H readability of FAINT minors: SEEN-combo (train held-out) vs NOVEL-combo (real test) ===")
for lab, pr, dd, rr, nn_ in [("SEEN combos (held-out train)", pr_Hev, d_ev, r_ev, n_ev),
                             ("NOVEL combos (real test)    ", pr_H,  d_te, r_te, n_te)]:
    mm = (nn_==5)&(rr>=3)
    print(f"  {lab}: N5 faint-minor probe(H) acc = {(pr[mm]==dd[mm]).mean():.3f}  (n={mm.sum()})")
print("  => HIGH on seen, LOW on novel for the SAME encoder = combo-dependent representation (generalization),")
print("     NOT a fixed capacity/washing loss (which would be low on both).")

# ---- report: per-peak donor-id accuracy by height-rank, N5 (and N4) ----
print("\n=== INDEPENDENT linear probe: recover GLOBAL donor id from representation ===")
print("    (fit on TRAIN-combo peaks, tested on REAL TEST = novel combos; chance=1/45=0.022)")
for target in (5,4):
    print(f"\n  NOC{target}  height-rank:        " + "  ".join(f"r{r}{'=maj' if r==0 else ('=min' if r==target-1 else '')}" for r in range(target)))
    for tag,pr in [("probe on x0 (INPUT)   ", pr_x0), ("probe on H  (ENCODER) ", pr_H)]:
        accs=[]
        for r in range(target):
            m = (n_te==target)&(r_te==r)
            accs.append((pr[m]==d_te[m]).mean() if m.sum() else float('nan'))
        print(f"    {tag}: " + "  ".join(f"{a:.3f}" for a in accs) + f"   | minor={accs[-1]:.3f}")
# ---- CLINCHER: for donors the model's SET head DROPPED, can the independent probe read them from H? ----
yp = np.load(RUN/"y_test_pred.npy"); yt = np.load(RUN/"y_test_true.npy")
ap = np.load(RUN/"attr_pred_test.npy")
print("\n=== Among N5 MINOR/low-rank donors: split by whether the model's SET head KEPT vs DROPPED the donor ===")
print("    (does the info exist in H for the very donors the model fails on?)")
for which,lo,hi in [("kept",None,None),("dropped",None,None)]:
    pass
m5 = (n_te==5)
is_dropped = np.array([ (yt[s,d]==1 and yp[s,d]==0) for s,d in zip(s_te, d_te) ])
# restrict to the 2 lowest-rank (minor) donors where misses concentrate
minor = m5 & (r_te>=3)
for label, sub in [("KEPT minors   ", minor & ~is_dropped), ("DROPPED minors", minor & is_dropped)]:
    n=sub.sum()
    probeH = (pr_H[sub]==d_te[sub]).mean() if n else float('nan')
    probeX = (pr_x0[sub]==d_te[sub]).mean() if n else float('nan')
    modelattr = np.mean([ (ap[s][np.load(DATA/'attr_test.npy')[s]==d]==d).mean() for s,d in zip(s_te[sub], d_te[sub]) ]) if n else float('nan')
    print(f"  {label} (n={n:4d} donors): INDEP-probe(H)={probeH:.3f}  indep-probe(x0)={probeX:.3f}  model-attr-head={modelattr:.3f}")
print("\n  => If INDEP-probe(H) stays HIGH on DROPPED minors while model-attr-head collapses,")
print("     the info IS in the encoder output; the model's DECODER fails to use it (not info-loss).")

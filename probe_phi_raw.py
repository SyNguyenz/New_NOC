"""
Raw diagnostic for EM-uniform phi on production model's N5 missed cases.
Checks:
  1. Raw phi distribution: missed-true vs decoy (histogram summary)
  2. AUC sensitivity to NITER / BG_PRIOR
  3. % cases phi(true) > phi(decoy)  -- direct check, not just AUC
  4. Why decoy beats true: is it because decoy has more covered peaks, or higher height?
"""
import os, json
from pathlib import Path
import numpy as np, torch
from models.set_transformer import SetTransformerMixture

DA = Path("data_insilico_w")
RUN = Path(os.environ.get("RUN", "results/inc6_maskp_seed42"))
G = "data/donor_geno.npy"
DEVc = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ab(a): return int(round(float(a)*10))
def kk(l, a): return (int(round(float(l))), ab(a))
g = np.load(G); gm = np.load(G.replace(".npy","_mask.npy")).astype(bool); C = g.shape[0]
carr = {}
for c in range(C):
    for j in range(g.shape[1]):
        if gm[c,j]: carr.setdefault(kk(g[c,j,0], g[c,j,1]), []).append(c)

cfg = json.load(open(RUN/"metrics.json"))["config"]; n_tok = cfg.get("n_token_feats", 8)
m = SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
    dropout=0.1,cls_decoder="per_donor",decoder_source="encoded",n_token_feats=n_tok,encoder="isab++",dec_layers=2,
    num_embed="periodic",n_freq=8,d_num_emb=8,periodic_sigma=0.3,aux_heads=True,sparse_attn=True).to(DEVc)
sd = torch.load(RUN/"best_model.pt", map_location=DEVc)
sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
m.load_state_dict(sd, strict=False); m.eval()

tk = np.load(DA/f"tokens{n_tok}_test.npy").astype(np.float32)
mk = np.load(DA/"mask_test.npy")
yt = np.load(DA/"y_test_set.npy").astype(bool)
nt = np.clip(np.load(DA/"noc_test.npy"), 1, 5)
H = np.expm1(tk[:,:,2])

@torch.no_grad()
def clslogits(t, k):
    o = []
    for i in range(0, len(t), 128):
        r = m(torch.from_numpy(t[i:i+128]).to(DEVc), torch.from_numpy(k[i:i+128].astype(bool)).to(DEVc))
        o.append(r["logits_cls"].cpu().numpy())
    return np.concatenate(o)

Lt = clslogits(tk, mk)
sel5 = np.where(nt == 5)[0]

def em_phi(idx, niter=5, bg=0.02):
    """Compute EM-uniform phi for a batch of sample indices."""
    PH = np.zeros((len(idx), C))
    for ii, i in enumerate(idx):
        peaks = [(kk(tk[i,k,0], tk[i,k,1]), H[i,k]) for k in np.where(mk[i])[0]
                 if kk(tk[i,k,0], tk[i,k,1]) in carr]
        if not peaks: continue
        n = len(peaks); h = np.array([p[1] for p in peaks])
        S = np.full((n, C+1), -1e9)
        for r, (it, _) in enumerate(peaks):
            for c in carr[it]: S[r,c] = 0.0
            S[r,C] = -2.0
        phi = np.ones(C+1)/(C+1); phi[C] = bg
        for _ in range(niter):
            z = S + np.log(phi+1e-9); z -= z.max(1, keepdims=True)
            A = np.exp(z); A /= A.sum(1, keepdims=True)
            w = (A[:,:C]*h[:,None]).sum(0); bg_h = (A[:,C]*h).sum()
            phi = np.concatenate([w, [bg_h]]) / max(w.sum()+bg_h, 1e-9)
        PH[ii] = phi[:C]
    return PH

def auc(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    if not len(p) or not len(q): return float("nan")
    a = np.concatenate([p,q]); _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); rk = ((cs-cnt+cs+1)/2.0)[inv]
    return (rk[:len(p)].sum() - len(p)*(len(p)+1)/2) / (len(p)*len(q))

# === Main: NITER=5, default ===
PH5 = em_phi(sel5, niter=5)
top5 = np.argsort(Lt[sel5], axis=1)[:,::-1][:,:5]

mt_phi=[]; dc_phi=[]; mt_h=[]; dc_h=[]
mt_ncov=[]; dc_ncov=[]  # #genotype-covered peaks for missed-true vs decoy
win=0; lose=0; tie=0
for ii, i in enumerate(sel5):
    miss = [c for c in np.where(yt[i])[0] if c not in set(top5[ii])]
    dec  = [c for c in top5[ii] if not yt[i,c]]
    # raw phi
    for c in miss:
        mt_phi.append(PH5[ii,c])
        # peak coverage: how many feasible peaks does donor c "own" in the mix?
        mt_h.append(PH5[ii,c])
    for c in dec:
        dc_phi.append(PH5[ii,c])
        dc_h.append(PH5[ii,c])
    # direct comparison: for each missed-true, vs best decoy phi
    if miss and dec:
        best_true = max(PH5[ii,c] for c in miss)
        best_dec  = max(PH5[ii,c] for c in dec)
        if best_true > best_dec: win += 1
        elif best_true < best_dec: lose += 1
        else: tie += 1

mt_phi = np.array(mt_phi); dc_phi = np.array(dc_phi)

print(f"=== Raw phi diagnostic — {RUN.name} — N5 (n={len(sel5)} samples) ===\n")
print(f"  Missed-true phi:  mean={mt_phi.mean():.4f}  median={np.median(mt_phi):.4f}  p25={np.percentile(mt_phi,25):.4f}  p75={np.percentile(mt_phi,75):.4f}")
print(f"  Decoy phi:        mean={dc_phi.mean():.4f}  median={np.median(dc_phi):.4f}  p25={np.percentile(dc_phi,25):.4f}  p75={np.percentile(dc_phi,75):.4f}")
print(f"  AUC(missed_true > decoy) = {auc(mt_phi, dc_phi):.3f}   (from probe_softsplit = 0.686)")
print(f"  Direct: best_true > best_dec = {win}/{win+lose+tie} = {win/max(1,win+lose+tie):.3f}")
print(f"           best_true < best_dec = {lose}/{win+lose+tie} = {lose/max(1,win+lose+tie):.3f}")

# === Sensitivity: NITER ===
print(f"\n  --- Sensitivity to NITER (bg=0.02) ---")
for nit in [1, 3, 5, 10, 20]:
    PHt = em_phi(sel5, niter=nit)
    mt2=[]; dc2=[]
    for ii, i in enumerate(sel5):
        miss=[c for c in np.where(yt[i])[0] if c not in set(top5[ii])]
        dec=[c for c in top5[ii] if not yt[i,c]]
        for c in miss: mt2.append(PHt[ii,c])
        for c in dec: dc2.append(PHt[ii,c])
    print(f"  NITER={nit:2d}  AUC={auc(mt2,dc2):.3f}")

# === Sensitivity: BG_PRIOR ===
print(f"\n  --- Sensitivity to BG_PRIOR (niter=5) ---")
for bg in [0.001, 0.01, 0.02, 0.05, 0.1]:
    PHt = em_phi(sel5, niter=5, bg=bg)
    mt2=[]; dc2=[]
    for ii, i in enumerate(sel5):
        miss=[c for c in np.where(yt[i])[0] if c not in set(top5[ii])]
        dec=[c for c in top5[ii] if not yt[i,c]]
        for c in miss: mt2.append(PHt[ii,c])
        for c in dec: dc2.append(PHt[ii,c])
    print(f"  BG={bg:.3f}  AUC={auc(mt2,dc2):.3f}")

# === Why does decoy beat true? ===
# For each "lose" case (decoy phi > true phi), inspect
print(f"\n  --- Why decoy beats true (phi analysis) ---")
case_phi_true=[]; case_phi_dec=[]; case_nprivate_true=[]; case_ncov_dec=[]
for ii, i in enumerate(sel5):
    miss=[c for c in np.where(yt[i])[0] if c not in set(top5[ii])]
    dec=[c for c in top5[ii] if not yt[i,c]]
    if not miss or not dec: continue
    best_true=max(miss, key=lambda c: PH5[ii,c])
    best_dec=max(dec, key=lambda c: PH5[ii,c])
    if PH5[ii,best_dec] >= PH5[ii,best_true]:  # decoy wins
        case_phi_true.append(PH5[ii,best_true])
        case_phi_dec.append(PH5[ii,best_dec])
        # count private alleles of missed-true present in mix
        true_alleles = {kk(g[best_true,j,0], g[best_true,j,1]) for j in range(g.shape[1]) if gm[best_true,j]}
        present_keys = {kk(tk[i,k,0], tk[i,k,1]) for k in np.where(mk[i])[0]}
        private_present = [it for it in true_alleles if it in present_keys and len(carr.get(it,[])) == 1]
        case_nprivate_true.append(len(private_present))
        # decoy peak coverage
        dec_alleles = {kk(g[best_dec,j,0], g[best_dec,j,1]) for j in range(g.shape[1]) if gm[best_dec,j]}
        cov_dec = len(dec_alleles & present_keys)
        case_ncov_dec.append(cov_dec)

case_phi_true=np.array(case_phi_true); case_phi_dec=np.array(case_phi_dec)
case_nprivate=np.array(case_nprivate_true); case_ncov_dec=np.array(case_ncov_dec)
print(f"  'Decoy beats true' cases: {len(case_phi_true)}/{win+lose+tie}")
if len(case_phi_true):
    print(f"  Missed-true phi:   mean={case_phi_true.mean():.4f}  median={np.median(case_phi_true):.4f}")
    print(f"  Decoy phi:         mean={case_phi_dec.mean():.4f}  median={np.median(case_phi_dec):.4f}")
    print(f"  Private alleles of missed-true: mean={case_nprivate.mean():.2f}  median={np.median(case_nprivate):.1f}  =0: {(case_nprivate==0).mean():.2f}")
    print(f"  Decoy peak coverage (#present alleles): mean={case_ncov_dec.mean():.2f}  median={np.median(case_ncov_dec):.1f}")

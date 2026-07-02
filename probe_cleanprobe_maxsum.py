"""
DECISIVE: is the ~.07 base "decoder under-read" I claimed real, or an attr_head artifact?

Re-uses F33's CLEAN combo-disjoint linear probe (probe_decoder_waste.py: fit a fresh per-peak 45-way
linear probe on TRAIN H, seed 0, NON-memorizing) but assembles the set TWO ways and compares to the
trained decoder, per NOC oracle EM on real test:
   sum-vote = sum_peaks softmax(probe)[:,d]          (F33's original readout)
   max-vote = max_peaks softmax(probe)[:,d]          (the stronger readout I used with attr_head)

If clean-probe MAX beats decoder by ~.07 at N5  => readout-strength under-read is REAL on base.
If clean-probe MAX ~= decoder (~0)              => my .07 was the attr_head (privileged, combo-mem) talking.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture

DATA = Path("data_insilico_w")
RUN  = Path(sys.argv[1] if len(sys.argv) > 1 else "results/inc2_2d_sparse_seed42")
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

cfg = json.load(open(RUN / "metrics.json"))["config"]
n_tok = cfg["n_token_feats"]
model = SetTransformerMixture(
    n_loci=cfg.get("n_loci",24), d_locus=cfg.get("d_locus",16), d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4), n_isab=cfg.get("n_isab",2), m_inducing=cfg.get("m_inducing",32),
    n_classes=cfg.get("n_classes",45), n_noc=cfg.get("n_noc",6), dropout=cfg.get("dropout",0.1),
    cls_decoder=cfg.get("cls_decoder","pooled"), decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok, encoder=cfg.get("encoder","isab"), dec_layers=cfg.get("dec_layers",2),
    num_embed=cfg.get("num_embed","raw"), n_freq=cfg.get("n_freq",8), d_num_emb=cfg.get("d_num_emb",8),
    periodic_sigma=cfg.get("periodic_sigma",1.0), aux_heads=cfg.get("aux_heads",False),
    sparse_attn=cfg.get("sparse_attn",False), vib=cfg.get("vib",False), mass_pool=cfg.get("mass_pool",False),
    attn_sink=int(cfg.get("attn_sink",0) or 0), donor_recon=cfg.get("donor_recon",False),
).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt", map_location=DEV, weights_only=True), strict=False)
model.eval(); print(f"loaded {RUN.name}")

def load(split):
    tk = np.load(DATA/f"tokens8_{split}.npy")[:,:,:n_tok].astype(np.float32)
    mk = np.load(DATA/f"mask_{split}.npy").astype(bool)
    at = np.load(DATA/f"attr_{split}.npy")
    return tk, mk, at

@torch.no_grad()
def encode_peaks(tk, mk, at, idxs, bs=128):
    HH=[]; DON=[]; SID=[]
    for s in range(0, len(idxs), bs):
        sel = idxs[s:s+bs]
        t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        _, H, _ = model._encode_set(t, m); H=H.cpu().numpy()
        for j,gi in enumerate(sel):
            a=at[gi]; v=np.where(a>=0)[0]
            if len(v)==0: continue
            HH.append(H[j][v]); DON.append(a[v]); SID.append(np.full(len(v), gi))
    return np.concatenate(HH), np.concatenate(DON).astype(int), np.concatenate(SID).astype(int)

@torch.no_grad()
def model_scores(tk, mk, idxs, bs=256):
    out=np.zeros((len(idxs), 45), np.float32)
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]
        o=model(torch.from_numpy(tk[sel]).to(DEV), torch.from_numpy(mk[sel]).to(DEV))
        out[s:s+len(sel)]=torch.sigmoid(o["logits_cls"]).cpu().numpy()
    return out

# ---- fit the SAME clean per-peak probe on TRAIN H (F33 settings) ----
tk_tr, mk_tr, at_tr = load("train")
rng=np.random.default_rng(0)
fit_idx = rng.choice(len(at_tr), size=6000, replace=False)
Htr, dtr, _ = encode_peaks(tk_tr, mk_tr, at_tr, fit_idx)
clf = nn.Linear(Htr.shape[1], 45).to(DEV)
opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
Xt=torch.from_numpy(Htr).to(DEV); yt=torch.from_numpy(dtr).long().to(DEV); lossf=nn.CrossEntropyLoss()
for ep in range(60):
    perm=torch.randperm(len(yt),device=DEV)
    for s in range(0,len(yt),8192):
        b=perm[s:s+8192]; opt.zero_grad(); lossf(clf(Xt[b]),yt[b]).backward(); opt.step()
print(f"clean probe fit on {len(dtr)} train peaks")

@torch.no_grad()
def probe_softmax(H):
    return torch.softmax(clf(torch.from_numpy(H).to(DEV)),1).cpu().numpy()

tk,mk,at = load("test")
true_set={}; noc={}
for gi in range(len(at)):
    a=at[gi]; d=np.unique(a[a>=0])
    if len(d)==0: continue
    true_set[gi]=set(int(x) for x in d); noc[gi]=len(d)
keep=np.array(sorted(true_set))
H,don,sid = encode_peaks(tk,mk,at,keep)
sm = probe_softmax(H)
vsum=defaultdict(lambda: np.zeros(45)); vmax=defaultdict(lambda: np.zeros(45))
for p in range(len(sid)):
    g=int(sid[p]); vsum[g]+=sm[p]; vmax[g]=np.maximum(vmax[g], sm[p])
ms = model_scores(tk,mk,keep); ms_map={int(g):ms[i] for i,g in enumerate(keep)}

print(f"\n  NOC :   decoder   probe_SUM   probe_MAX   (sum-dec)  (max-dec)   n")
for k in range(1,6):
    gis=[g for g in keep if noc[g]==k]
    if not gis: continue
    de=[]; se=[]; xe=[]
    for g in gis:
        ts=true_set[g]
        de.append(set(int(x) for x in np.argsort(ms_map[g])[::-1][:k])==ts)
        se.append(set(int(x) for x in np.argsort(vsum[g])[::-1][:k])==ts)
        xe.append(set(int(x) for x in np.argsort(vmax[g])[::-1][:k])==ts)
    d,s_,x=np.mean(de),np.mean(se),np.mean(xe)
    print(f"  N{k}  :   {d:.3f}     {s_:.3f}       {x:.3f}      {s_-d:+.3f}     {x-d:+.3f}     {len(gis)}")
print("\n  max-dec ~ +.07 at N5  => readout-strength under-read REAL on base (clean probe, no attr_head)")
print("  max-dec ~  0   at N5  => my +.07 was the attr_head (privileged) — base decoder gap ~0")

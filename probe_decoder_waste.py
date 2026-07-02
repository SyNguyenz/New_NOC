"""
DECISIVE test of the 'decoder is the weak link' claim.

Setup: the per-peak donor identity in H GENERALIZES (probe seen~=novel). The model's trained
set-decoder produces an oracle N5 ~0.66 on novel combos. Question: is that a DECODER decision
that overfit combos (=> a SIMPLER, non-memorizing assembler on the SAME H should beat it),
or is it compounding/info (=> nothing simple beats it)?

Build a non-parametric SET assembler from the SAME encoder H:
  1. fit an independent per-peak 45-way donor probe on TRAIN H (combo-disjoint readout).
  2. for each test sample: per-donor score = sum_peaks softmax(probe)[:,d]  (a soft vote).
     predicted set = top-NOC donors (oracle count) -> oracle EM.
  3. compare, sample-for-sample, to the model's own set-head oracle (top-NOC by sigmoid logits).

If probe-assembled oracle >= model oracle on novel N5  => trained decoder WASTES generalizing
   per-peak info (its learned assembly overfit combos) = decoder decision is the lever. CONFIRM.
If probe-assembled oracle <  model oracle               => no simple readout of H beats the
   decoder; the limit is compounding/H-info, not a separable decoder decision. REFUTE.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture

DATA = Path("data_insilico_w")
RUN  = Path(sys.argv[1] if len(sys.argv) > 1 else "results/inc8_v2_vicreg_inv_seed42")
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

cfg = json.load(open(RUN / "metrics.json"))["config"]
n_tok = cfg["n_token_feats"]
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
model.load_state_dict(torch.load(RUN/"best_model.pt", map_location=DEV, weights_only=True), strict=False)
model.eval()
print(f"loaded {RUN.name}")

def load(split):
    tk = np.load(DATA/f"tokens8_{split}.npy")[:,:,:n_tok].astype(np.float32)
    mk = np.load(DATA/f"mask_{split}.npy").astype(bool)
    at = np.load(DATA/f"attr_{split}.npy")
    return tk, mk, at

@torch.no_grad()
def encode_peaks(tk, mk, at, idxs, bs=128):
    """per-peak H + donor id + sample id (for probe fit/eval)."""
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

# ---- fit per-peak probe on TRAIN H ----
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
print(f"probe fit on {len(dtr)} train peaks")

@torch.no_grad()
def probe_softmax(H):
    return torch.softmax(clf(torch.from_numpy(H).to(DEV)),1).cpu().numpy()

def assemble_and_score(split):
    tk,mk,at = load(split)
    idx=np.arange(len(at))
    # true sets + noc from attr
    true_set={}; noc={}
    for gi in idx:
        a=at[gi]; d=np.unique(a[a>=0])
        if len(d)==0: continue
        true_set[gi]=set(int(x) for x in d); noc[gi]=len(d)
    keep=np.array(sorted(true_set));
    # probe per-peak -> per-donor soft vote -> top-k set
    H,don,sid = encode_peaks(tk,mk,at,keep)
    sm = probe_softmax(H)
    vote=defaultdict(lambda: np.zeros(45))
    for p in range(len(sid)): vote[int(sid[p])]+=sm[p]
    # model set-head scores
    ms = model_scores(tk,mk,keep)
    ms_map={int(g):ms[i] for i,g in enumerate(keep)}
    # oracle EM per NOC for both
    res={}
    for k in range(1,6):
        gis=[g for g in keep if noc[g]==k]
        if not gis: continue
        pe=[]; me=[]
        for g in gis:
            ts=true_set[g]
            ppred=set(int(x) for x in np.argsort(vote[g])[::-1][:k])
            mpred=set(int(x) for x in np.argsort(ms_map[g])[::-1][:k])
            pe.append(ppred==ts); me.append(mpred==ts)
        res[k]=(float(np.mean(pe)), float(np.mean(me)), len(gis))
    return res

print("\n=== SET-oracle: probe-assembled (soft-vote on H, top-NOC)  vs  model set-head (top-NOC) ===")
for split in ["test"]:
    res=assemble_and_score(split)
    print(f"\n  [{split}]  NOC :  probe_oracle   model_oracle   delta(probe-model)   n")
    for k in sorted(res):
        p,m,n=res[k]
        print(f"        N{k}  :    {p:.3f}          {m:.3f}          {p-m:+.3f}            {n}")
print("\n  delta>0 on N5 => a simple non-memorizing assembler on H beats the trained decoder")
print("                  => decoder learned-assembly overfit combos = the lever (CONFIRM).")
print("  delta<=0       => decoder already extracts H optimally; limit is compounding/info (REFUTE).")

"""
WHERE is the encoder weak, and WHERE is the decoder weak? (N5, faintest minor = the bottleneck rank r4)

For each N5 sample's faintest minor d (rank r4), classify by two binary readouts of the SAME H:
   probe_incl   = d in top-NOC of the independent linear soft-vote on H   (info readable in H?)
   decoder_incl = d in top-NOC of the model's set-head sigmoid scores     (does the trained decoder use it?)

Cross-tab gives 4 cells:
   probe1/dec1  both get it
   probe1/dec0  H HAS it, decoder DROPS it      -> pure DECODER waste
   probe0/dec1  decoder gets what probe misses   (rare)
   probe0/dec0  NEITHER -> H does not carry it    -> ENCODER limit (info absent or unreadable)

Then for the ENCODER question, split the faint minor by how much RAW signal it actually has:
   n_peaks  = #peaks attributed to d (its detected alleles)   -> low = physical dropout (info absent)
   hsum     = sum of its peak heights (template proxy)
If 'both-miss' minors have far fewer peaks/lower height than 'kept' minors -> encoder loss is
   dropout/info-absent, NOT washing. If they have similar raw signal but still unread -> encoder
   discards present signal (entanglement/washing).
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc8_v2_vicreg_inv_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
dg=dgm=None
if cfg.get("geno_query"):
    dg=torch.from_numpy(np.load(DATA/"donor_geno.npy").astype(np.float32)); dgm=torch.from_numpy(np.load(DATA/"donor_geno_mask.npy"))
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),geno_query=cfg.get("geno_query",False),
    donor_geno=dg,donor_geno_mask=dgm,vib=cfg.get("vib",False),mass_pool=cfg.get("mass_pool",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()
print(f"loaded {RUN.name}")

def load(s):
    return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
            np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))

@torch.no_grad()
def encode_peaks(tk,mk,at,idxs,bs=128):
    HH=[];DON=[];SID=[]
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        _,H,_=model._encode_set(t,m); H=H.cpu().numpy()
        for j,gi in enumerate(sel):
            a=at[gi]; v=np.where(a>=0)[0]
            if len(v)==0: continue
            HH.append(H[j][v]); DON.append(a[v]); SID.append(np.full(len(v),gi))
    return np.concatenate(HH),np.concatenate(DON).astype(int),np.concatenate(SID).astype(int)

@torch.no_grad()
def model_scores(tk,mk,idxs,bs=256):
    out=np.zeros((len(idxs),45),np.float32)
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; o=model(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        out[s:s+len(sel)]=torch.sigmoid(o["logits_cls"]).cpu().numpy()
    return out

# probe
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0)
Htr,dtr,_=encode_peaks(tk_tr,mk_tr,at_tr,rng.choice(len(at_tr),size=6000,replace=False))
clf=nn.Linear(Htr.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
Xt=torch.from_numpy(Htr).to(DEV); yt=torch.from_numpy(dtr).long().to(DEV); lf=nn.CrossEntropyLoss()
for ep in range(60):
    perm=torch.randperm(len(yt),device=DEV)
    for s in range(0,len(yt),8192):
        b=perm[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
print("probe fit")

tk,mk,at=load("test")
# per-sample true donors + rank + raw signal (peak count, height sum) of each donor
samp={}
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: continue
    lh=tk[gi][:,2]
    info={}
    for d in np.unique(a[v]):
        pk=(a==d); info[int(d)]={"n":int(pk.sum()),"h":float(np.exp(lh[pk]).sum())}
    order=sorted(info,key=lambda d:-info[d]["h"]);
    for r,d in enumerate(order): info[d]["rank"]=r
    samp[gi]={"info":info,"noc":len(order)}
keep=np.array([g for g in samp if samp[g]["noc"]==5])

H,don,sid=encode_peaks(tk,mk,at,keep)
with torch.no_grad():
    sm=torch.softmax(clf(torch.from_numpy(H).to(DEV)),1).cpu().numpy()
vote=defaultdict(lambda:np.zeros(45))
for p in range(len(sid)): vote[int(sid[p])]+=sm[p]
ms=model_scores(tk,mk,keep); ms_map={int(g):ms[i] for i,g in enumerate(keep)}

# faint minor (rank 4) per sample; classify probe/decoder inclusion (top-5 oracle)
cells=defaultdict(list)  # (probe_incl,dec_incl) -> list of (g,d)
for g in keep:
    k=5; ptop=set(int(x) for x in np.argsort(vote[g])[::-1][:k]); dtop=set(int(x) for x in np.argsort(ms_map[g])[::-1][:k])
    d=[dd for dd in samp[g]["info"] if samp[g]["info"][dd]["rank"]==4][0]
    cells[(d in ptop, d in dtop)].append((g,d))

print("\n=== faint-minor (rank r4) inclusion cross-tab, N5 (n={}) ===".format(len(keep)))
print("   probe_H \\ decoder      dec=KEEP   dec=DROP")
for pi in (True,False):
    row=[]
    for di in (True,False): row.append(len(cells[(pi,di)]))
    print(f"   probe={'KEEP' if pi else 'DROP'}            {row[0]:6d}     {row[1]:6d}")
pure_dec = len(cells[(True,False)]); both_miss=len(cells[(False,False)])
print(f"\n   pure DECODER waste (H has it, decoder drops): {pure_dec}")
print(f"   ENCODER limit (neither reads it)            : {both_miss}")

# raw signal of the faint minor by category: kept (both keep) vs both-miss vs pure-decoder-miss
def stats(lst):
    if not lst: return (0,float('nan'),float('nan'))
    n=[samp[g]['info'][d]['n'] for g,d in lst]; h=[samp[g]['info'][d]['h'] for g,d in lst]
    return (len(lst),float(np.mean(n)),float(np.median(h)))
print("\n=== RAW signal of the faint minor by category (encoder question) ===")
print("   category                 n    mean#peaks   median heightsum")
for lab,key in [("both KEEP        ",(True,True)),("pure DECODER-miss",(True,False)),("both MISS (enc) ",(False,False))]:
    n,mp,mh=stats(cells[key]); print(f"   {lab}  {n:4d}    {mp:7.2f}      {mh:10.1f}")
print("\n   if both-MISS has far fewer peaks/lower height than KEEP -> encoder loss = physical dropout (info absent).")
print("   if pure-DECODER-miss has peaks ~= KEEP -> decoder drops a perfectly readable minor (pure decoder fault).")

# decoder vs probe per-rank inclusion (where decoder leaks across ranks)
print("\n=== per-rank inclusion: probe(H) vs decoder, N5 ===")
pr=defaultdict(list); dr=defaultdict(list)
for g in keep:
    ptop=set(int(x) for x in np.argsort(vote[g])[::-1][:5]); dtop=set(int(x) for x in np.argsort(ms_map[g])[::-1][:5])
    for d in samp[g]["info"]:
        r=samp[g]["info"][d]["rank"]; pr[r].append(d in ptop); dr[r].append(d in dtop)
print("   rank :   probe   decoder   gap")
for r in range(5):
    print(f"   r{r}   :   {np.mean(pr[r]):.3f}   {np.mean(dr[r]):.3f}   {np.mean(pr[r])-np.mean(dr[r]):+.3f}")

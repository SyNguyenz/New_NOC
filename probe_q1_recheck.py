"""
Q1 RE-CHECK: are the 'both-miss' faint minors INFO-ABSENT (real dropout) or INFO-PRESENT but unread?
measure_noc5_ceiling says N5 = 100% RANKABLE (every contributor has a PRIVATE allele present).
So tag each N5 faint minor (rank r4) with RANKABLE/DROPOUT/MASKED using raw genotypes, and report
that tag WITHIN each cross-tab cell (both-keep / decoder-only-miss / both-miss).

If both-miss are mostly RANKABLE -> info IS present, the failure is MODEL-limited even there
(my earlier 'physical info floor' claim was wrong).
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN, LOCUS_TO_IDX  # reuse authoritative loader

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc8_v2_vicreg_inv_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
geno=load_raw_genotypes()

cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
    vib=cfg.get("vib",False),mass_pool=cfg.get("mass_pool",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()
print(f"loaded {RUN.name}")

def load(s):
    return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
            np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))

def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")

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
samp={}
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: continue
    lh=tk[gi][:,2]; info={}
    for d in np.unique(a[v]):
        pk=(a==d); info[int(d)]={"h":float(np.exp(lh[pk]).sum())}
    order=sorted(info,key=lambda d:-info[d]["h"])
    for r,d in enumerate(order): info[d]["rank"]=r
    # observed alleles
    obs=set((int(tk[gi][j,0]),akey(tk[gi][j,1])) for j in v)
    samp[gi]={"info":info,"noc":len(order),"obs":obs}
keep=np.array([g for g in samp if samp[g]["noc"]==5])

H,don,sid=encode_peaks(tk,mk,at,keep)
with torch.no_grad():
    sm=torch.softmax(clf(torch.from_numpy(H).to(DEV)),1).cpu().numpy()
vote=defaultdict(lambda:np.zeros(45))
for p in range(len(sid)): vote[int(sid[p])]+=sm[p]
ms=model_scores(tk,mk,keep); ms_map={int(g):ms[i] for i,g in enumerate(keep)}

def geno_cat(g,d):
    """RANKABLE / DROPOUT / MASKED for true donor d (class idx) in sample g."""
    others=[KNOWN[o] for o in samp[g]["info"] if o!=d]
    gX=geno.get(KNOWN[d],{});
    if not gX: return "NO_GENO"
    private=set()
    for L,all in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in all:
            if a not in oh: private.add((L,a))
    if not private: return "MASKED"
    return "RANKABLE" if (private & samp[g]["obs"]) else "DROPOUT"

cells=defaultdict(list)
for g in keep:
    ptop=set(int(x) for x in np.argsort(vote[g])[::-1][:5]); dtop=set(int(x) for x in np.argsort(ms_map[g])[::-1][:5])
    d=[dd for dd in samp[g]["info"] if samp[g]["info"][dd]["rank"]==4][0]
    cells[(d in ptop, d in dtop)].append((g,d))

# sanity: overall N5 faint-minor RANKABLE rate (should be ~100% per ceiling)
allfm=[(g,[dd for dd in samp[g]["info"] if samp[g]["info"][dd]["rank"]==4][0]) for g in keep]
cnt=defaultdict(int)
for g,d in allfm: cnt[geno_cat(g,d)]+=1
print(f"\nSANITY (all {len(allfm)} N5 faint minors): "+", ".join(f"{k}={v}({100*v/len(allfm):.0f}%)" for k,v in cnt.items()))

print("\n=== genotype category of the faint minor WITHIN each cross-tab cell ===")
print("   cell                       n    RANKABLE   DROPOUT   MASKED")
for lab,key in [("both KEEP        ",(True,True)),("pure DECODER-miss",(True,False)),
                ("both MISS (enc?) ",(False,False))]:
    lst=cells[key]; c=defaultdict(int)
    for g,d in lst: c[geno_cat(g,d)]+=1
    n=len(lst)
    f=lambda k: f"{c[k]:3d}({100*c[k]/n:3.0f}%)" if n else "  -"
    print(f"   {lab}  {n:4d}   {f('RANKABLE')}  {f('DROPOUT')}  {f('MASKED')}")
print("\n   both-MISS mostly RANKABLE => info IS present even there => MODEL-limited, not a physical floor.")

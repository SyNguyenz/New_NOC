"""
CAUSAL no-train test of the per-peak HEIGHT-DOMINANCE hypothesis for the encoder's N5 isolation miss.

Mechanism claim: a faint contributor's PRIVATE-allele peak (<5% height) is absorbed into a taller donor
during ISAB context-mixing BECAUSE it is faint. Test: hold IDENTITY (locus+allele) fixed, raise only the
HEIGHT features of the faint contributor's peaks to the sample's MAX (major-like), re-encode the whole set,
re-read the SAME private peaks with the fixed clean probe. Restricted to the 5 contributors.

  if absorbed faint peaks now point ->self  => height-dominance is the CAUSE (the lever target). CONFIRM.
  if isolation unchanged                     => encoder ignores input height value; mechanism is elsewhere.

Controls:
  (ctrl-SR)    raise a NON-height feature (SR) instead -> expect ~no change (specificity).
  (ctrl-major) raise height of an already-isolated rank-0 major -> expect stays isolated (no breakage).

tokens8 cols: 0=locus 1=allele 2=log_h 3=Hb 4=SR 5=rank_inv 6=n/10 7=glob_rel
HEIGHT cols (monotone with abundance) = log_h(2), rank_inv(5), glob_rel(7).
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc2_2d_sparse_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
HCOLS=[2,5,7]; SRCOL=4
geno=load_raw_genotypes()
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
    vib=cfg.get("vib",False),mass_pool=cfg.get("mass_pool",False),
    attn_sink=int(cfg.get("attn_sink",0) or 0),donor_recon=cfg.get("donor_recon",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()
print(f"loaded {RUN.name}  HEIGHT cols={HCOLS}")

def load(s):
    return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
            np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")

@torch.no_grad()
def encode_one(tkrow, mkrow):
    t=torch.from_numpy(tkrow[None]).to(DEV); m=torch.from_numpy(mkrow[None]).to(DEV)
    _,H,_=model._encode_set(t,m); return H[0].cpu().numpy()

# clean probe on train H
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0)
HH=[];DD=[]
with torch.no_grad():
    for s in range(0,6000,128):
        sel=rng.choice(len(at_tr),size=min(128,6000-s),replace=False)
        t=torch.from_numpy(tk_tr[sel]).to(DEV); m=torch.from_numpy(mk_tr[sel]).to(DEV)
        _,H,_=model._encode_set(t,m); H=H.cpu().numpy()
        for j,gi in enumerate(sel):
            a=at_tr[gi]; v=np.where(a>=0)[0]; HH.append(H[j][v]); DD.append(a[v])
Htr=np.concatenate(HH).astype(np.float32); dtr=np.concatenate(DD).astype(int)
clf=nn.Linear(Htr.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
Xt=torch.from_numpy(Htr).to(DEV); yt=torch.from_numpy(dtr).long().to(DEV); lf=nn.CrossEntropyLoss()
for ep in range(60):
    perm=torch.randperm(len(yt),device=DEV)
    for s in range(0,len(yt),8192):
        b=perm[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
W=clf.weight.detach().cpu().numpy(); B=clf.bias.detach().cpu().numpy()
def pred_donor(hrow, contribs):
    z=hrow@W.T+B; return contribs[int(np.argmax([z[c] for c in contribs]))]
print("probe fit\n")

tk,mk,at=load("test")
def setup(gi):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: return None
    lh=tk[gi][:,2]; info={}
    for d in np.unique(a[v]): info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
    order=sorted(info,key=lambda d:-info[d]["h"])
    for r,d in enumerate(order): info[d]["rank"]=r
    if len(order)!=5: return None
    return info,v
def private_of(gi,d,info):
    others=[KNOWN[o] for o in info if o!=d]; gX=geno.get(KNOWN[d],{}); priv=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: priv.add((L,a))
    return priv

def run(target_rank, cols, label):
    base_self=base_n=iv_self=0; recov=recov_n=0
    for gi in range(len(at)):
        su=setup(gi)
        if su is None: continue
        info,v=su; contribs=list(info.keys())
        d=[c for c in info if info[c]["rank"]==target_rank][0]
        priv=private_of(gi,d,info)
        pk=[j for j in v if int(at[gi][j])==d and (int(tk[gi][j,0]),akey(tk[gi][j,1])) in priv]
        if not pk: continue
        H0=encode_one(tk[gi],mk[gi])
        # intervene: raise `cols` of ALL of d's peaks to sample-max over valid peaks
        tki=tk[gi].copy(); dpk=[j for j in v if int(at[gi][j])==d]
        for c in cols: tki[dpk,c]=tk[gi][v,c].max()
        H1=encode_one(tki,mk[gi])
        for j in pk:
            s0=(pred_donor(H0[j],contribs)==d); s1=(pred_donor(H1[j],contribs)==d)
            base_self+=s0; iv_self+=s1; base_n+=1
            if not s0:
                recov_n+=1; recov+=s1
    print(f"  {label:34s} base->self {base_self/base_n:.3f}  intervened->self {iv_self/base_n:.3f}"
          f"  | recovery of absorbed: {recov}/{recov_n}={recov/max(recov_n,1):.2f}  (n_peaks={base_n})")

def run_suppress(target_rank, cols, label):
    """gel the COMPETITION: lower `cols` of all STRONGER peaks to sample-min, measure target isolation."""
    base_self=base_n=iv_self=recov=recov_n=0
    for gi in range(len(at)):
        su=setup(gi)
        if su is None: continue
        info,v=su; contribs=list(info.keys())
        d=[c for c in info if info[c]["rank"]==target_rank][0]
        priv=private_of(gi,d,info)
        pk=[j for j in v if int(at[gi][j])==d and (int(tk[gi][j,0]),akey(tk[gi][j,1])) in priv]
        if not pk: continue
        H0=encode_one(tk[gi],mk[gi])
        tki=tk[gi].copy()
        strong=[j for j in v if info[int(at[gi][j])]["rank"]<target_rank]
        for c in cols: tki[strong,c]=tk[gi][v,c].min()
        H1=encode_one(tki,mk[gi])
        for j in pk:
            s0=(pred_donor(H0[j],contribs)==d); s1=(pred_donor(H1[j],contribs)==d)
            base_self+=s0; iv_self+=s1; base_n+=1
            if not s0: recov_n+=1; recov+=s1
    print(f"  {label:34s} base->self {base_self/base_n:.3f}  intervened->self {iv_self/base_n:.3f}"
          f"  | recovery of absorbed: {recov}/{recov_n}={recov/max(recov_n,1):.2f}  (n_peaks={base_n})")

print("=== faintest contributor (rank 4) private peaks ===")
run(4, HCOLS, "HEIGHT-boost all3 (log_h,rank,grel)")
run(4, [2], "HEIGHT-boost log_h ONLY (clean)")
run(4, [SRCOL], "ctrl: SR-boost (non-height)")
run_suppress(4, HCOLS, "SUPPRESS majors' height (rank<4)")
print("\n=== control: a major (rank 0) — must stay isolated ===")
run(0, HCOLS, "HEIGHT-boost on rank-0 major")

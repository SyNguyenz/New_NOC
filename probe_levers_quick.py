"""
Quick NO-TRAIN probes of the candidate levers (V2). Skips train-only ones (PIT/IFM/slot/exact-loss).
  L1 INTER-LOCUS genotype-match prior (domain lever + geno_query proxy): ensemble neural soft-vote with a
     closed-set full-genotype match score (uses ALL loci jointly). Does it recover both-miss/decoder-miss minors?
  L2 ENCODER dominance (softmax-1 / slot-competition / mass-balance proxy): RE-ENCODE with compressed peak
     heights (shrink major dominance). Does the faint minor's private-peak ->minor rate rise vs ->major?
  L3 DECODER rebalance (balanced-loss proxy, label-free): per-class z-score the decoder logits, re-rank.
     If N5 oracle rises -> under-read is partly calibration; if flat -> needs retraining (rank problem).
Baselines printed alongside.
"""
import sys, io, json, numpy as np, torch, torch.nn as nn
try: sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
except Exception: pass
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

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
def encode_full(tkarr,mk,idxs,bs=128):
    out={}
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tkarr[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        _,H,_=model._encode_set(t,m); H=H.cpu().numpy()
        for j,gi in enumerate(sel): out[int(gi)]=H[j]
    return out
@torch.no_grad()
def mscores(tkarr,mk,idxs,bs=256):
    out=np.zeros((len(idxs),45),np.float32)
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; o=model(torch.from_numpy(tkarr[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        out[s:s+len(sel)]=torch.sigmoid(o["logits_cls"]).cpu().numpy()
    return out

# probe on train H
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0)
Hm=encode_full(tk_tr,mk_tr,rng.choice(len(at_tr),size=6000,replace=False))
HH=[];DD=[]
for gi,H in Hm.items():
    a=at_tr[gi]; v=np.where(a>=0)[0]; HH.append(H[v]); DD.append(a[v])
clf=nn.Linear(128,45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
Xt=torch.from_numpy(np.concatenate(HH).astype(np.float32)).to(DEV); yt=torch.from_numpy(np.concatenate(DD).astype(int)).long().to(DEV)
lf=nn.CrossEntropyLoss()
for ep in range(60):
    perm=torch.randperm(len(yt),device=DEV)
    for s in range(0,len(yt),8192):
        b=perm[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
W=clf.weight.detach().cpu().numpy(); B=clf.bias.detach().cpu().numpy()
def smrows(X):
    Z=X@W.T+B; Z-=Z.max(1,keepdims=True); E=np.exp(Z); return E/E.sum(1,keepdims=True)
print("probe fit\n")

# test setup
tk,mk,at=load("test")
samp={}
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: continue
    lh=tk[gi][:,2]; info={}
    for d in np.unique(a[v]): info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
    order=sorted(info,key=lambda d:-info[d]["h"])
    for r,d in enumerate(order): info[d]["rank"]=r
    obs=set((int(tk[gi][j,0]),akey(tk[gi][j,1])) for j in v)
    samp[gi]={"info":info,"noc":len(order),"obs":obs}
keep=np.array([g for g in samp if samp[g]["noc"]==5])
Hte=encode_full(tk,mk,keep)
ms=mscores(tk,mk,keep); ms_map={int(g):ms[i] for i,g in enumerate(keep)}

# neural vote
def vote_of(g,Hmap):
    a=at[g]; v=np.where(a>=0)[0]; return smrows(Hmap[g][v]).sum(0)
nvote={int(g):vote_of(g,Hte) for g in keep}

# genotype-match score (inter-locus): fraction of donor d's genotype alleles present in obs
geno_pairs={}  # class idx -> set of (locus, allele_key)
for d in range(45):
    g=geno.get(KNOWN[d],{}); geno_pairs[d]=set((L,a) for L,al in g.items() for a in al)
def genomatch_vec(g):
    obs=samp[g]["obs"]; out=np.zeros(45)
    for d in range(45):
        gp=geno_pairs[d]
        if gp: out[d]=len(gp & obs)/len(gp)
    return out
gmatch={int(g):genomatch_vec(g) for g in keep}

def n5_oracle(scoref):
    em=[]
    for g in keep:
        true=set(samp[g]["info"].keys()); pred=set(np.argsort(scoref(g))[::-1][:5].tolist())
        em.append(pred==true)
    return float(np.mean(em))

def faint_incl(scoref):
    inc=[]
    for g in keep:
        d=[dd for dd in samp[g]["info"] if samp[g]["info"][dd]["rank"]==4][0]
        inc.append(d in set(np.argsort(scoref(g))[::-1][:5].tolist()))
    return float(np.mean(inc))

# cells for recovery tracking (neural vote vs decoder)
cells=defaultdict(list)
for g in keep:
    g=int(g); ptop=set(np.argsort(nvote[g])[::-1][:5].tolist()); dtop=set(np.argsort(ms_map[g])[::-1][:5].tolist())
    d=[dd for dd in samp[g]["info"] if samp[g]["info"][dd]["rank"]==4][0]
    cells[(d in ptop,d in dtop)].append(g)

print("=== L1: INTER-LOCUS genotype-match prior (ensemble with neural soft-vote) ===")
print(f"   neural-vote only        : N5 oracle {n5_oracle(lambda g:nvote[g]):.3f}  faint-incl {faint_incl(lambda g:nvote[g]):.3f}")
print(f"   genomatch only          : N5 oracle {n5_oracle(lambda g:gmatch[g]):.3f}  faint-incl {faint_incl(lambda g:gmatch[g]):.3f}")
for lam in [0.5,1.0,2.0,4.0,8.0]:
    f=lambda g,l=lam: nvote[g]+l*gmatch[g]
    print(f"   neural + {lam:>3}*genomatch    : N5 oracle {n5_oracle(f):.3f}  faint-incl {faint_incl(f):.3f}")
# recovery of the hard cells by best ensemble
best=lambda g: nvote[g]+4.0*gmatch[g]
for lab,key in [("both MISS",(False,False)),("decoder-miss",(True,False))]:
    gs=cells[key]; rec=0
    for g in gs:
        d=[dd for dd in samp[g]["info"] if samp[g]["info"][dd]["rank"]==4][0]
        if d in set(np.argsort(best(g))[::-1][:5].tolist()): rec+=1
    print(f"     ensemble(lam=4) recovers faint minor in {rec}/{len(gs)} {lab}")

print("\n=== L2: ENCODER dominance — RE-ENCODE with compressed peak heights ===")
# private-peak ->minor baseline + compressed
def private_of(g,d):
    others=[KNOWN[o] for o in samp[g]["info"] if o!=d]; gX=geno.get(KNOWN[d],{}); priv=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: priv.add((L,a))
    return priv
def isolate_rate(Hmap):
    tom=tot=0
    for g in keep:
        g=int(g); d=[dd for dd in samp[g]["info"] if samp[g]["info"][dd]["rank"]==4][0]
        contribs=list(samp[g]["info"].keys()); priv=private_of(g,d); a=at[g]; v=np.where(a>=0)[0]
        pk=[j for j in v if int(a[j])==d and (int(tk[g][j,0]),akey(tk[g][j,1])) in priv]
        if not pk: continue
        sm=smrows(Hmap[g][pk])
        for row in sm:
            sc={c:row[c] for c in contribs}; tot+=1; tom+= (max(sc,key=sc.get)==d)
    return tom/tot if tot else float('nan')
base_iso=isolate_rate(Hte)
print(f"   baseline private-peak ->minor: {base_iso:.3f}")
for alpha in [0.7,0.5,0.3]:
    tk2=tk.copy()
    for g in keep:
        g=int(g); v=np.where(at[g]>=0)[0]; lh=tk2[g][v,2]; mu=lh.mean()
        tk2[g][v,2]=mu+alpha*(lh-mu)        # compress dynamic range (shrink major dominance)
    H2=encode_full(tk2,mk,keep)
    print(f"   compress α={alpha}: ->minor {isolate_rate(H2):.3f}   soft-vote N5 oracle {n5_oracle(lambda g:vote_of(g,H2)):.3f}")

print("\n=== L3: DECODER rebalance (label-free per-class z-score of logits) ===")
M=np.stack([ms_map[int(g)] for g in keep]); mu=M.mean(0); sd=M.std(0)+1e-6
print(f"   decoder raw           : N5 oracle {n5_oracle(lambda g:ms_map[g]):.3f}  faint-incl {faint_incl(lambda g:ms_map[g]):.3f}")
print(f"   decoder per-class z   : N5 oracle {n5_oracle(lambda g:(ms_map[g]-mu)/sd):.3f}  faint-incl {faint_incl(lambda g:(ms_map[g]-mu)/sd):.3f}")
print(f"   decoder minus-classmean: N5 oracle {n5_oracle(lambda g:ms_map[g]-mu):.3f}  faint-incl {faint_incl(lambda g:ms_map[g]-mu):.3f}")

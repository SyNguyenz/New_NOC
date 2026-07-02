"""
Is the encoder failure on the hard N5 faint minors really 'failed to ISOLATE', or is it just the
soft-vote pooling diluting a signal the encoder DID isolate?

Decisive: look at the minor's OWN PRIVATE-allele peak (a peak whose (locus,allele) is unique to the
minor among the 5 contributors, and is present). Read the encoder H of THAT peak with the independent
probe, restricted to the 5 contributors. Where does it point?
   -> minor  : encoder isolated it (info in H at that peak); a miss is pooling/decoder, NOT encoder.
   -> a major: encoder absorbed the minor's private peak into a dominant donor = ISOLATION FAILURE.

Report per cross-tab cell (both-keep / decoder-only-miss / both-miss).
"""
import sys, json, numpy as np, torch, torch.nn as nn
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
def encode_full(tk,mk,idxs,bs=128):
    """return list aligned per sample: H[peaks], valid idxs."""
    out={}
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        _,H,_=model._encode_set(t,m); H=H.cpu().numpy()
        for j,gi in enumerate(sel): out[int(gi)]=H[j]
    return out

# fit probe on train H (per-peak donor id)
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0)
HH=[];DD=[]
Hmap=encode_full(tk_tr,mk_tr,rng.choice(len(at_tr),size=6000,replace=False))
for gi,H in Hmap.items():
    a=at_tr[gi]; v=np.where(a>=0)[0]
    HH.append(H[v]); DD.append(a[v])
Htr=np.concatenate(HH).astype(np.float32); dtr=np.concatenate(DD).astype(int)
clf=nn.Linear(Htr.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
Xt=torch.from_numpy(Htr).to(DEV); yt=torch.from_numpy(dtr).long().to(DEV); lf=nn.CrossEntropyLoss()
for ep in range(60):
    perm=torch.randperm(len(yt),device=DEV)
    for s in range(0,len(yt),8192):
        b=perm[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
W=clf.weight.detach().cpu().numpy(); B=clf.bias.detach().cpu().numpy()
print("probe fit")

# test setup
tk,mk,at=load("test")
samp={}
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: continue
    lh=tk[gi][:,2]; info={}
    for d in np.unique(a[v]):
        info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
    order=sorted(info,key=lambda d:-info[d]["h"])
    for r,d in enumerate(order): info[d]["rank"]=r
    obs=set((int(tk[gi][j,0]),akey(tk[gi][j,1])) for j in v)
    samp[gi]={"info":info,"noc":len(order),"obs":obs}
keep=[g for g in samp if samp[g]["noc"]==5]
Hmap_te=encode_full(tk,mk,np.array(keep))

# pooled vote + model scores to rebuild the 3 cells
def softmax_rows(X):
    Z=X@W.T+B; Z-=Z.max(1,keepdims=True); E=np.exp(Z); return E/E.sum(1,keepdims=True)
@torch.no_grad()
def model_scores(idxs,bs=256):
    out=np.zeros((len(idxs),45),np.float32)
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; o=model(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        out[s:s+len(sel)]=torch.sigmoid(o["logits_cls"]).cpu().numpy()
    return out
ms=model_scores(np.array(keep)); ms_map={g:ms[i] for i,g in enumerate(keep)}

def private_of(g,d):
    others=[KNOWN[o] for o in samp[g]["info"] if o!=d]; gX=geno.get(KNOWN[d],{}); priv=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: priv.add((L,a))
    return priv

cells=defaultdict(list)
for g in keep:
    a=at[g]; v=np.where(a>=0)[0]
    sm=softmax_rows(Hmap_te[g][v]); vote=np.zeros(45)
    for r in sm: vote+=r
    ptop=set(np.argsort(vote)[::-1][:5].tolist()); dtop=set(np.argsort(ms_map[g])[::-1][:5].tolist())
    d=[dd for dd in samp[g]["info"] if samp[g]["info"][dd]["rank"]==4][0]
    cells[(d in ptop, d in dtop)].append(g)

print("\n=== private-allele PEAK of the faint minor: where does its encoder-H point (among the 5 contributors)? ===")
print("   cell                  n_minors  n_privpeaks  ->minor  ->major(r<4)  ->other-minor")
for lab,key in [("both KEEP        ",(True,True)),("pure DECODER-miss",(True,False)),("both MISS        ",(False,False))]:
    gs=cells[key]; tom=tomaj=toother=npk=nmin=0
    for g in gs:
        d=[dd for dd in samp[g]["info"] if samp[g]["info"][dd]["rank"]==4][0]
        contribs=list(samp[g]["info"].keys()); priv=private_of(g,d)
        a=at[g]; v=np.where(a>=0)[0]
        pk=[j for j in v if int(a[j])==d and (int(tk[g][j,0]),akey(tk[g][j,1])) in priv]
        if not pk: continue
        nmin+=1
        sm=softmax_rows(Hmap_te[g][pk])           # (n_privpeaks, 45)
        for row in sm:
            sc={c:row[c] for c in contribs}; pred=max(sc,key=sc.get); npk+=1
            if pred==d: tom+=1
            elif samp[g]["info"][pred]["rank"]<4: tomaj+=1
            else: toother+=1
    if npk:
        print(f"   {lab}  {nmin:7d}  {npk:11d}   {tom/npk:.2f}      {tomaj/npk:.2f}        {toother/npk:.2f}")
print("\n   ->minor HIGH on both-MISS => encoder isolated the private peak; miss is pooling/decoder (NOT encoder).")
print("   ->major HIGH on both-MISS => encoder absorbed it into a dominant donor = ISOLATION FAILURE confirmed.")

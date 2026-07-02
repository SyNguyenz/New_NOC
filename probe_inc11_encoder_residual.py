"""
Decompose the RESIDUAL encoder miss (N5 H-vote ceiling < 1.0) for each arm.
For every N5 sample the encoder's own linear H-vote gets WRONG, look at each missed true donor and ask
WHY the encoder lost it:

  (A) ISOLATION-FAILURE residual : the donor HAS private peaks present, but the encoder-H of those peaks
                                   routes to a MAJOR (rank<donor)  -> need MORE isolation / de-competition.
  (B) UNDER-DETERMINATION         : the donor's private peaks route to ITSELF (encoder isolated them), yet the
                                   donor still loses the set-vote -> few private alleles, shared-allele mass
                                   goes to others. Needs the encoder to CREDIT shared alleles (genotype/
                                   co-occurrence consistency), not more isolation.
  (C) NO private peak present     : physical-ish floor for THIS readout (rare; F33 says still rankable globally).

Reported per arm so we can see which lever the remaining .16 encoder hole demands.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w")
RUNS=[Path(p) for p in sys.argv[1:]] or [
    Path("results/inc2_2d_sparse_seed42"), Path("results/inc11_nc_mab0_seed42"), Path("results/inc11_nc_both_seed42")]
DEV="cuda" if torch.cuda.is_available() else "cpu"; geno=load_raw_genotypes()

def build(run):
    cfg=json.load(open(run/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
    m=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
        dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
        n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
        n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
        aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),vib=cfg.get("vib",False),
        mass_pool=cfg.get("mass_pool",False),nc_attn=cfg.get("nc_attn","none"),nc_learnable_bias=cfg.get("nc_learnable_bias",False)).to(DEV)
    m.load_state_dict(torch.load(run/"best_model.pt",map_location=DEV,weights_only=True),strict=False); m.eval(); return m,n_tok
def load(s,n_tok):
    return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
            np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")
@torch.no_grad()
def encode(model,tk,mk,idxs,bs=128):
    out={}
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; _,H,_=model._encode_set(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        for j,gi in enumerate(sel): out[int(gi)]=H[j].cpu().numpy()
    return out
def private_of(g,d,info,tk):
    others=[KNOWN[o] for o in info if o!=d]; gX=geno.get(KNOWN[d],{}); priv=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: priv.add((L,a))
    return priv

print(f"device={DEV}")
for run in RUNS:
    model,n_tok=build(run); tk_tr,mk_tr,at_tr=load("train",n_tok); tk,mk,at=load("test",n_tok)
    rng=np.random.default_rng(0)
    Hmap=encode(model,tk_tr,mk_tr,rng.choice(len(at_tr),size=5000,replace=False))
    HH=[];DD=[]
    for gi,H in Hmap.items():
        a=at_tr[gi]; v=np.where(a>=0)[0]; HH.append(H[v]); DD.append(a[v])
    Htr=np.concatenate(HH).astype(np.float32); dtr=np.concatenate(DD).astype(int)
    clf=nn.Linear(Htr.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
    Xt=torch.from_numpy(Htr).to(DEV); yt=torch.from_numpy(dtr).long().to(DEV); lf=nn.CrossEntropyLoss()
    for ep in range(60):
        perm=torch.randperm(len(yt),device=DEV)
        for s in range(0,len(yt),8192):
            b=perm[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
    W=clf.weight.detach().cpu().numpy(); B=clf.bias.detach().cpu().numpy()
    def softmax_rows(X):
        Z=X@W.T+B; Z-=Z.max(1,keepdims=True); E=np.exp(Z); return E/E.sum(1,keepdims=True)

    samp={}
    for gi in range(len(at)):
        a=at[gi]; v=np.where(a>=0)[0]
        if len(v)==0: continue
        lh=tk[gi][:,2]; info={}
        for d in np.unique(a[v]): info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
        order=sorted(info,key=lambda d:-info[d]["h"])
        for r,d in enumerate(order): info[d]["rank"]=r
        samp[gi]={"info":info,"noc":len(order),"true":set(int(x) for x in np.unique(a[v]))}
    n5=[g for g in samp if samp[g]["noc"]==5]; Hm=encode(model,tk,mk,np.array(n5))

    A=B_=C=0; nmiss=0; npriv_missed=[]; vote_share_missed=[]
    for g in n5:
        a=at[g]; v=np.where(a>=0)[0]; vote=softmax_rows(Hm[g][v]).sum(0)
        top5=set(np.argsort(vote)[::-1][:5].tolist())
        missed=samp[g]["true"]-top5
        if not missed: continue
        for d in missed:
            nmiss+=1; info=samp[g]["info"]; contribs=list(info.keys())
            priv=private_of(g,d,info,tk)
            pk=[j for j in v if int(a[j])==d and (int(tk[g][j,0]),akey(tk[g][j,1])) in priv]
            npriv_missed.append(len(pk)); vote_share_missed.append(vote[d]/vote[list(top5)].min() if vote[list(top5)].min()>0 else 0)
            if len(pk)==0: C+=1; continue
            sm=softmax_rows(Hm[g][pk]); self_n=major_n=0
            for row in sm:
                sc={c:row[c] for c in contribs}; pred=max(sc,key=sc.get)
                if pred==d: self_n+=1
                elif info[pred]["rank"]<info[d]["rank"]: major_n+=1
            if major_n>=self_n: A+=1            # private peaks mostly absorbed into a major
            else: B_+=1                          # private peaks self-isolated, yet donor still lost the vote
    nca=json.load(open(run/'metrics.json'))['config'].get('nc_attn','none')
    print(f"\n=== {run.name} (nc_attn={nca}) — decompose N5 H-vote ENCODER misses ===")
    print(f"  missed true-donors: {nmiss}   (mean #private present = {np.mean(npriv_missed):.1f}, median {int(np.median(npriv_missed))})")
    if nmiss:
        print(f"   (A) isolation-failure (private peaks -> MAJOR)   : {A:4d}  {A/nmiss:.0%}   -> lever = more de-competition/isolation")
        print(f"   (B) under-determination (isolated but outvoted) : {B_:4d}  {B_/nmiss:.0%}   -> lever = credit SHARED alleles (geno/co-occur)")
        print(f"   (C) no private peak present (this readout)       : {C:4d}  {C/nmiss:.0%}")
        lo=sum(1 for x in npriv_missed if x<=2)
        print(f"   of missed donors, {lo}/{nmiss}={lo/nmiss:.0%} have <=2 private alleles present")

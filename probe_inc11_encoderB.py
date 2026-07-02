"""
Encoder-B alternatives that are COMBO-INVARIANT by construction (use each donor's OWN reference genotype +
the OBSERVED peaks; NO learned donor co-occurrence -> no combo-overfit -> OOD-safe by design).
Question: do any have SIGNAL, or are they swamped by decoy donors (the reason deployable ref_match failed)?

All scores deployed over ALL 45 donors. NO privileged ground-truth (peaks taken from the mask, not at>=0).

Schemes (score per donor d, then rank):
  COV      : fraction of d's reference alleles that are PRESENT in the profile
  COVh     : height-weighted coverage (present matches weighted by peak height)
  RARITY   : coverage but each matched allele weighted by panel-rarity 1/freq (discriminative alleles count more)
  RESID    : CONDITIONAL — remove alleles already explained by the model's confident donors (prob>0.5),
             score d by coverage of the RESIDUAL alleles only (combo-invariant peeling at genotype level)

For the faintest minor (rank-4) of each N5 sample we report:
  - median RANK of the true minor among 45 under the score ALONE (1=best)  -> raw separability
  - median #DECOYS (non-contributors) outranking it                        -> contamination
  - logit-add ceiling: rank by  model_logit + lam * zscore(score); N5 set-EM vs model, with N3/N4 guard
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w")
RUN=Path(sys.argv[1]) if len(sys.argv)>1 else Path("results/inc11_nc_mab0_seed42")
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

model,n_tok=build(RUN); tk,mk,at=load("test",n_tok)

# panel allele frequency for rarity weighting
gsize={}; freq={}
for d in range(45):
    gd=geno.get(KNOWN[d],{}); gsize[d]=sum(len(v) for v in gd.values())
    for L,al in gd.items():
        for a in al: freq[(L,a)]=freq.get((L,a),0)+1
def rar(L,a): return 1.0/freq.get((L,a),1)

@torch.no_grad()
def mscores(idxs,bs=256):
    out=np.zeros((len(idxs),45),np.float32)
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; o=model(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        out[s:s+len(sel)]=torch.sigmoid(o["logits_cls"]).cpu().numpy()
    return out

# build N5 samples
samp={}
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: continue
    lh=tk[gi][:,2]; info={}
    for d in np.unique(a[v]): info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
    order=sorted(info,key=lambda d:-info[d]["h"])
    for r,d in enumerate(order): info[d]["rank"]=r
    if len(order)!=5: continue
    samp[gi]={"info":info,"true":set(int(x) for x in np.unique(a[v]))}
ids=np.array(list(samp.keys())); MS=mscores(ids); MSm={g:MS[i] for i,g in enumerate(ids)}

def present_of(g):
    """observed (locus,allele-key) -> max height. DEPLOYABLE set = all mask-valid peaks (the EPG), INCLUDING
    unattributed drop-in/noise. Does NOT filter by ground-truth attribution (that was the retracted privilege)."""
    p={}
    valid=np.where(mk[g])[0]
    for j in valid:
        av=float(tk[g,j,1])
        if av<=0: continue            # -1/-2 are markers, not real alleles
        L=int(tk[g,j,0]); a=akey(av); h=float(np.exp(tk[g,j,2]))
        p[(L,a)]=max(p.get((L,a),0.0),h)
    return p

# diagnostic: how many valid peaks are UNATTRIBUTED (noise/drop-in) — the gap that makes it non-privileged
_gv=[int(mk[g].sum()) for g in list(samp.keys())[:500]]; _ga=[int((at[g]>=0).sum()) for g in list(samp.keys())[:500]]
print(f"  [diag] mean mask-valid peaks={np.mean(_gv):.1f}  vs attributed(at>=0)={np.mean(_ga):.1f}  -> ~{np.mean(_gv)-np.mean(_ga):.1f} unattributed/sample")

def scores_all(g, present, explained=None):
    """return dict scheme-> np.array(45)."""
    Lpresent=present
    cov=np.zeros(45); covh=np.zeros(45); rrt=np.zeros(45); resid=np.zeros(45)
    rsum=np.zeros(45)
    for d in range(45):
        gd=geno.get(KNOWN[d],{}); gz=max(gsize[d],1); rz=0.0; rmatch=0.0
        mc=0.0; mh=0.0; rc=0.0; rcz=0.0
        for L,al in gd.items():
            for a in al:
                w=rar(L,a); rz+=w
                if (L,a) in Lpresent:
                    mc+=1; mh+=Lpresent[(L,a)]; rmatch+=w
                    if explained is not None and (L,a) not in explained:
                        rc+=1
                if explained is not None and (L,a) not in (explained or set()):
                    rcz+=1
        cov[d]=mc/gz; covh[d]=mh; rrt[d]=rmatch/max(rz,1e-9)
        resid[d]=rc/max(rcz,1e-9) if explained is not None else 0.0
    return {"COV":cov,"COVh":covh,"RARITY":rrt,"RESID":resid}

def z(x):
    s=x.std(); return (x-x.mean())/s if s>1e-9 else x*0

# per-sample raw separability of the faint minor + collect for logit-add
faint=[]; rows={k:{"rank":[],"dec":[]} for k in ["COV","COVh","RARITY","RESID"]}
S_store={}
for g in ids:
    info=samp[g]["info"]; d=[dd for dd in info if info[dd]["rank"]==4][0]; faint.append((g,d))
    present=present_of(g)
    expl=set()
    for dd in range(45):
        if MSm[g][dd]>0.5:
            for L,al in geno.get(KNOWN[dd],{}).items():
                for a in al: expl.add((L,a))
    sc=scores_all(g,present,explained=expl); S_store[g]=sc
    contribs=samp[g]["true"]
    for k,arr in sc.items():
        order=np.argsort(arr)[::-1]; rank=int(np.where(order==d)[0][0])+1
        dec=sum(1 for x in order[:rank-1] if x not in contribs)
        rows[k]["rank"].append(rank); rows[k]["dec"].append(dec)

print(f"=== {RUN.name}: combo-invariant Encoder-B candidates on N5 faint minor (n={len(faint)}) ===")
print(f"  model-decode N5 faint-minor recall (top5) = {np.mean([(d in set(np.argsort(MSm[g])[::-1][:5])) for g,d in faint]):.3f}")
print(f"  {'scheme':8} | median minor-RANK/45 | median #DECOYS above | recall@5(score alone)")
for k in ["COV","COVh","RARITY","RESID"]:
    r=np.array(rows[k]["rank"]); dc=np.array(rows[k]["dec"])
    rec5=np.mean(r<=5)
    print(f"  {k:8} |        {np.median(r):4.0f}          |        {np.median(dc):4.0f}          |   {rec5:.3f}")

# logit-add ceiling: rank by model_logit + lam*z(score); report N5/N4/N3 set-EM
def setem_at(lam, key):
    out={5:[],4:[],3:[]}
    # need N3/N4 too: recompute scores for those NOCs is heavy; reuse model for guard via a quick pass
    return None
print("\n  logit-add ceiling (rank by model_logit + lam*z(score)); N5 set-EM vs model baseline:")
base_em=np.mean([set(np.argsort(MSm[g])[::-1][:5])==samp[g]["true"] for g in ids])
print(f"   model N5 set-EM = {base_em:.3f}")
eps=1e-6
for k in ["COV","RARITY","RESID"]:
    best=None
    for lam in [0.0,0.25,0.5,1.0,2.0,4.0]:
        ems=[]
        for g in ids:
            ml=np.log(MSm[g]+eps)-np.log(1-MSm[g]+eps)
            comb=ml+lam*z(S_store[g][k])
            ems.append(set(np.argsort(comb)[::-1][:5])==samp[g]["true"])
        em=np.mean(ems)
        if best is None or em>best[1]: best=(lam,em)
    print(f"   {k:8}: best lam={best[0]:.2f} -> N5 set-EM {best[1]:.3f}  (delta {best[1]-base_em:+.3f})")

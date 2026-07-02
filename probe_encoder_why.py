"""
WHY does the encoder absorb a faint minor's OWN PRIVATE-allele peak into a MAJOR donor (the ~.20 wall)?
Height-dominance is refuted (probe1/2/3). A private allele is unique to the minor among the 5 contributors,
so the only ways its peak can be mis-assigned are STRUCTURAL:
  (H-stutter)   the private allele sits at a STUTTER / adjacent position of a major's allele at the same
                locus (a_minor = a_major - 1 = the n-1 back-stutter slot) -> the encoder reads it as the
                major's stutter artifact, not a separate donor's real allele.
  (H-colocate) a major contributor has a present (usually tall) peak at the SAME locus -> local
                "explaining-away": the faint peak is folded into the dominant local explanation.
  (H-height)   the refuted control: the peak is just faint.

Decisive: split each faint minor's PRESENT private peaks into ISOLATED (clean encoder probe argmax over the
5 contributors == minor) vs ABSORBED (argmax == a major). Compare the two groups on the structural axes,
with height shown as a control. Then the causal direction: of ABSORBED peaks, how often is the absorbing
donor the one whose allele is stutter/adjacent to the peak?

  ABSORBED >> ISOLATED on stutter/colocate, height ~equal  => structural confusion is the encoder mechanism.
  groups equal on structure                                => not positional; mechanism is global (carrier).

eval-only, frozen ckpt. tokens8 cols: 0=locus 1=allele 2=log_h ... 7=glob_rel.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc2_2d_sparse_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
geno=load_raw_genotypes()
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","per_donor"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
    attn_sink=int(cfg.get("attn_sink",0) or 0),donor_recon=cfg.get("donor_recon",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()
print(f"loaded {RUN.name}")

def load(s):
    return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
            np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")
def anum(k):
    try:
        f=float(k)
        return f if f>=0 else None     # X/Y sentinels -> no repeat geometry
    except ValueError:
        return None

@torch.no_grad()
def encode_full(tk,mk,idxs,bs=128):
    out={}
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; _,H,_=model._encode_set(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        H=H.cpu().numpy()
        for j,gi in enumerate(sel): out[int(gi)]=H[j]
    return out

# clean per-peak donor-id probe on TRAIN H
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0)
HH=[];DD=[]
for gi,H in encode_full(tk_tr,mk_tr,rng.choice(len(at_tr),size=6000,replace=False)).items():
    a=at_tr[gi]; v=np.where(a>=0)[0]; HH.append(H[v]); DD.append(a[v])
Htr=np.concatenate(HH).astype(np.float32); dtr=np.concatenate(DD).astype(int)
clf=nn.Linear(Htr.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
Xt=torch.from_numpy(Htr).to(DEV); yt=torch.from_numpy(dtr).long().to(DEV); lf=nn.CrossEntropyLoss()
for ep in range(60):
    perm=torch.randperm(len(yt),device=DEV)
    for s in range(0,len(yt),8192):
        b=perm[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
W=clf.weight.detach().cpu().numpy(); B=clf.bias.detach().cpu().numpy()
def pred_donor(hrow,contribs):
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
    others=[KNOWN[o] for o in info if o!=d]; gX=geno.get(KNOWN[d],{}); pr=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: pr.add((L,a))
    return pr

keep=[g for g in range(len(at)) if setup(g) is not None]
Hmap=encode_full(tk,mk,np.array(keep))

# per private-peak records of the faintest (rank4) minor
ISO=[]; ABS=[]   # each: dict of structural features
absorb_into_adjacent=0; absorb_into_colocate=0; absorb_total=0
for gi in keep:
    info,v=setup(gi); contribs=list(info.keys())
    d=[c for c in info if info[c]["rank"]==4][0]
    majors=[c for c in info if info[c]["rank"]<4]
    priv=private_of(gi,d,info)
    # present peaks per (donor,locus) for colocation + numeric alleles present per locus per donor
    pk=[j for j in v if int(at[gi][j])==d and (int(tk[gi][j,0]),akey(tk[gi][j,1])) in priv]
    if not pk: continue
    # locus -> present major peaks (count) and major numeric alleles present
    loc_major_alleles={}; loc_major_count={}
    for j in v:
        dj=int(at[gi][j])
        if info.get(dj,{}).get("rank",9)<4:
            L=int(tk[gi][j,0]); an=anum(akey(tk[gi][j,1]))
            loc_major_count[L]=loc_major_count.get(L,0)+1
            if an is not None: loc_major_alleles.setdefault(L,[]).append(an)
    # major GENOTYPE alleles per locus (present or not)
    def major_geno_alleles(L):
        out=[]
        for m in majors:
            for k in geno.get(KNOWN[m],{}).get(L,set()):
                an=anum(k)
                if an is not None: out.append((m,an))
        return out
    for j in pk:
        L=int(tk[gi][j,0]); aj=anum(akey(tk[gi][j,1]))
        glob=float(tk[gi][j,7]); logh=float(tk[gi][j,2])
        pred=pred_donor(Hmap[gi][j],contribs); iso=(pred==d)
        # structural axes
        gma=major_geno_alleles(L)            # [(major_donor, allele_num)]
        backstutter=0; adjacent=0; nearest=99.0; stutter_owner=None; adj_owner=None
        if aj is not None and gma:
            for (m,am) in gma:
                dd=abs(am-aj);
                if dd<nearest: nearest=dd
                if abs(am-aj-1.0)<1e-6:  # aj is one repeat BELOW the major allele = back-stutter slot
                    backstutter=1; stutter_owner=m
                if dd<=1.0+1e-6:
                    adjacent=1; adj_owner=m
        colocate=1 if loc_major_count.get(L,0)>0 else 0   # a major has ANY present peak at this locus
        nmaj=loc_major_count.get(L,0)
        rec={"glob":glob,"logh":logh,"backstutter":backstutter,"adjacent":adjacent,
             "nearest":nearest if nearest<99 else np.nan,"colocate":colocate,"nmaj":nmaj}
        (ISO if iso else ABS).append(rec)
        if not iso:
            absorb_total+=1
            # is the absorbing major the stutter/adjacent owner?
            if stutter_owner==pred or adj_owner==pred: absorb_into_adjacent+=1
            if loc_major_count.get(L,0)>0:
                # absorbed into a donor with a present peak at L?
                pres=set(int(at[gi][j2]) for j2 in v if int(tk[gi][j2,0])==L and info.get(int(at[gi][j2]),{}).get("rank",9)<4)
                if pred in pres: absorb_into_colocate+=1

def summ(name,G):
    if not G: print(f"  {name}: n=0"); return
    n=len(G); f=lambda k:np.nanmean([g[k] for g in G])
    print(f"  {name:9s} n={n:4d} | glob_rel {f('glob'):.3f}  log_h {f('logh'):.2f}  "
          f"|| backstutter-of-major {f('backstutter'):.2f}  adjacent(|d|<=1) {f('adjacent'):.2f}  "
          f"colocate-major {f('colocate'):.2f}  #majpk@locus {f('nmaj'):.2f}")

print("=== faint (rank4) minor PRIVATE peaks: ISOLATED (encoder->minor) vs ABSORBED (encoder->major) ===")
summ("ISOLATED",ISO); summ("ABSORBED",ABS)
nI=len(ISO); nA=len(ABS); tot=nI+nA
print(f"\n  isolation rate on private peaks: {nI/max(tot,1):.3f}  (absorbed {nA}/{tot})")
# decisive conditional risk
def cond(G_iso,G_abs,key):
    a=[g[key] for g in G_abs]; i=[g[key] for g in G_iso]
    pa=np.mean(a) if a else 0; pi=np.mean(i) if i else 0
    # P(absorbed | feature=1)
    n1=sum(1 for g in G_iso+G_abs if g[key]==1); a1=sum(1 for g in G_abs if g[key]==1)
    n0=sum(1 for g in G_iso+G_abs if g[key]==0); a0=sum(1 for g in G_abs if g[key]==0)
    r1=a1/max(n1,1); r0=a0/max(n0,1)
    print(f"    {key:12s}: P(absorbed|{key}=1)={r1:.3f} (n={n1})  vs P(absorbed|{key}=0)={r0:.3f} (n={n0})  lift={r1-r0:+.3f}")
print("\n  conditional absorption risk by structural feature:")
for k in ["backstutter","adjacent","colocate"]:
    cond(ISO,ABS,k)
print(f"\n  of {absorb_total} absorbed peaks: {absorb_into_adjacent} ({absorb_into_adjacent/max(absorb_total,1):.2f}) went to the "
      f"stutter/adjacent-allele owner; {absorb_into_colocate} ({absorb_into_colocate/max(absorb_total,1):.2f}) to a same-locus present major.")

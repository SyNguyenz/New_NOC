"""
WHY don't 1-2 private alleles suffice (oracle needs 1)? Test H1: a "private-among-the-5-contributors"
allele may be carried by MANY of the 45 panel donors -> a single such allele does NOT uniquely point to
the true minor -> the model rationally needs the CONJUNCTION of several. vs H2: the model under-weights
even a genuinely panel-UNIQUE private allele (irrational, fixable).

panel rarity of an allele = #of the 45 known donors whose genotype carries that (locus,allele).

(A) descriptive: recall vs the donor's RAREST private allele's panel count (does a panel-rare private allele
    predict recall?).
(B) causal (decoder, within-donor): keep exactly ONE private peak + all shared, re-decode --
      keep the RAREST (most panel-unique) private  vs  keep the COMMONEST.
    rarest >> commonest => model DOES exploit discriminativeness; the miss is donors whose privates are all
       panel-common = a RATIONAL floor (1 allele genuinely ambiguous across 45).
    rarest ~= commonest (both ~.24) => model under-weights single alleles regardless of uniqueness = a real
       (fixable) capability gap at sparse-evidence integration.
"""
import sys, json, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w"); RUN=sys.argv[1] if len(sys.argv)>1 else "results/inc2_2d_sparse_seed42"
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
geno=load_raw_genotypes()
cfg=json.load(open(Path(RUN)/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
dg=dgm=None
if cfg.get("geno_query"):
    gp=DATA/"donor_geno.npy"; gp=gp if gp.exists() else Path("data/donor_geno.npy")
    dg=torch.from_numpy(np.load(gp).astype(np.float32))
    dgm=torch.from_numpy(np.load(gp.parent/"donor_geno_mask.npy"))
m=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","per_donor"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
    geno_query=cfg.get("geno_query",False),donor_geno=dg,donor_geno_mask=dgm,
    attn_sink=int(cfg.get("attn_sink",0) or 0),donor_recon=cfg.get("donor_recon",False)).to(DEV)
m.load_state_dict(torch.load(Path(RUN)/"best_model.pt",map_location=DEV,weights_only=True),strict=False); m.eval()

tk=np.load(DATA/"tokens8_test.npy")[:,:,:n_tok].astype(np.float32); mk=np.load(DATA/"mask_test.npy").astype(bool)
at=np.load(DATA/"attr_test.npy"); y=np.load(DATA/"y_test_set.npy"); noc=np.clip(np.load(DATA/"noc_test.npy").astype(int),1,5)
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")

# panel rarity: same (L, akey) key space that private_of uses
panel=defaultdict(int)
for donor,loc in geno.items():
    for L,al in loc.items():
        for a in al: panel[(L,a)]+=1

GI=0
@torch.no_grad()
def decode(mkrow):
    o=m(torch.from_numpy(tk[GI][None]).to(DEV),torch.from_numpy(mkrow[None]).to(DEV))
    return torch.sigmoid(o["logits_cls"])[0].cpu().numpy()
def recalled(mkrow,d,k=5): return d in set(np.argsort(decode(mkrow))[::-1][:k])
def private_of(gi,d,contribs):
    others=[KNOWN[o] for o in contribs if o!=d]; gX=geno.get(KNOWN[d],{}); pr=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: pr.add((L,a))
    return pr

# build faint-donor records
recs=[]
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if noc[gi]!=5 or len(np.unique(a[v]))!=5: continue
    lh=tk[gi][:,2]; hsum={int(d):float(np.exp(lh[a==d]).sum()) for d in np.unique(a[v])}
    faint=min(hsum,key=hsum.get); contribs=list(hsum); pr=private_of(gi,faint,contribs)
    ppk=[j for j in v if int(a[j])==faint and (int(tk[gi][j,0]),akey(tk[gi][j,1])) in pr]
    if not ppk: continue
    rar=[panel[(int(tk[gi][j,0]),akey(tk[gi][j,1]))] for j in ppk]   # panel count per private peak
    GI=gi; rec=int(recalled(mk[gi],faint))
    recs.append((gi,faint,ppk,rar,rec))
print(f"N5 faint donors with >=1 private peak: {len(recs)}  overall recall {np.mean([r[4] for r in recs]):.3f}\n")

# (A) recall vs the donor's RAREST private allele panel-count, and #private
print("(A) recall by donor's RAREST private allele's PANEL COUNT (#of 45 donors carrying it):")
rar_best=np.array([min(r[3]) for r in recs]); rec=np.array([r[4] for r in recs])
for lo,hi in [(1,2),(2,4),(4,8),(8,46)]:
    s=(rar_best>=lo)&(rar_best<hi)
    if s.sum(): print(f"    rarest priv in [{lo},{hi}) donors: recall {rec[s].mean():.3f}  n={int(s.sum())}")
print("\n(A2) recall by #private peaks (context):")
npv=np.array([len(r[2]) for r in recs])
for lo,hi in [(1,2),(2,3),(3,5),(5,100)]:
    s=(npv>=lo)&(npv<hi)
    if s.sum(): print(f"    #priv in [{lo},{hi}): recall {rec[s].mean():.3f}  mean rarest-panel {rar_best[s].mean():.1f}  n={int(s.sum())}")

# (B) causal: among baseline-recalled donors with >=3 private, keep ONLY rarest-1 vs commonest-1 (+all shared)
print("\n(B) CAUSAL keep exactly ONE private peak (+ all shared), re-decode:")
sub=[r for r in recs if r[4]==1 and len(r[2])>=3]
hr=hc=0; uniq=[0,0]; nonuniq=[0,0]
for gi,faint,ppk,rar,_ in sub:
    GI=gi; order=np.argsort(rar)                       # ascending panel count = rarest first
    rare_j=ppk[order[0]]; common_j=ppk[order[-1]]; rare_cnt=rar[order[0]]
    for keep_j,acc in [(rare_j,'r'),(common_j,'c')]:
        mkr=mk[gi].copy()
        for j in ppk:
            if j!=keep_j: mkr[j]=False
        ok=recalled(mkr,faint)
        if acc=='r':
            hr+=ok
            (uniq if rare_cnt==1 else nonuniq)[0]+=ok; (uniq if rare_cnt==1 else nonuniq)[1]+=1
        else: hc+=ok
n=len(sub)
print(f"    keep RAREST-1 private:    recall {hr/n:.3f}")
print(f"    keep COMMONEST-1 private: recall {hc/n:.3f}   (n={n})")
print(f"    -- of which kept rarest is PANEL-UNIQUE (in only the true donor): recall {uniq[0]/max(uniq[1],1):.3f}  (n={uniq[1]})")
print(f"    -- kept rarest is panel-NON-unique (>=2 donors):                   recall {nonuniq[0]/max(nonuniq[1],1):.3f}  (n={nonuniq[1]})")
print("\n  PANEL-UNIQUE keep-1 ~1.0 => fully rational (decisive evidence suffices; miss=only-common-privates floor).")
print("  PANEL-UNIQUE keep-1 << 1.0 => model UNDER-USES even a decisive unique allele = real fixable gap.")

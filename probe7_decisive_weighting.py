"""
NO-TRAIN feasibility of the two proposed directions for the verified mechanism (model under-credits
decisive/rare reference-matching alleles):
  dir-1 AFIA / decisive-allele WEIGHTING : score a donor by SUM of 1/panel_count over its reference
         alleles present  (rare-allele match dominates)               -> M_rarity
  dir-2 explicit per-allele MATCHING      : score a donor by COUNT of its reference alleles present
         (unweighted genotype match)                                  -> M_count

Test on N5 faint donor (lowest height-sum), recall = faint in top-5 of the 45-donor score. Compare to the
neural model, and the ensemble neural + lambda*M. SPLIT by #private alleles (the regime where neural fails).

  If M_rarity (and especially neural+M_rarity) RECOVERS the few-private faint donors the neural misses,
  and M_rarity > M_count there => crediting decisive rare alleles is the right lever => DIRECTIONS FEASIBLE.
  If no recovery / M_rarity ~= M_count ~= neural => the hypothesis is wrong, decisive-weighting won't help.
"""
import json, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w"); RUN="results/inc2_2d_sparse_seed42"; DEV="cuda" if torch.cuda.is_available() else "cpu"
geno=load_raw_genotypes()
cfg=json.load(open(Path(RUN)/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
m=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","per_donor"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
    attn_sink=int(cfg.get("attn_sink",0) or 0),donor_recon=cfg.get("donor_recon",False)).to(DEV)
m.load_state_dict(torch.load(Path(RUN)/"best_model.pt",map_location=DEV,weights_only=True),strict=False); m.eval()

tk=np.load(DATA/"tokens8_test.npy")[:,:,:n_tok].astype(np.float32); mk=np.load(DATA/"mask_test.npy").astype(bool)
at=np.load(DATA/"attr_test.npy"); y=np.load(DATA/"y_test_set.npy"); noc=np.clip(np.load(DATA/"noc_test.npy").astype(int),1,5)
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")

panel=defaultdict(int)
for donor,loc in geno.items():
    for L,al in loc.items():
        for a in al: panel[(L,a)]+=1
# donor reference allele lists keyed to class index
ref={}
for c in range(45):
    g=geno.get(KNOWN[c],{}); ref[c]=[(L,a) for L,al in g.items() for a in al]

@torch.no_grad()
def neural_scores(idx,bs=256):
    P=np.zeros((len(idx),45),np.float32)
    for s in range(0,len(idx),bs):
        sel=idx[s:s+bs]; o=m(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        P[s:s+len(sel)]=torch.sigmoid(o["logits_cls"]).cpu().numpy()
    return P

# N5 setup
recs=[]
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if noc[gi]!=5 or len(np.unique(a[v]))!=5: continue
    lh=tk[gi][:,2]; hsum={int(d):float(np.exp(lh[a==d]).sum()) for d in np.unique(a[v])}
    faint=min(hsum,key=hsum.get); contribs=list(hsum)
    obs=set((int(tk[gi][j,0]),akey(tk[gi][j,1])) for j in v)
    # #private of faint among contributors
    others=[KNOWN[o] for o in contribs if o!=faint]; gX=geno.get(KNOWN[faint],{}); npriv=0
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others])
        for aa in al:
            if aa not in oh and (L,aa) in obs: npriv+=1
    recs.append((gi,faint,obs,npriv))
idx=np.array([r[0] for r in recs]); NS=neural_scores(idx); ns_map={g:NS[i] for i,g in enumerate(idx)}

def match_scores(obs):
    mc=np.zeros(45); mr=np.zeros(45)
    for c in range(45):
        for (L,a) in ref[c]:
            if (L,a) in obs:
                mc[c]+=1.0; mr[c]+=1.0/panel[(L,a)]
    return mc,mr

def topk_hit(score,faint,k=5): return faint in set(np.argsort(score)[::-1][:k])
def z(s): s=s.astype(float); return (s-s.mean())/(s.std()+1e-8)

# evaluate per #private bin
bins=[(0,3),(3,5),(5,100)]
res={b:defaultdict(lambda:[0,0]) for b in bins}
LAM=2.0
for gi,faint,obs,npriv in recs:
    mc,mr=match_scores(obs); ns=ns_map[gi]
    cand={"neural":ns,"M_count":mc,"M_rarity":mr,
          "ens+count":z(ns)+LAM*z(mc),"ens+rarity":z(ns)+LAM*z(mr)}
    b=[bb for bb in bins if bb[0]<=npriv<bb[1]][0]
    for name,sc in cand.items():
        res[b][name][0]+=topk_hit(sc,faint); res[b][name][1]+=1

names=["neural","M_count","M_rarity","ens+count","ens+rarity"]
print(f"N5 faint-donor recall (top-5 of 45), by #PRIVATE alleles present:\n")
print(f"  {'#priv':<8}"+"".join(f"{n:>12}" for n in names)+f"{'n':>6}")
for b in bins:
    row=res[b]
    print(f"  [{b[0]},{b[1]}){'':<3}"+"".join(f"{row[n][0]/max(row[n][1],1):>12.3f}" for n in names)+f"{row['neural'][1]:>6}")

# recovery of neural-missed by rarity ensemble
print("\nrecovery on neural-MISSED faint donors:")
miss_tot=rec_rar=rec_cnt=0
for gi,faint,obs,npriv in recs:
    mc,mr=match_scores(obs); ns=ns_map[gi]
    if topk_hit(ns,faint): continue
    miss_tot+=1
    rec_rar+=topk_hit(z(ns)+LAM*z(mr),faint); rec_cnt+=topk_hit(z(ns)+LAM*z(mc),faint)
print(f"  neural missed {miss_tot} faint donors; ens+rarity recovers {rec_rar} ({rec_rar/max(miss_tot,1):.2f}), "
      f"ens+count recovers {rec_cnt} ({rec_cnt/max(miss_tot,1):.2f})")
# N5 SET-EM (all 5 in top-5) — does the gain survive to the actual metric?
print("\nN5 SET-EM (all 5 correct in top-5):")
for name in ["neural","ens+count","ens+rarity"]:
    e=0
    for gi,faint,obs,npriv in recs:
        mc,mr=match_scores(obs); ns=ns_map[gi]
        sc={"neural":ns,"ens+count":z(ns)+LAM*z(mc),"ens+rarity":z(ns)+LAM*z(mr)}[name]
        top=set(np.argsort(sc)[::-1][:5].tolist()); e+=(top==set(np.where(y[gi]==1)[0].tolist()))
    print(f"  {name:<12}: {e/len(recs):.3f}")
print("\n  rarity>count & recovers few-private misses & set-EM up => decisive-allele weighting is the right lever (FEASIBLE).")

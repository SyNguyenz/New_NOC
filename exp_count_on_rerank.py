"""Validate the ORDERING: rerank first, then count on the RERANKED score (count the thing you
threshold) vs the legacy count on the raw prob-profile. Post-hoc on inc22_fixed (single profile)."""
import json, numpy as np, torch
from models.set_transformer import SetTransformerMixture
from train_set_transformer import posthoc_cardinality, posthoc_cardinality_rank, per_noc_em
import phi_rerank as pr
RUN="results/inc22_fixed_aslot_seed42"
def LD(f): return np.load("data_insilico_w/%s.npy"%f)
Xt,Mt,yt,nt=LD("tokens8_test"),LD("mask_test").astype(bool),LD("y_test_set"),LD("noc_test").clip(1,5)
Xv,Mv,yv,nv=LD("tokens8_val"),LD("mask_val").astype(bool),LD("y_val_set"),LD("noc_val").clip(1,5)
g=np.load("data/donor_geno.npy").astype(np.float32); gmask=np.load("data/donor_geno_mask.npy")
ol=torch.zeros(24,1024,45); gm=torch.from_numpy(gmask).bool()
for c in range(45):
  for j in range(g.shape[1]):
    if gm[c,j]:
      li=int(g[c,j,0]); ab=int(round(float(g[c,j,1])*10))+30
      if 0<=li<24 and 0<=ab<1024: ol[li,ab,c]=1.0
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
  dropout=0.1,cls_decoder="aslot",n_token_feats=8,encoder="isab++",num_embed="periodic",periodic_sigma=0.3,
  aux_heads=True,sparse_attn=True,donor_geno=torch.from_numpy(g),donor_geno_mask=torch.from_numpy(gmask),
  nc_attn="mab0",soft_geno_attr=True,feas_filter=True,set_of_set=True,owner_lut=ol,n_slot_iters=3,ot_eps=0.05,
  ot_iters=5,noc_head_v2=True)
m.load_state_dict(torch.load(RUN+"/best_model.pt",weights_only=True,map_location="cpu"),strict=False); m.eval()
@torch.no_grad()
def infer(X,M):
  P,L=[],[]
  for s in range(0,len(X),256):
    lg=m(torch.tensor(X[s:s+256]),torch.tensor(M[s:s+256]))["logits_cls"]; L.append(lg.numpy()); P.append(torch.sigmoid(lg).numpy())
  return np.concatenate(P),np.concatenate(L)
P_te,L_te=infer(Xt,Mt); P_va,L_va=infer(Xv,Mv)
PHv=pr.deconv_phi(Xv,Mv,g,gmask,12); PHt=pr.deconv_phi(Xt,Mt,g,gmask,12)
a=pr.tune_alpha(L_va,PHv,yv,nv)
rank_te=pr.rerank_scores(L_te,PHt,a); rank_va=pr.rerank_scores(L_va,PHv,a)
kP=posthoc_cardinality(P_va,yv,P_te); kR=posthoc_cardinality_rank(rank_va,yv,rank_te)
yti=(yt>0.5).astype(int); ktrue=yti.sum(1); C=45
def dec(sc,k):
  yp=np.zeros((len(sc),C),int)
  for i in range(len(sc)): yp[i,np.argsort(sc[i])[::-1][:int(k[i])]]=1
  return yp
def rep(nm,sc,k):
  e=per_noc_em(yti,dec(sc,k),nt); print(f"  {nm:34s} overall {e[0]:.4f} | N3 {e[3]:.4f} N4 {e[4]:.4f} N5 {e[5]:.4f}")
# combined: count reads BOTH prob-profile (good N3/N4) AND reranked-score profile (good N5)
from sklearn.ensemble import RandomForestClassifier
from train_set_transformer import _card_feats, _rank_feats
def cost_tk(S,Y,lam=0.02):
    tk=np.ones(len(S),int)
    for i in range(len(S)):
        K=max(int(Y[i].sum()),1); best,bc=1,9e9
        for k in range(1,6):
            yp=np.zeros(C); yp[np.argsort(S[i])[::-1][:k]]=1
            c=(Y[i]*(1-yp)).sum()/K+((1-Y[i])*yp).sum()/k+lam*k
            if c<bc: bc,best=c,k
        tk[i]=best
    return tk
Fv=np.concatenate([_card_feats(P_va),_rank_feats(rank_va)],1); Ft=np.concatenate([_card_feats(P_te),_rank_feats(rank_te)],1)
kC=RandomForestClassifier(n_estimators=300,max_depth=6,random_state=42).fit(Fv,cost_tk(rank_va,yv)).predict(Ft)
print(f"alpha={a} | count-acc P={(kP==nt).mean():.3f} rerank={(kR==nt).mean():.3f} combined={(kC==nt).mean():.3f}")
rep("phi-rank + count-on-P (legacy)", rank_te, kP)
rep("phi-rank + count-on-RERANK", rank_te, kR)
rep("phi-rank + count-on-P+RERANK (combined)", rank_te, kC)
rep("phi-rank + ORACLE-k (ceiling)", rank_te, ktrue)

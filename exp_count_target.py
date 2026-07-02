"""
exp_count_target.py — is the N5 deployable count gap a TRAINING-TARGET artifact?
Deployed count (posthoc_cardinality) fits RF on a cost-derived k (lam penalty biases small).
Compare against RF fit on TRUE noc, both on the model prob-profile, under the phi-rerank ranking.
"""
import json, numpy as np, torch
from models.set_transformer import SetTransformerMixture
from sklearn.ensemble import RandomForestClassifier
from train_set_transformer import posthoc_cardinality, per_noc_em, _card_feats
import phi_rerank as pr

DEVICE="cpu"; RUN="results/inc22_fixed_aslot_seed42"; cfg=json.load(open(RUN+"/metrics.json"))["config"]
def LD(f): return np.load("data_insilico_w/%s.npy"%f)
Xt,Mt,yt,nt=LD("tokens8_test"),LD("mask_test").astype(bool),LD("y_test_set"),LD("noc_test").clip(1,5)
Xv,Mv,yv,nv=LD("tokens8_val"),LD("mask_val").astype(bool),LD("y_val_set"),LD("noc_val").clip(1,5)
g=np.load("data/donor_geno.npy").astype(np.float32); gmask=np.load("data/donor_geno_mask.npy")
ALLELE_OFF,n_cls,LUT_W=30,int(cfg.get("n_classes",45)),1024
owner_lut=torch.zeros(24,LUT_W,n_cls); gm=torch.from_numpy(gmask).bool()
for c in range(min(n_cls,g.shape[0])):
    for j in range(g.shape[1]):
        if gm[c,j]:
            li=int(g[c,j,0]); ab=int(round(float(g[c,j,1])*10))+ALLELE_OFF
            if 0<=li<24 and 0<=ab<LUT_W: owner_lut[li,ab,c]=1.0
model=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,
    n_classes=n_cls,n_noc=6,dropout=0.1,cls_decoder="aslot",n_token_feats=8,encoder="isab++",
    num_embed="periodic",periodic_sigma=0.3,aux_heads=True,sparse_attn=True,
    donor_geno=torch.from_numpy(g),donor_geno_mask=torch.from_numpy(gmask),nc_attn="mab0",
    soft_geno_attr=True,feas_filter=True,set_of_set=True,owner_lut=owner_lut,
    n_slot_iters=3,ot_eps=0.05,ot_iters=5,noc_head_v2=True).to(DEVICE)
model.load_state_dict(torch.load(RUN+"/best_model.pt",weights_only=True,map_location=DEVICE),strict=False); model.eval()
@torch.no_grad()
def infer(X,M):
    P=[];L=[]
    for s in range(0,len(X),256):
        o=model(torch.tensor(X[s:s+256]),torch.tensor(M[s:s+256]))
        L.append(o["logits_cls"].numpy()); P.append(torch.sigmoid(o["logits_cls"]).numpy())
    return np.concatenate(P),np.concatenate(L)
P_te,L_te=infer(Xt,Mt); P_va,L_va=infer(Xv,Mv)

PH_te=pr.deconv_phi(Xt,Mt,g,gmask,12); PH_va=pr.deconv_phi(Xv,Mv,g,gmask,12)
alpha=pr.tune_alpha(L_va,PH_va,yv,nv); rank_te=pr.rerank_scores(L_te,PH_te,alpha)

k_cost = posthoc_cardinality(P_va,yv,P_te)                                              # deployed (cost-derived k)
k_true = RandomForestClassifier(300,max_depth=6,random_state=42).fit(_card_feats(P_va),nv).predict(_card_feats(P_te))

def pnc(pred): return {j:round(float((pred[nt==j]==j).mean()),3) for j in range(2,6)}
print("count k-accuracy per NOC:")
print("  cost-derived (deployed):",pnc(k_cost),"overall",round(float((k_cost==nt).mean()),4))
print("  true-noc target        :",pnc(k_true),"overall",round(float((k_true==nt).mean()),4))

C=n_cls; yti=(yt>0.5).astype(int); ktrue=yti.sum(1)
def dec(sc,k):
    yp=np.zeros((len(sc),C),int)
    for i in range(len(sc)): yp[i,np.argsort(sc[i])[::-1][:int(k[i])]]=1
    return yp
def rep(name,sc,k):
    em=per_noc_em(yti,dec(sc,k),nt); print(f"  {name:32s} overall {em[0]:.4f} | N3 {em[3]:.4f} N4 {em[4]:.4f} N5 {em[5]:.4f}")
print("\ndecode EM:")
rep("phi-rank + cost-k (deployed)",rank_te,k_cost)
rep("phi-rank + true-noc-k",       rank_te,k_true)
rep("phi-rank + ORACLE-k (ceiling)",rank_te,ktrue)

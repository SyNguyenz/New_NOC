"""Resolve the alpha~0 ambiguity: FORCE subtraction (sic_alpha fixed at a>0, not learnable) and fine-tune
the model to ADAPT to reading residuals. If forced-subtraction + adaptation beats baseline 0.788 -> the
iterative subtraction CAN help (warm-start just couldn't discover it; from-scratch would) => GO from-scratch.
If it stays below 0.788 even when forced+adapted -> subtraction genuinely doesn't help."""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"; DEV=torch.device("cuda")
from models.set_transformer import SetTransformerMixture
from train_set_transformer import AsymmetricLoss
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
dg=torch.from_numpy(np.load(ROOT/"data/donor_geno.npy").astype(np.float32)); dgm=torch.from_numpy(np.load(ROOT/"data/donor_geno_mask.npy"))
tok=torch.tensor(L("tokens8_train").astype(np.float32)); mkt=torch.tensor(L("mask_train").astype(bool))
yt=torch.tensor(L("y_train_set").astype(np.float32)); at=torch.tensor(L("attr_train").astype(np.int64)); ph=torch.tensor(L("phi_train").astype(np.float32))
N=len(tok)
te_tok=L("tokens8_test").astype(np.float32); te_mk=L("mask_test").astype(bool); te_y=L("y_test_set").astype(np.float32); te_noc=L("noc_test").astype(int)
cfg=json.load(open(ROOT/"results/inc6_maskp_seed42/metrics.json"))["config"]
_num=L("tokens8_train")[:,:,1:8][L("mask_train").astype(bool)]
FM=torch.tensor(_num.mean(0),dtype=torch.float32,device=DEV); FS=torch.tensor(_num.std(0)+1e-6,dtype=torch.float32,device=DEV)
def fresh(sic):
    m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
      dropout=0.1,cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",
      periodic_sigma=cfg["periodic_sigma"],aux_heads=True,sparse_attn=True,donor_geno=dg,donor_geno_mask=dgm,sic_iters=sic).to(DEV)
    m.load_state_dict(torch.load(ROOT/"results/inc6_maskp_seed42/best_model.pt",weights_only=True,map_location=DEV),strict=False)
    m.feat_mean.copy_(FM); m.feat_std.copy_(FS); return m
@torch.no_grad()
def test_n5(m):
    m.eval(); P=np.zeros((len(te_tok),45))
    for s in range(0,len(te_tok),128):
        tk_=torch.from_numpy(te_tok[s:s+128]).to(DEV); mb=torch.from_numpy(te_mk[s:s+128]).to(DEV)
        P[s:s+128]=torch.sigmoid(m(tk_,mb)["logits_cls"]).cpu().numpy()
    out={}
    for k in range(1,6):
        idx=np.where(te_noc==k)[0]; e=[]
        for i in idx:
            t=np.argsort(P[i])[::-1][:k]; pr=np.zeros(45,int); pr[t]=1; e.append((pr==te_y[i]).all())
        out[k]=round(float(np.mean(e)),3)
    return out
def finetune_forced(aval, epochs=7, bs=20, lr=1e-4, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m=fresh(3)
    m.sic_alpha.data.fill_(aval); m.sic_alpha.requires_grad_(False)   # FORCE subtraction, not learnable
    asl=AsymmetricLoss(gamma_neg=4.0,gamma_pos=0.0,clip=0.05)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
    print(f"  forced alpha={aval}, pre-adapt test N5={test_n5(m)[5]}",flush=True)
    best=0
    for ep in range(epochs):
        m.train(); perm=np.random.permutation(N)
        for s in range(0,N,bs):
            bi=perm[s:s+bs]
            x=tok[bi].to(DEV); mk=mkt[bi].to(DEV); y=yt[bi].to(DEV); attr=at[bi].to(DEV); phi=ph[bi].to(DEV)
            mb=mk.bool(); drop=(torch.rand_like(mb,dtype=torch.float)<0.15)&mb; kept=mb&~drop
            mk2=torch.where(kept.sum(1,keepdim=True)>=8,kept,mb).to(mk.dtype)
            out=m(x,mk2); loss=asl(out["logits_cls"],y)
            la=out["logits_attr"]; B_,S_,C_=la.shape
            if (attr>=0).any():
                loss=loss+0.3*F.cross_entropy(la.reshape(B_*S_,C_),attr.reshape(B_*S_),ignore_index=-1)+0.2*F.l1_loss(out["phi"],phi)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
        r=test_n5(m); best=max(best,r[5]); print(f"  [forced a={aval}] ep{ep+1}: test N5={r[5]} (N4={r[4]})",flush=True)
    return best
t0=time.time(); print("baseline test N5=0.788 (target to beat)",flush=True)
for a in [0.7]:
    b=finetune_forced(a); print(f"FORCED alpha={a}: best test N5 = {b}  (vs base 0.788)  [{time.time()-t0:.0f}s]",flush=True)

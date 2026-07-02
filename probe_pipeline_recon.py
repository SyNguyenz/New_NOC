"""WHOLE-PIPELINE probe of the 'learn the process' (forward-reconstruction) idea — fair this time:
warm-start inc6_maskp, then FINE-TUNE the encoder+decoder (a) with the base objective only (CONTROL) and
(b) with the base objective + forward reconstruction (RECON). Measure REAL TEST N5 oracle (test combos are
never seen -> clean generalization). recon > control on test => 'learn the process' shapes the pipeline
toward better combinatorial generalization. This is the pipeline test the bolt-on probes couldn't do.
"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, copy, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda")
from models.set_transformer import SetTransformerMixture
from train_set_transformer import AsymmetricLoss
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy"); OFF=30; Wd=1024
owner_lut=torch.zeros(24,Wd,45)
for c in range(45):
    for j in range(g.shape[1]):
        if gm[c,j]:
            li=int(g[c,j,0]); ab=int(round(float(g[c,j,1])*10))+OFF
            if 0<=li<24 and 0<=ab<Wd: owner_lut[li,ab,c]=1.0
owner_lut=owner_lut.to(DEV)
def gather_owner(t):
    loc=t[:,:,0].long().clamp(0,23); ab=(torch.round(t[:,:,1]*10).long()+OFF).clamp(0,Wd-1); return owner_lut[loc,ab]

tok=torch.tensor(L("tokens8_train").astype(np.float32)); mkt=torch.tensor(L("mask_train").astype(bool))
yt=torch.tensor(L("y_train_set").astype(np.float32)); at=torch.tensor(L("attr_train").astype(np.int64)); ph=torch.tensor(L("phi_train").astype(np.float32))
N=len(tok)
te_tok=L("tokens8_test").astype(np.float32); te_mk=L("mask_test").astype(bool); te_y=L("y_test_set").astype(np.float32); te_noc=L("noc_test").astype(int)
cfg=json.load(open(ROOT/"results/inc6_maskp_seed42/metrics.json"))["config"]
def fresh():
    m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
      dropout=0.1,cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",
      periodic_sigma=cfg["periodic_sigma"],aux_heads=True,sparse_attn=True).to(DEV)
    m.load_state_dict(torch.load(ROOT/"results/inc6_maskp_seed42/best_model.pt",weights_only=True,map_location=DEV))
    return m
_num=L("tokens8_train")[:,:,1:8][L("mask_train").astype(bool)]
FM=torch.tensor(_num.mean(0),dtype=torch.float32,device=DEV); FS=torch.tensor(_num.std(0)+1e-6,dtype=torch.float32,device=DEV)

@torch.no_grad()
def test_n5(m):
    m.eval(); P=np.zeros((len(te_tok),45))
    for s in range(0,len(te_tok),256):
        tk=torch.from_numpy(te_tok[s:s+256]).to(DEV); mb=torch.from_numpy(te_mk[s:s+256]).to(DEV)
        P[s:s+256]=torch.sigmoid(m(tk,mb)["logits_cls"]).cpu().numpy()
    out={}
    for k in range(1,6):
        idx=np.where(te_noc==k)[0]; e=[]
        for i in idx:
            t=np.argsort(P[i])[::-1][:k]; pr=np.zeros(45,int); pr[t]=1; e.append((pr==te_y[i]).all())
        out[k]=round(float(np.mean(e)),3)
    return out

def finetune(use_recon, epochs=8, bs=40, lr=2e-4, recon_w=0.3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m=fresh(); m.feat_mean.copy_(FM); m.feat_std.copy_(FS)
    asl=AsymmetricLoss(gamma_neg=4.0,gamma_pos=0.0,clip=0.05)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    for ep in range(epochs):
        m.train(); perm=np.random.permutation(N)
        for s in range(0,N,bs):
            bi=perm[s:s+bs]
            x=tok[bi].to(DEV); mk=mkt[bi].to(DEV); y=yt[bi].to(DEV); attr=at[bi].to(DEV); phi=ph[bi].to(DEV)
            mb=mk.bool()                                   # mask_peaks 0.15 augmentation (as base)
            drop=(torch.rand_like(mb,dtype=torch.float)<0.15)&mb; kept=mb&~drop
            mk2=torch.where(kept.sum(1,keepdim=True)>=8,kept,mb).to(mk.dtype)
            out=m(x,mk2)
            loss=asl(out["logits_cls"],y)
            la=out["logits_attr"]; B_,S_,C_=la.shape
            if (attr>=0).any():
                loss=loss+0.3*F.cross_entropy(la.reshape(B_*S_,C_),attr.reshape(B_*S_),ignore_index=-1)
                loss=loss+0.1*F.l1_loss(out["phi"],phi)
            if use_recon:
                owner=gather_owner(x); w=torch.sigmoid(out["logits_cls"])*out["phi"]
                pred=(owner*w.unsqueeze(1)).sum(-1); obs=torch.expm1(x[:,:,2])
                em=((owner.sum(-1)>0)&mk2.bool()).float()
                pn=pred/(pred*em).sum(1,keepdim=True).clamp(min=1e-6); on=obs/(obs*em).sum(1,keepdim=True).clamp(min=1e-6)
                loss=loss+recon_w*((F.l1_loss(torch.log1p(1e3*pn),torch.log1p(1e3*on),reduction='none')*em).sum()/em.sum().clamp(min=1))
            opt.zero_grad(); loss.backward(); opt.step()
        r=test_n5(m); print(f"  [{'RECON' if use_recon else 'CONTROL'}] ep{ep+1}: test N5={r[5]} (N1-4 {r[1]}/{r[2]}/{r[3]}/{r[4]})",flush=True)
    return r

t0=time.time()
print("baseline (inc6_maskp, no fine-tune):", test_n5(fresh()),flush=True)
print("\n=== CONTROL fine-tune (base objective only) ==="); rc=finetune(False)
print("\n=== RECON fine-tune (+ forward reconstruction) ==="); rr=finetune(True)
print(f"\nSUMMARY  control test N5={rc[5]}  |  recon test N5={rr[5]}  |  recon-control={rr[5]-rc[5]:+.3f}  ({time.time()-t0:.0f}s)")

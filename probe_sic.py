"""Fast whole-pipeline read of deep-unfolded SIC (Inc16): warm-start inc6_maskp (model REDUCES to base at
init via sic_alpha=0), fine-tune CONTROL (sic_iters=1) vs SIC (sic_iters=3) with the SAME objective + low
lr (warm-start stays near 0.788). recon-probe lesson: low lr so control doesn't drift; the sic-vs-control
gap isolates the iterative-subtraction value. Measure REAL TEST N5 (clean novel-combo generalization)."""
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
def finetune(sic, epochs=6, bs=20, lr=5e-5, seed=0, tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    m=fresh(sic); asl=AsymmetricLoss(gamma_neg=4.0,gamma_pos=0.0,clip=0.05)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    for ep in range(epochs):
        m.train(); perm=np.random.permutation(N)
        for s in range(0,N,bs):
            bi=perm[s:s+bs]
            x=tok[bi].to(DEV); mk=mkt[bi].to(DEV); y=yt[bi].to(DEV); attr=at[bi].to(DEV); phi=ph[bi].to(DEV)
            mb=mk.bool(); drop=(torch.rand_like(mb,dtype=torch.float)<0.15)&mb; kept=mb&~drop
            mk2=torch.where(kept.sum(1,keepdim=True)>=8,kept,mb).to(mk.dtype)
            out=m(x,mk2)
            loss=asl(out["logits_cls"],y)
            la=out["logits_attr"]; B_,S_,C_=la.shape
            if (attr>=0).any():
                loss=loss+0.3*F.cross_entropy(la.reshape(B_*S_,C_),attr.reshape(B_*S_),ignore_index=-1)+0.2*F.l1_loss(out["phi"],phi)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
        r=test_n5(m)
        a=m.sic_alpha.detach().cpu().numpy().round(2) if sic>1 else "-"
        print(f"  [{tag}] ep{ep+1}: test N5={r[5]} (N1-4 {r[1]}/{r[2]}/{r[3]}/{r[4]})  alpha={a}",flush=True)
    return r
t0=time.time()
print("baseline:", test_n5(fresh(1)),flush=True)
print("\n=== CONTROL (sic_iters=1) ==="); rc=finetune(1,tag="CTRL")
print("\n=== SIC (sic_iters=3) ==="); rs=finetune(3,tag="SIC")
print(f"\nSUMMARY  control N5={rc[5]}  |  SIC N5={rs[5]}  |  SIC-control={rs[5]-rc[5]:+.3f}  ({time.time()-t0:.0f}s)")

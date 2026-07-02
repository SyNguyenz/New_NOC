"""Smoke Inc14 B-v2 + A-v2: exercise the new code paths (ml_attr head, MLD distill, additive-subtraction
counterfactual, conditional MMD) on a tiny real batch — catch shape/index/finite bugs before Kaggle."""
import json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
from train_set_transformer import subtract_height, cond_slot_mmd

def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
B=12
tok=torch.tensor(L("tokens8_train")[:B].astype(np.float32),device=DEV)
mk=torch.tensor(L("mask_train")[:B].astype(bool),device=DEV)
y=torch.tensor(L("y_train_set")[:B].astype(np.float32),device=DEV)
attr=torch.tensor(L("attr_train")[:B].astype(np.int64),device=DEV)
phi=torch.tensor(L("phi_train")[:B].astype(np.float32),device=DEV)

# owner_lut + gather (mirror train)
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
ALLELE_OFF=30; LUT_W=1024
owner_lut=torch.zeros(24,LUT_W,45)
for c in range(min(45,g.shape[0])):
    for j in range(g.shape[1]):
        if gm[c,j]:
            li=int(g[c,j,0]); ab=int(round(float(g[c,j,1])*10))+ALLELE_OFF
            if 0<=li<24 and 0<=ab<LUT_W: owner_lut[li,ab,c]=1.0
owner_lut=owner_lut.to(DEV)
def gather_owner(t):
    loc=t[:,:,0].long().clamp(0,23)
    ab=(torch.round(t[:,:,1]*10).long()+ALLELE_OFF).clamp(0,owner_lut.size(1)-1)
    return owner_lut[loc,ab]

cfg=json.load(open(ROOT/"results/inc6_maskp_seed42/metrics.json"))["config"]
m=SetTransformerMixture(n_loci=cfg["n_loci"],d_locus=cfg["d_locus"],d_model=cfg["d_model"],n_heads=cfg["n_heads"],
    n_isab=cfg["n_isab"],m_inducing=cfg["m_inducing"],n_classes=45,n_noc=6,dropout=cfg["dropout"],
    cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",periodic_sigma=cfg["periodic_sigma"],
    aux_heads=True,sparse_attn=True,ml_attr=True).to(DEV)
m.train()
# standardize feats so the periodic embedding sees sane scale
_num=L("tokens8_train")[:2000,:,1:8][L("mask_train")[:2000].astype(bool)]
m.feat_mean.copy_(torch.tensor(_num.mean(0),dtype=torch.float32,device=DEV))
m.feat_std.copy_(torch.tensor(_num.std(0)+1e-6,dtype=torch.float32,device=DEV))

out=m(tok,mk)
assert "logits_mlattr" in out and out["logits_mlattr"].shape==(B,tok.size(1),45), out["logits_mlattr"].shape
reps_main=m.cls_decoder_module.last_reps.detach()
print("forward OK | logits_mlattr",tuple(out["logits_mlattr"].shape),"| last_reps",tuple(reps_main.shape))

# B-v2: multi-label BCE + MLD distill
owner=gather_owner(tok); ml_tgt=owner*(y>0.5).float().unsqueeze(1); vpk=mk.bool().unsqueeze(-1).float()
lm=out["logits_mlattr"]
bce=F.binary_cross_entropy_with_logits(lm,ml_tgt,reduction="none",pos_weight=torch.tensor(20.0,device=DEV))
loss_mlattr=(bce*vpk).sum()/vpk.sum().clamp(min=1.0)/45
ap=torch.sigmoid(lm).masked_fill(~mk.bool().unsqueeze(-1),0.0); teacher_p=ap.max(1).values.clamp(1e-4,1-1e-4)
loss_mld=F.binary_cross_entropy_with_logits(out["logits_cls"],teacher_p)
print(f"B-v2: ml_attr BCE={loss_mlattr.item():.4f}  MLD distill={loss_mld.item():.4f}  "
      f"owners/peak(present)={(ml_tgt.sum(-1)[mk.bool()]).mean().item():.2f}")

# A-v2: additive-subtraction counterfactual + conditional MMD
present=(y>0.5); valid_s=present.sum(1)>=2
w=owner*phi.clamp(min=0).unsqueeze(1)*present.float().unsqueeze(1)
probs=present.float().clone(); probs[~valid_s]=1.0
c_i=torch.multinomial(probs,1).squeeze(1)
wsum=w.sum(-1).clamp(min=1e-6)
wc=w.gather(-1,c_i.view(-1,1,1).expand(-1,w.size(1),1)).squeeze(-1)
mult=(1.0-(wc/wsum)).clamp(0.0,1.0)
t_sub=subtract_height(tok,mk,mult)
_=m(t_sub,mk); reps_cf=m.cls_decoder_module.last_reps
keep=present.clone(); keep[torch.arange(B,device=DEV),c_i]=False; keep=keep&valid_s.unsqueeze(1)
id_dim=64
loss_addinv=cond_slot_mmd(reps_main[:,:,:id_dim],reps_cf[:,:,:id_dim],keep)
frac_mean=(wc/wsum)[mk.bool()].mean().item()
print(f"A-v2: subtract mult mean={mult[mk.bool()].mean().item():.3f} (frac_c mean={frac_mean:.3f})  "
      f"cond-MMD={float(loss_addinv):.5f}  kept-donor slots={int((keep.sum(0)>=4).sum())}")

tot=loss_mlattr+loss_mld+loss_addinv+out["logits_cls"].abs().mean()*0
tot.backward()
gnorm=sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
allfinite=all(torch.isfinite(v).all() for v in [loss_mlattr,loss_mld,loss_addinv])
print(f"backward OK | grad-norm sum={gnorm:.1f} | all losses finite={allfinite}")
print("SMOKE PASS" if allfinite and np.isfinite(gnorm) else "SMOKE FAIL")

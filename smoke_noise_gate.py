import numpy as np, torch, torch.nn.functional as F
from models.set_transformer import SetTransformerMixture
g=np.load("data/donor_geno.npy").astype(np.float32); gmask=np.load("data/donor_geno_mask.npy")
owner_lut=torch.zeros(24,1024,45); gm=torch.from_numpy(gmask).bool()
for c in range(45):
    for j in range(g.shape[1]):
        if gm[c,j]:
            li=int(g[c,j,0]); ab=int(round(float(g[c,j,1])*10))+30
            if 0<=li<24 and 0<=ab<1024: owner_lut[li,ab,c]=1.0
def build(noise_gate):
    return SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,
        n_noc=6,dropout=0.1,cls_decoder="aslot",n_token_feats=8,encoder="isab++",num_embed="periodic",
        periodic_sigma=0.3,aux_heads=True,sparse_attn=True,donor_geno=torch.from_numpy(g),
        donor_geno_mask=torch.from_numpy(gmask),nc_attn="mab0",soft_geno_attr=True,feas_filter=True,
        set_of_set=True,owner_lut=owner_lut,n_slot_iters=3,ot_eps=0.05,ot_iters=5,noc_head_v2=True,
        noise_gate=noise_gate).eval()
sd=torch.load("results/inc22_fixed_aslot_seed42/best_model.pt",weights_only=True,map_location="cpu")

# (a) default-off = bit-identical: strict load must succeed (no new params)
A=build(False); A.load_state_dict(sd, strict=True); print("(a) default-off strict load OK -> bit-identical to prior")

# (b) on adds ONLY the gate head
B=build(True); miss,unexp=B.load_state_dict(sd, strict=False)
print(f"(b) noise_gate=True load: missing={list(miss)} unexpected={unexp}")

# (c) forward: shape + near-no-op start + grads flow
Xt=torch.tensor(np.load("data_insilico_w/tokens8_test.npy")[:8])
Mt=torch.tensor(np.load("data_insilico_w/mask_test.npy")[:8].astype(bool))
yt=torch.tensor(np.load("data_insilico_w/y_test_set.npy")[:8])
with torch.no_grad():
    oA=A(Xt,Mt)["logits_cls"]; oB=B(Xt,Mt)
print(f"(c) noise_gate_logit shape {tuple(oB['noise_gate_logit'].shape)} | start max|Δlogit| vs off = {float((oB['logits_cls']-oA).abs().max()):.4f} (small => graceful start)")
out=B(Xt,Mt)
real=((torch.einsum('bnc,bc->bn', _g:=(owner_lut[Xt[:,:,0].long().clamp(0,23),(Xt[:,:,1]*10).round().long().clamp(0,1023)]), yt))>0).float()
mv=Mt.bool()
loss=F.binary_cross_entropy_with_logits(out["noise_gate_logit"][mv], real[mv]); loss.backward()
gnorm=B.noise_gate_head[0].weight.grad.abs().sum().item()
print(f"(c) supervised BCE={loss.item():.4f}  gate-head grad-norm={gnorm:.4f} (>0 => trains)")

# (d) structural features flag a synthetic back-stutter peak
# locus 5: parent allele 12 @ 1000 RFU, stutter allele 11 @ 60 RFU
tok=torch.zeros(1,4,8); msk=torch.zeros(1,4,dtype=torch.bool)
tok[0,0]=torch.tensor([5,12,np.log1p(1000),0,0,0,0,0]); tok[0,1]=torch.tensor([5,11,np.log1p(60),0,0,0,0,0])
msk[0,0]=msk[0,1]=True
f=B._stutter_feats(tok,msk)[0]   # (4,6): [log1p h, BSR, has_back, FSR, has_fwd, rel_h]
print(f"(d) parent(a12): BSR={f[0,1]:.3f} has_back={f[0,2]:.0f} rel_h={f[0,5]:.2f}")
print(f"(d) stutter(a11): BSR={f[1,1]:.3f} has_back={f[1,2]:.0f} rel_h={f[1,5]:.2f}  (BSR~0.06, has_back=1, rel_h low => flagged)")

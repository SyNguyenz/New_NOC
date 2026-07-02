"""
Size m_inducing with evidence instead of guessing. The Encoder-A absorption hypothesis is that faint peaks
must SHARE inducing slots with majors in the m=32 bottleneck. Decisive question: is the m=32 bottleneck
SATURATED on N5 (high effective rank -> widening to 64 should relieve sharing) or UNDER-USED (low effective
rank -> capacity is not the binding constraint, prefer a bypass/de-competition lever)?

effective rank = participation ratio of the 32 inducing-encoded vectors' singular values:  (Σσ)² / Σσ²
(=32 means all slots used independently; <<32 means redundant slots).
Also: per-slot 'specialization' — does any slot dominate (would explain faint peaks losing their slot)?
"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture

DATA=Path("data_insilico_w")
RUN=Path(sys.argv[1]) if len(sys.argv)>1 else Path("results/inc11_nc_mab0_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"

cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]; M=cfg.get("m_inducing",32)
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=M,n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),nc_attn=cfg.get("nc_attn","none"),
    nc_learnable_bias=cfg.get("nc_learnable_bias",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()
tk=np.load(DATA/"tokens8_test.npy")[:,:,:n_tok].astype(np.float32); mk=np.load(DATA/"mask_test.npy").astype(bool)
at=np.load(DATA/"attr_test.npy")

# N5 sample ids
n5=[g for g in range(len(at)) if len(np.unique(at[g][at[g]>=0]))==5]
n5=np.array(n5[:600])

isab0=model.encoder[0]
@torch.no_grad()
def inducing_encoded(idxs,bs=128):
    outs=[]
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        x0,pad=model._project_tokens(t,m)
        I=isab0.I.expand(x0.size(0),-1,-1)
        Hi=isab0.mab0(I,x0,kv_mask=pad) if 'kv_mask' in isab0.mab0.forward.__code__.co_varnames else isab0.mab0(I,x0)
        outs.append(Hi.cpu().numpy())
    return np.concatenate(outs,0)   # (n,M,d)

Hi=inducing_encoded(n5)
# per-sample effective rank
ers=[]
for s in range(Hi.shape[0]):
    A=Hi[s]-Hi[s].mean(0,keepdims=True)
    sv=np.linalg.svd(A,compute_uv=False)
    ers.append((sv.sum()**2)/(np.square(sv).sum()+1e-9))
ers=np.array(ers)
# slot dominance: norm per slot averaged over samples
slotnorm=np.linalg.norm(Hi,axis=2).mean(0)   # (M,)
print(f"=== {RUN.name}  m_inducing={M}  (N5 n={len(n5)}) ===")
print(f"  effective rank of the {M} inducing slots : mean {ers.mean():.1f}  median {np.median(ers):.1f}  (max possible {M})")
print(f"  saturation = mean_eff_rank / M           : {ers.mean()/M:.2f}   (>~0.8 => bottleneck saturated => widening to {M*2} should relieve slot-sharing)")
print(f"  slot-norm spread (max/min)               : {slotnorm.max()/ (slotnorm.min()+1e-9):.1f}  (high => a few slots dominate)")
print(f"  # slots carrying <10% of max norm (dead) : {(slotnorm < 0.1*slotnorm.max()).sum()} / {M}")

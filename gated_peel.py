"""
gated_peel.py — GATED greedy peel (no-train). Fix the net-negative of greedy_peel.py:
KEEP neural's confident picks (prob >= tau) FIXED, use residual-peel ONLY to fill the
remaining uncertain slots (where buried minor donors live). tau=0 => pure neural baseline;
tau=1 => pure peel. Sweep tau to find if any setting beats neural N5 set-EM (.726).
Same scoring space as quick_peel/greedy_peel. Count given (K=true NOC).
"""
import json
from pathlib import Path
import numpy as np, torch
from scipy.optimize import nnls
import pgnoc as PG
from train_set_transformer import DEVICE
from models.set_transformer import SetTransformerMixture
from features.enrich import enrich_tokens

D = Path("data_insilico_w"); arm = "inc4_p3_irm_seed42"
def build_any(cfg):
    return SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,
        n_classes=45,n_noc=6,dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),
        decoder_source=cfg.get("decoder_source","encoded"),n_token_feats=cfg.get("n_token_feats",8),
        encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
        n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
        aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False)).to(DEVICE)
cfg=json.load(open(Path("results")/arm/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats",8)
st=torch.load(Path("results")/arm/"best_model.pt",map_location=DEVICE,weights_only=True)
model=build_any(cfg); model.load_state_dict(st); model.eval()

tt=np.load(D/"tokens_test.npy").astype(np.float32); mt=np.load(D/"mask_test.npy").astype(bool)
yt=np.load(D/"y_test_set.npy").astype(int); nt=np.clip(np.load(D/"noc_test.npy").astype(int),1,5)
Xf=np.load(D/"Xflat_test.npy").astype(np.float64); ent=enrich_tokens(tt,mt)[:,:,:n_tok]
P=[]
with torch.no_grad():
    for i in range(0,len(ent),256):
        P.append(torch.sigmoid(model(torch.from_numpy(ent[i:i+256]).to(DEVICE),torch.from_numpy(mt[i:i+256]).to(DEVICE))["logits_cls"]).cpu().numpy())
P=np.concatenate(P)
G=PG.build_refs(); Gn=G/(np.linalg.norm(G,axis=1,keepdims=True)+1e-9)

def gated_pick(hrel, K, order, probs, tau):
    picks=[d for d in order if probs[d]>=tau][:K]    # confident neural picks, kept fixed
    while len(picks)<K:
        if picks:
            A=G[picks].T; phi,_=nnls(A,hrel); resid=np.clip(hrel-A@phi,0,None)
        else:
            resid=hrel.copy()
        if resid.sum()<1e-9:
            for d in order:
                if d not in picks: picks.append(int(d)); break
            continue
        rn=resid/(np.linalg.norm(resid)+1e-9); sims=Gn@rn; sims[picks]=-1e9
        picks.append(int(np.argmax(sims)))
    return picks

def bucket(nr): return "neural<=5" if nr<=5 else ("neural 6-15" if nr<=15 else "neural>=16")
B=["neural<=5","neural 6-15","neural>=16"]
idx5=np.where(nt==5)[0]
taus=[0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

print(f"arm={arm}  N5 GATED greedy-peel (no-train) | n_samples={len(idx5)}")
print("tau=0 => neural baseline; tau=1 => pure peel. Goal: beat neural set-EM .726\n")
print(f"{'tau':>5}{'setEM':>9}{'avg#kept':>10} | recovery  {'<=5':>7}{'6-15':>7}{'>=16':>7}")
best=(-1,None)
for tau in taus:
    setem=[]; rec={k:[] for k in B}; kept=[]
    for i in idx5:
        S=set(np.where(yt[i]==1)[0]); hrel=Xf[i]/(Xf[i].sum()+1e-12)
        order=list(np.argsort(P[i])[::-1]); probs=P[i]
        nkept=min(len([d for d in order if probs[d]>=tau]),5); kept.append(nkept)
        picks=set(gated_pick(hrel,5,order,probs,tau))
        setem.append(S==picks)
        for d in S: rec[bucket(order.index(d)+1)].append(d in picks)
    se=np.mean(setem)
    if se>best[0]: best=(se,tau)
    r=lambda k:(np.mean(rec[k]) if rec[k] else float('nan'))
    print(f"{tau:>5.1f}{se:>9.3f}{np.mean(kept):>10.2f} | {'':9}{r('neural<=5'):>7.2f}{r('neural 6-15'):>7.2f}{r('neural>=16'):>7.2f}")
print(f"\nneural baseline set-EM = 0.726 (tau=0 row). BEST gated = {best[0]:.3f} at tau={best[1]}")
print("If BEST > .726 => gating fixes the net-negative => peel-assisted decode FEASIBLE even with hand scorer.")
print("If BEST <= .726 => even gated, the hand cosine tail can't beat neural => MUST train a residual scorer.")

"""
neural_peel.py — proxy for "does end-to-end break the .91 frozen-anchor?".
Replace the WEAK cosine tail-scorer of gated_peel.py with the FROZEN NEURAL model itself
re-run on the physically-correct LINEAR residual (subtract picked donors' NNLS contribution
in linear RFU space, re-tokenize, re-enrich, re-score). Still NO training — uses the existing
strong scorer as a proxy for a learned residual scorer. Gated (keep neural picks prob>=tau).
If gated+neural-residual >> gated+cosine (.758) and approaches the oracle-peel band, the SCORER
was the bottleneck => a learned/end-to-end scorer has real headroom past .91. If it stalls near
.758, the frozen representation lacks the info => end-to-end won't help much either.
"""
import json
from pathlib import Path
import numpy as np, torch
from scipy.optimize import nnls
import pgnoc as PG
from train_set_transformer import DEVICE
from models.set_transformer import SetTransformerMixture
from features.enrich import enrich_tokens
from make_insilico import xflat_to_tokens   # flat<->token map (BIN_LOCUS/ALLELE, MAX_SEQ)

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
G=PG.build_refs()                       # (45,590) LINEAR relative reference profiles

@torch.no_grad()
def neural_score(resid_lin):
    """re-tokenize a LINEAR residual profile, enrich, run frozen model -> 45 probs."""
    rflat=np.log1p(np.clip(resid_lin,0,None)).astype(np.float32)
    tok,mask,_,_=xflat_to_tokens(rflat)
    en=enrich_tokens(tok[None],mask[None])[:,:,:n_tok]
    pr=torch.sigmoid(model(torch.from_numpy(en).to(DEVICE),torch.from_numpy(mask[None]).to(DEVICE))["logits_cls"])
    return pr[0].cpu().numpy()

def neural_gated_pick(i, K, tau):
    mix=np.expm1(Xf[i])                              # linear RFU
    order=list(np.argsort(P[i])[::-1]); probs=P[i]
    picks=[int(d) for d in order if probs[d]>=tau][:K]
    if not picks: picks=[int(order[0])]             # always seed a strong first pick
    while len(picks)<K:
        A=G[picks].T; phi,_=nnls(A,mix); resid=np.clip(mix-A@phi,0,None)
        if resid.sum()<1e-9:
            for d in order:
                if d not in picks: picks.append(int(d)); break
            continue
        pr=neural_score(resid); pr[picks]=-1e9; picks.append(int(np.argmax(pr)))
    return picks

def bucket(nr): return "neural<=5" if nr<=5 else ("neural 6-15" if nr<=15 else "neural>=16")
B=["neural<=5","neural 6-15","neural>=16"]
idx5=np.where(nt==5)[0]
taus=[0.0,0.7,0.8,0.9]

print(f"arm={arm}  N5 NEURAL-residual gated peel (no-train, physical linear peel) | n={len(idx5)}")
print("baselines: neural top5 .726 | gated+COSINE best .758 | oracle-peel upper recovery .99/.85/.65\n")
print(f"{'tau':>5}{'setEM':>9} | recovery  {'<=5':>7}{'6-15':>7}{'>=16':>7}")
best=(-1,None)
for tau in taus:
    setem=[]; rec={k:[] for k in B}
    for i in idx5:
        S=set(np.where(yt[i]==1)[0]); order=list(np.argsort(P[i])[::-1])
        picks=set(neural_gated_pick(i,5,tau))
        setem.append(S==picks)
        for d in S: rec[bucket(order.index(d)+1)].append(d in picks)
    se=np.mean(setem)
    if se>best[0]: best=(se,tau)
    r=lambda k:(np.mean(rec[k]) if rec[k] else float('nan'))
    print(f"{tau:>5.1f}{se:>9.3f} | {'':9}{r('neural<=5'):>7.2f}{r('neural 6-15'):>7.2f}{r('neural>=16'):>7.2f}")
print(f"\nBEST neural-residual gated = {best[0]:.3f} at tau={best[1]}  (vs cosine .758, neural .726)")
print("Headroom verdict: if >> .758 and recovers 6-15/>=16 toward .85/.65 => SCORER was the bottleneck,")
print("learned/end-to-end has real headroom past .91. If ~.758 => frozen repr is the limit.")

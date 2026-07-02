"""
greedy_peel.py — REALISTIC (no ground-truth) feasibility of recursive peeling for N5.
Counterpart to quick_peel.py (which used oracle-peel = ground-truth others => UPPER bound).
Here we peel GREEDILY in the order the model/scorer itself chooses, NNLS-subtract the
picked donors' fitted contributions, re-rank the residual, repeat K times. This exposes
error-compounding (a wrong early pick corrupts the residual) => a realistic LOWER band.

Two variants bracket the real learned recursive-matching decoder:
  - greedy_seed : pick #1 = neural argmax (proxy for a strong learned 1st pick), tail = cosine-on-residual
  - greedy_cos  : ALL picks = cosine-on-residual (pure matched-filter peel; weak scorer => pessimistic)
Compared against: neural top-5 baseline, and the oracle-peel UPPER bound (.99/.85/.65 from quick_peel).
Same scoring space as quick_peel (raw log1p Xflat normalized vs ref G) so numbers are apples-to-apples.
Count is given (K=true NOC) => this isolates the ID-oracle question.
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

def greedy_pick(hrel, K, seed_neural_order=None):
    """Return list of K picked donor indices via greedy NNLS-peel. If seed_neural_order given,
    pick #1 = its argmax; else pick #1 by cosine to hrel."""
    picks=[]
    if seed_neural_order is not None:
        picks.append(int(seed_neural_order[0]))
    else:
        rn=hrel/(np.linalg.norm(hrel)+1e-9); picks.append(int(np.argmax(Gn@rn)))
    while len(picks)<K:
        A=G[picks].T; phi,_=nnls(A,hrel); resid=np.clip(hrel-A@phi,0,None)
        if resid.sum()<1e-9:
            sims=np.full(45,-1e9)            # nothing left to explain; fall back to neural order
            if seed_neural_order is not None:
                for d in seed_neural_order:
                    if d not in picks: picks.append(int(d)); break
                continue
        rn=resid/(np.linalg.norm(resid)+1e-9); sims=Gn@rn
        sims[picks]=-1e9; picks.append(int(np.argmax(sims)))
    return picks

def bucket(nr):
    return "neural<=5" if nr<=5 else ("neural 6-15" if nr<=15 else "neural>=16")
B=["neural<=5","neural 6-15","neural>=16"]
rec={"neural_top5":{k:[] for k in B},"greedy_seed":{k:[] for k in B},"greedy_cos":{k:[] for k in B}}
setEM={"neural_top5":[],"greedy_seed":[],"greedy_cos":[]}

idx5=np.where(nt==5)[0]
for i in idx5:
    S=set(np.where(yt[i]==1)[0]); hrel=Xf[i]/(Xf[i].sum()+1e-12)
    order=list(np.argsort(P[i])[::-1])
    top5=set(order[:5])
    g_seed=set(greedy_pick(hrel,5,seed_neural_order=order))
    g_cos =set(greedy_pick(hrel,5,seed_neural_order=None))
    setEM["neural_top5"].append(S==top5); setEM["greedy_seed"].append(S==g_seed); setEM["greedy_cos"].append(S==g_cos)
    for d in S:
        b=bucket(order.index(d)+1)
        rec["neural_top5"][b].append(d in top5)
        rec["greedy_seed"][b].append(d in g_seed)
        rec["greedy_cos"][b].append(d in g_cos)

print(f"arm={arm}  N5 GREEDY-peel (NO ground-truth) — realistic lower band  | n_samples={len(idx5)}\n")
print("Per-true-donor RECOVERY (in the K=5 picks), bucketed by NEURAL rank:")
print(f"{'bucket':<14}{'n':>5}{'neural@5':>10}{'oraclePeel':>12}{'greedy_seed':>13}{'greedy_cos':>12}")
ora={"neural<=5":0.99,"neural 6-15":0.85,"neural>=16":0.65}   # quick_peel.py top5% (upper bound)
for b in B:
    n=len(rec["neural_top5"][b])
    if not n: continue
    f=lambda k:np.mean(rec[k][b])
    print(f"{b:<14}{n:>5}{f('neural_top5'):>10.2f}{ora[b]:>12.2f}{f('greedy_seed'):>13.2f}{f('greedy_cos'):>12.2f}")
print(f"\nN5 set-EM (all 5 correct): neural_top5={np.mean(setEM['neural_top5']):.3f}  "
      f"greedy_seed={np.mean(setEM['greedy_seed']):.3f}  greedy_cos={np.mean(setEM['greedy_cos']):.3f}")
print("\nREAD: if greedy_seed recovers the missed buckets (6-15, >=16) WELL ABOVE neural@5 (which is ~0 there),")
print("the explaining-away mechanism survives without ground-truth => recursive decoder FEASIBLE.")
print("If greedy collapses toward neural@5 (or set-EM <= neural), error-compounding kills it => need learned residual scorer / stronger first picks.")

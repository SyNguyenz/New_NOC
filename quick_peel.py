"""
quick_peel.py — NO-TRAIN feasibility of per-donor PEELING for N5. For each true donor d:
subtract the OTHER 4 true donors' contributions (NNLS on reference genotypes) -> residual,
then rank all 45 donors by match to the residual (cosine with reference). If d (esp. neural-BURIED
ones) becomes top-1/top-5 after peeling, the decision is recoverable by decoupling => peeling worth building.
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
def build_any(cfg, st):
    return SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,
        n_classes=45,n_noc=6,dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),
        decoder_source=cfg.get("decoder_source","encoded"),n_token_feats=cfg.get("n_token_feats",8),
        encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
        n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
        aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False)).to(DEVICE)
cfg=json.load(open(Path("results")/arm/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats",8)
st=torch.load(Path("results")/arm/"best_model.pt",map_location=DEVICE,weights_only=True)
model=build_any(cfg,st); model.load_state_dict(st); model.eval()

tt=np.load(D/"tokens_test.npy").astype(np.float32); mt=np.load(D/"mask_test.npy").astype(bool)
yt=np.load(D/"y_test_set.npy").astype(int); nt=np.clip(np.load(D/"noc_test.npy").astype(int),1,5)
Xf=np.load(D/"Xflat_test.npy").astype(np.float64); ent=enrich_tokens(tt,mt)[:,:,:n_tok]
P=[]
with torch.no_grad():
    for i in range(0,len(ent),256):
        P.append(torch.sigmoid(model(torch.from_numpy(ent[i:i+256]).to(DEVICE),torch.from_numpy(mt[i:i+256]).to(DEVICE))["logits_cls"]).cpu().numpy())
P=np.concatenate(P)
G=PG.build_refs(); Gn=G/(np.linalg.norm(G,axis=1,keepdims=True)+1e-9)

# bucket each true donor by neural rank; compare neural rank vs oracle-peel residual rank
buckets={"neural<=5":[], "neural 6-15":[], "neural>=16":[]}
def addb(nr, pr):
    k="neural<=5" if nr<=5 else ("neural 6-15" if nr<=15 else "neural>=16")
    buckets[k].append(pr)
for i in np.where(nt==5)[0]:
    S=list(np.where(yt[i]==1)[0]); hrel=Xf[i]/(Xf[i].sum()+1e-12); order=list(np.argsort(P[i])[::-1])
    for d in S:
        others=[e for e in S if e!=d]
        A=G[others].T; phi,_=nnls(A,hrel); resid=np.clip(hrel-A@phi,0,None)
        if resid.sum()<1e-9: pr=45
        else:
            rn=resid/(np.linalg.norm(resid)+1e-9); sims=Gn@rn; pr=list(np.argsort(sims)[::-1]).index(d)+1
        addb(order.index(d)+1, pr)

print(f"arm={arm}  N5 oracle-peel feasibility (per true donor)\n")
print(f"{'neural-rank bucket':<18}{'n':>5}{'peel med':>10}{'peel top1%':>11}{'peel top5%':>11}")
for k,v in buckets.items():
    v=np.array(v)
    if len(v): print(f"{k:<18}{len(v):>5}{np.median(v):>10.0f}{(v==1).mean():>11.2f}{(v<=5).mean():>11.2f}")
print("\nIf neural>=16 donors get peel top5%/top1% HIGH => decoupling-by-peeling recovers buried donors => peeling FEASIBLE")

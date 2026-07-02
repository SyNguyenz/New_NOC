"""
probe_height.py — for N5, is a true donor BURIED (rank>=16 in neural P) because its private-allele
peak is FAINT (suppression) or DESPITE being tall (decision entanglement)? Compare glob_rel
(height/profile-max) of each true donor's PRESENT private alleles, split by donor rank class.
"""
import json
from pathlib import Path
import numpy as np, torch
import pgnoc as PG
from train_set_transformer import DEVICE
from models.set_transformer import SetTransformerMixture
from features.enrich import enrich_tokens

D = Path("data_insilico_w"); arm = "inc4_p3_irm_seed42"
def build_any(cfg, state):
    kw = dict(n_loci=cfg.get("n_loci",24), d_locus=cfg.get("d_locus",16), d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4), n_isab=cfg.get("n_isab",2), m_inducing=cfg.get("m_inducing",32),
        n_classes=45, n_noc=6, dropout=cfg.get("dropout",0.1), cls_decoder=cfg.get("cls_decoder","pooled"),
        decoder_source=cfg.get("decoder_source","encoded"), n_token_feats=cfg.get("n_token_feats",8),
        encoder=cfg.get("encoder","isab"), dec_layers=cfg.get("dec_layers",2), num_embed=cfg.get("num_embed","raw"),
        n_freq=cfg.get("n_freq",8), d_num_emb=cfg.get("d_num_emb",8), periodic_sigma=cfg.get("periodic_sigma",1.0),
        aux_heads=cfg.get("aux_heads",False), sparse_attn=cfg.get("sparse_attn",False))
    return SetTransformerMixture(**kw).to(DEVICE)
cfg = json.load(open(Path("results")/arm/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats",8)
state = torch.load(Path("results")/arm/"best_model.pt", map_location=DEVICE, weights_only=True)
model = build_any(cfg,state); model.load_state_dict(state); model.eval()

tt=np.load(D/"tokens_test.npy").astype(np.float32); mt=np.load(D/"mask_test.npy").astype(bool)
yt=np.load(D/"y_test_set.npy").astype(int); nt=np.clip(np.load(D/"noc_test.npy").astype(int),1,5)
Xf=np.load(D/"Xflat_test.npy").astype(np.float64)
ent=enrich_tokens(tt,mt)[:,:,:n_tok]
P=[]
with torch.no_grad():
    for i in range(0,len(ent),256):
        P.append(torch.sigmoid(model(torch.from_numpy(ent[i:i+256]).to(DEVICE),torch.from_numpy(mt[i:i+256]).to(DEVICE))["logits_cls"]).cpu().numpy())
P=np.concatenate(P)

G=PG.build_refs()  # (45,590) rel-RFU
geno=[set(np.where(G[d] >= 0.25*G[d].max())[0].tolist()) for d in range(45)]
print(f"avg genotype bins/donor = {np.mean([len(g) for g in geno]):.1f}")

cls = {"recovered(<=5)":[], "nearmiss(6-10)":[], "buried(>=16)":[]}
present_frac = {k:[0,0] for k in cls}   # [present, total]
for i in np.where(nt==5)[0]:
    S=set(np.where(yt[i]==1)[0].tolist()); order=list(np.argsort(P[i])[::-1])
    gmax=Xf[i].max()+1e-9; present_bins=set(np.where(Xf[i]>0)[0].tolist())
    for d in S:
        rank=order.index(d)+1
        others=set().union(*[geno[e] for e in S if e!=d]) if len(S)>1 else set()
        priv=geno[d]-others
        priv_present=priv & present_bins
        gr = max((Xf[i,b]/gmax for b in priv_present), default=0.0)
        if rank<=5: k="recovered(<=5)"
        elif 6<=rank<=10: k="nearmiss(6-10)"
        elif rank>=16: k="buried(>=16)"
        else: continue
        present_frac[k][1]+=1
        if priv_present: present_frac[k][0]+=1; cls[k].append(gr)

print(f"\n{'class':<16}{'n':>5}{'has_priv%':>10}{'med globrel':>13}{'p25':>8}{'p75':>8}")
for k in cls:
    pf=present_frac[k]; arr=np.array(cls[k])
    hp=pf[0]/max(pf[1],1)
    if len(arr): print(f"{k:<16}{pf[1]:>5}{hp:>10.2f}{np.median(arr):>13.3f}{np.percentile(arr,25):>8.3f}{np.percentile(arr,75):>8.3f}")
    else: print(f"{k:<16}{pf[1]:>5}{hp:>10.2f}        (no present-private)")
print("\nburied glob_rel << recovered => FAINT private allele = suppression(attention/pooling); ~equal => DECISION/representation")

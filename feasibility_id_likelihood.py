"""
feasibility_id_likelihood.py — QUICK feasibility of using the biological likelihood to RANK
ID (not just count). For N4/N5 full-test samples, given the arm's neural top-`pool` candidates:
  - recall@K: are the true donors even in the neural pool? (precondition)
  - neural ID-EM @ true-k (oracle count) vs LIKELIHOOD ID-EM = min-fit_cost size-k subset of pool
  - precision-when-recoverable: among samples whose true set ⊆ pool, does likelihood pick it?
If likelihood-EM > neural-EM at N5, biology ranks ID better where neural fails → ID-likelihood worth building.

Usage: python feasibility_id_likelihood.py inc4_p3_irm_seed42 [pool=10]
"""
import sys, json, itertools
from pathlib import Path
import numpy as np, torch

import pgnoc as PG  # __main__-guarded; reuse build_refs + fit_cost
from train_set_transformer import DEVICE
from models.set_transformer import SetTransformerMixture
from features.enrich import enrich_tokens

D = Path("data_insilico_w")
arm = sys.argv[1] if len(sys.argv) > 1 else "inc4_p3_irm_seed42"
POOL = int(sys.argv[2]) if len(sys.argv) > 2 else 10

def build_any(cfg, state):
    kw = dict(n_loci=cfg.get("n_loci",24), d_locus=cfg.get("d_locus",16), d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4), n_isab=cfg.get("n_isab",2), m_inducing=cfg.get("m_inducing",32),
        n_classes=cfg.get("n_classes",45), n_noc=cfg.get("n_noc",6), dropout=cfg.get("dropout",0.1),
        cls_decoder=cfg.get("cls_decoder","pooled"), decoder_source=cfg.get("decoder_source","encoded"),
        n_token_feats=cfg.get("n_token_feats",8), encoder=cfg.get("encoder","isab"), dec_layers=cfg.get("dec_layers",2),
        num_embed=cfg.get("num_embed","raw"), n_freq=cfg.get("n_freq",8), d_num_emb=cfg.get("d_num_emb",8),
        periodic_sigma=cfg.get("periodic_sigma",1.0), aux_heads=cfg.get("aux_heads",False),
        noc_contrast=cfg.get("noc_contrast",False), noc_detach=(cfg.get("noc_contrast_mode","shared")=="detach"),
        d_proj=cfg.get("d_proj",64), sparse_attn=cfg.get("sparse_attn",False),
        geno_query=bool(cfg.get("geno_query")), donor_contrast=cfg.get("donor_contrast",False),
        noc_ord_head=cfg.get("noc_ord_head",False), noc_ord_detach=cfg.get("noc_ord_detach",False),
        noc_ord_replace=cfg.get("noc_ord_replace",False))
    if cfg.get("geno_query"):
        kw["donor_geno"]=torch.zeros_like(state["donor_geno"]).float(); kw["donor_geno_mask"]=torch.zeros_like(state["donor_geno_mask"]).bool()
    return SetTransformerMixture(**kw).to(DEVICE)

cfg = json.load(open(Path("results")/arm/"metrics.json"))["config"]; n_tok = cfg.get("n_token_feats",8)
state = torch.load(Path("results")/arm/"best_model.pt", map_location=DEVICE, weights_only=True)
model = build_any(cfg, state); model.load_state_dict(state); model.eval()

t = np.load(D/"tokens_test.npy").astype(np.float32); mk = np.load(D/"mask_test.npy").astype(bool)
en = enrich_tokens(t, mk)[:, :, :n_tok]
P = []
with torch.no_grad():
    for i in range(0, len(en), 256):
        P.append(torch.sigmoid(model(torch.from_numpy(en[i:i+256]).to(DEVICE), torch.from_numpy(mk[i:i+256]).to(DEVICE))["logits_cls"]).cpu().numpy())
P = np.concatenate(P)
y = np.load(D/"y_test_set.npy").astype(int); noc = np.clip(np.load(D/"noc_test.npy").astype(int),1,5)
H = np.expm1(np.load(D/"Xflat_test.npy").astype(np.float64))
G = PG.build_refs()

print(f"arm={arm} pool={POOL}\n")
print(f"{'NOC':>4}{'n':>5}{'rec@8':>7}{'rec@10':>8}{'rec@16':>8}{'inPool':>8}{'neurEM':>8}{'likEM':>7}{'likEM|inPool':>13}")
for k in [4, 5]:
    idx = np.where(noc == k)[0]
    rec8=rec10=rec16=inp=neur=lik=lik_ip=0; n_ip=0
    for i in idx:
        true = set(np.where(y[i]==1)[0].tolist())
        order = np.argsort(P[i])[::-1]
        rec8  += len(true & set(order[:8].tolist()))/k
        rec10 += len(true & set(order[:10].tolist()))/k
        rec16 += len(true & set(order[:16].tolist()))/k
        pool = order[:POOL].tolist()
        in_pool = true <= set(pool); inp += in_pool; n_ip += in_pool
        neur += (set(order[:k].tolist()) == true)
        best, bc = None, np.inf
        for comb in itertools.combinations(pool, k):
            c = PG.fit_cost(H[i], G, list(comb))
            if c < bc: bc, best = c, comb
        hit = (set(best) == true); lik += hit
        if in_pool: lik_ip += hit
    n = len(idx)
    print(f"{k:>4}{n:>5}{rec8/n:>7.3f}{rec10/n:>8.3f}{rec16/n:>8.3f}{inp/n:>8.3f}{neur/n:>8.3f}{lik/n:>7.3f}{(lik_ip/max(n_ip,1)):>13.3f}")
print("\nlegend: neurEM=neural top-k(true) ID-EM | likEM=min-fit_cost size-k subset of pool | likEM|inPool=likelihood precision when true⊆pool")

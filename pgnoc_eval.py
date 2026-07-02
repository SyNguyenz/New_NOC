"""
pgnoc_eval.py — run pgNOC (continuous gamma+dropout likelihood, EuroForMix/NOCIt-faithful)
on a GIVEN per_donor checkpoint, on the FULL real test. Unlike pgnoc.py (pinned to the stale
hybrid_50k_weight ckpt), this builds the requested arm robustly (sparse/geno_query/noc_contrast
aware) and uses ITS neural per-donor probs for the candidate pool. Pure-likelihood helpers
(build_refs/fit_cost/estimate_noc/tune_penalty) reused from pgnoc.py (imported, __main__-guarded).

Non-destructive: writes results/<arm>/pgnoc_fulltest.json only. Touches nothing else.

Usage: python pgnoc_eval.py inc2_2b_privsup inc2_2c_fix_sh inc4_p3_irm
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import pgnoc as PG  # __main__-guarded: safe import; reuse build_refs/fit_cost/estimate_noc/tune_penalty
from train_set_transformer import topk_decode, per_noc_em, DEVICE
from models.set_transformer import SetTransformerMixture
from features.enrich import enrich_tokens

D = Path("data_insilico_w")
KMAX, ar = PG.KMAX, np.arange(1, PG.KMAX + 1)

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
        kw["donor_geno"]=torch.zeros_like(state["donor_geno"]).float()
        kw["donor_geno_mask"]=torch.zeros_like(state["donor_geno_mask"]).bool()
    return SetTransformerMixture(**kw).to(DEVICE)

def load(arm):
    ck = Path("results")/arm; cfg = json.load(open(ck/"metrics.json"))["config"]
    state = torch.load(ck/"best_model.pt", map_location=DEVICE, weights_only=True)
    m = build_any(cfg, state); m.load_state_dict(state); m.eval()
    return m, cfg.get("n_token_feats", 8)

def neural_P(model, n_tok, split):
    t = np.load(D/f"tokens_{split}.npy").astype(np.float32); mk = np.load(D/f"mask_{split}.npy").astype(bool)
    en = enrich_tokens(t, mk)[:, :, :n_tok]
    P = []
    with torch.no_grad():
        for i in range(0, len(en), 256):
            P.append(torch.sigmoid(model(torch.from_numpy(en[i:i+256]).to(DEVICE),
                     torch.from_numpy(mk[i:i+256]).to(DEVICE))["logits_cls"]).cpu().numpy())
    return np.concatenate(P)

def cost_matrix_P(P, split, G):
    H = np.expm1(np.load(D/f"Xflat_{split}.npy").astype(np.float64))
    C = np.zeros((len(P), KMAX))
    for i in range(len(P)):
        _, C[i] = PG.estimate_noc(H[i], P[i], G)
    return C

nva = np.load(D/"noc_val.npy").astype(int); nte = np.clip(np.load(D/"noc_test.npy").astype(int),1,5)
yte = np.load(D/"y_test_set.npy")
G = PG.build_refs()
print(f"full test n={len(nte)}  per-NOC={[int((nte==k).sum()) for k in range(1,6)]}\n")

for arm in sys.argv[1:]:
    model, n_tok = load(arm)
    Pva = neural_P(model, n_tok, "val"); Pte = neural_P(model, n_tok, "test")
    Cva = cost_matrix_P(Pva, "val", G); Cte = cost_matrix_P(Pte, "test", G)
    pen, acc_va = PG.tune_penalty(Cva, nva)
    k_star = (Cte + pen*ar).argmin(1) + 1
    cnt = {k: float((k_star[nte==k]==k).mean()) for k in [1,2,3,4,5]}
    em = per_noc_em(yte, topk_decode(Pte, k_star), nte)
    orc = per_noc_em(yte, topk_decode(Pte, nte), nte)
    m45 = np.isin(nte,[4,5]); auc = roc_auc_score((nte[m45]==5).astype(int), (Cte[:,3]-Cte[:,4])[m45])
    print(f"=== {arm}  (pen={pen:.4f}, val acc {acc_va:.3f}) ===")
    print(f"  {'':10}{'all':>7}{'N1':>7}{'N2':>7}{'N3':>7}{'N4':>7}{'N5':>7}")
    print(f"  {'oracle':10}" + "".join(f"{x:>7.3f}" for x in orc))
    print(f"  {'pgNOC-EM':10}" + "".join(f"{x:>7.3f}" for x in em))
    print(f"  {'count':10}{(k_star==nte).mean():>7.3f}" + "".join(f"{cnt[k]:>7.3f}" for k in [1,2,3,4,5]))
    print(f"  AUC(4v5) NLL-gain 5th donor = {auc:.3f}\n")
    out = {"arm":arm, "test_n":int(len(nte)), "penalty":float(pen), "count_acc":float((k_star==nte).mean()),
           "count_per_noc":cnt, "pgNOC_em":[float(x) for x in em], "oracle_em":[float(x) for x in orc],
           "auc_4v5":float(auc), "note":"pgNOC on FULL real test, arm's own per-donor probs (not hybrid)"}
    json.dump(out, open(Path("results")/arm/"pgnoc_fulltest.json","w"), indent=2)
    print(f"  wrote results/{arm}/pgnoc_fulltest.json\n")

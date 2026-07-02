"""
quick_gain_features.py — does the likelihood cost-curve gain_k ADD count signal over the
neural count head, for EVERY NOC transition (not just 4v5)? For each adjacent pair k vs k+1
(true_noc in {k,k+1}) on the FULL test: AUC of neural card score, AUC of gain_{k+1}, their
correlation, and AUC of a 2-feature logistic (incremental value). Cheap: cost curve on test only.
"""
import json
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import pgnoc as PG
from train_set_transformer import DEVICE
from models.set_transformer import SetTransformerMixture
from features.enrich import enrich_tokens
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

D = Path("data_insilico_w"); arm = "inc4_p3_irm_seed42"
cfg = json.load(open(Path("results")/arm/"metrics.json"))["config"]; n_tok = cfg.get("n_token_feats",8)
state = torch.load(Path("results")/arm/"best_model.pt", map_location=DEVICE, weights_only=True)
model = build_any(cfg, state); model.load_state_dict(state); model.eval()

t = np.load(D/"tokens_test.npy").astype(np.float32); mk = np.load(D/"mask_test.npy").astype(bool)
en = enrich_tokens(t, mk)[:, :, :n_tok]
P, CARD = [], []
with torch.no_grad():
    for i in range(0, len(en), 256):
        o = model(torch.from_numpy(en[i:i+256]).to(DEVICE), torch.from_numpy(mk[i:i+256]).to(DEVICE))
        P.append(torch.sigmoid(o["logits_cls"]).cpu().numpy()); CARD.append(o["logits_card"].cpu().numpy())
P = np.concatenate(P); CARD = np.concatenate(CARD)  # CARD[:,j] = logit NOC=j+1
noc = np.clip(np.load(D/"noc_test.npy").astype(int),1,5)
H = np.expm1(np.load(D/"Xflat_test.npy").astype(np.float64)); G = PG.build_refs()

cost = np.zeros((len(P), PG.KMAX))
for i in range(len(P)):
    _, cost[i] = PG.estimate_noc(H[i], P[i], G)
gain = cost[:, :-1] - cost[:, 1:]   # gain[:,j] = cost[k=j+1]-cost[k=j+2] = adding (j+2)-th donor

print(f"arm={arm}  full test\n")
print(f"{'pair':>7}{'n':>6}{'AUC_neural':>12}{'AUC_gain':>10}{'corr':>8}{'AUC_both':>10}")
for k in [1, 2, 3, 4]:                       # NOC=k vs k+1
    m = np.isin(noc, [k, k+1]); lab = (noc[m] == k+1).astype(int)
    s_neu = CARD[m, k] - CARD[m, k-1]        # neural: logit(NOC=k+1) - logit(NOC=k)
    s_gain = gain[m, k-1]                     # gain of adding (k+1)-th donor = cost[k]-cost[k+1]
    auc_n = roc_auc_score(lab, s_neu); auc_g = roc_auc_score(lab, s_gain)
    corr = np.corrcoef(s_neu, s_gain)[0, 1]
    X = np.c_[(s_neu-s_neu.mean())/s_neu.std(), (s_gain-s_gain.mean())/s_gain.std()]
    lr = LogisticRegression().fit(X, lab); auc_b = roc_auc_score(lab, lr.predict_proba(X)[:, 1])
    print(f"{k}v{k+1:<5}{m.sum():>6}{auc_n:>12.3f}{auc_g:>10.3f}{corr:>8.2f}{auc_b:>10.3f}")
print("\nAUC_both (2-feat logistic, in-sample) > max(neural,gain) => gain adds orthogonal count signal")

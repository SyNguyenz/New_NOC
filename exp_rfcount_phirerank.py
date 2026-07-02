"""
exp_rfcount_phirerank.py — does RF post-hoc count on top of the phi-rerank RANKING beat the
model's own ranking?  Local CPU inference on the LIVE checkpoint (inc22_fixed_aslot_seed42).

Decode = top-k(score) with k from RandomForest count (posthoc_cardinality, fit on val probs).
We compare ranking = {model probs P} vs {phi-reranked score} under the SAME RF count, and report
oracle-k (true k) as the ceiling.  Nothing here is trained — pure eval + numpy rerank.
"""
import json, numpy as np, torch
from models.set_transformer import SetTransformerMixture
import phi_rerank as pr
from train_set_transformer import posthoc_cardinality, per_noc_em

DEVICE = "cpu"; RUN = "results/inc22_fixed_aslot_seed42"
cfg = json.load(open(RUN + "/metrics.json"))["config"]
def LD(f): return np.load("data_insilico_w/%s.npy" % f)
Xt, Mt, yt, nt = LD("tokens8_test"), LD("mask_test").astype(bool), LD("y_test_set"), LD("noc_test")
Xv, Mv, yv, nv = LD("tokens8_val"),  LD("mask_val").astype(bool),  LD("y_val_set"),  LD("noc_val")
g  = np.load("data/donor_geno.npy").astype(np.float32); gmask = np.load("data/donor_geno_mask.npy")

# owner_lut exactly as training builds it (line ~799)
ALLELE_OFF, n_cls, LUT_W = 30, int(cfg.get("n_classes", 45)), 1024
owner_lut = torch.zeros(24, LUT_W, n_cls); gm = torch.from_numpy(gmask).bool()
for c in range(min(n_cls, g.shape[0])):
    for j in range(g.shape[1]):
        if gm[c, j]:
            li = int(g[c, j, 0]); ab = int(round(float(g[c, j, 1]) * 10)) + ALLELE_OFF
            if 0 <= li < 24 and 0 <= ab < LUT_W: owner_lut[li, ab, c] = 1.0

n_tok = int(cfg.get("n_token_feats", 8))
model = SetTransformerMixture(
    n_loci=cfg.get("n_loci",24), d_locus=cfg.get("d_locus",16), d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4), n_isab=cfg.get("n_isab",2), m_inducing=cfg.get("m_inducing",32),
    n_classes=n_cls, n_noc=cfg.get("n_noc",6), dropout=cfg.get("dropout",0.1),
    cls_decoder=cfg.get("cls_decoder","pooled"), decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok, encoder=cfg.get("encoder","isab"), dec_layers=cfg.get("dec_layers",2),
    dec_aggr=cfg.get("dec_aggr","sparsemax"), num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8), d_num_emb=cfg.get("d_num_emb",8), periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False), ml_attr=cfg.get("ml_attr",False),
    sparse_attn=cfg.get("sparse_attn",False), geno_query=cfg.get("geno_query",False),
    donor_geno=torch.from_numpy(g), donor_geno_mask=torch.from_numpy(gmask),
    ref_match=cfg.get("ref_match",False), nc_attn=cfg.get("nc_attn","none"),
    soft_geno_attr=cfg.get("soft_geno_attr",False), feas_filter=cfg.get("feas_filter",False),
    set_of_set=cfg.get("set_of_set",False), owner_lut=owner_lut,
    n_slot_iters=int(cfg.get("n_slot_iters",3)), ot_eps=float(cfg.get("ot_eps",0.05)),
    ot_iters=int(cfg.get("ot_iters",5)), gumbel_temp=float(cfg.get("gumbel_temp",1.0)),
    noc_head_v2=bool(cfg.get("noc_head_v2",False)), em_phi_feature=bool(cfg.get("em_phi_feature",False)),
).to(DEVICE)
miss, unexp = model.load_state_dict(torch.load(RUN+"/best_model.pt", weights_only=True, map_location=DEVICE), strict=False)
print(f"loaded checkpoint (missing={len(miss)} unexpected={len(unexp)})"); model.eval()

@torch.no_grad()
def infer(X, M):
    P, Lg = [], []
    for s in range(0, len(X), 256):
        out = model(torch.tensor(X[s:s+256]), torch.tensor(M[s:s+256]))
        lg = out["logits_cls"]; Lg.append(lg.numpy()); P.append(torch.sigmoid(lg).numpy())
    return np.concatenate(P), np.concatenate(Lg)

P_te, L_te = infer(Xt, Mt); P_va, L_va = infer(Xv, Mv)
print(f"inferred P_te {P_te.shape}  P_va {P_va.shape}")

# RF post-hoc count (fit on val probs), and phi-rerank ranking
k_rf  = posthoc_cardinality(P_va, yv, P_te)
PH_te = pr.deconv_phi(Xt, Mt, g, gmask, n_iters=12)
PH_va = pr.deconv_phi(Xv, Mv, g, gmask, n_iters=12)
alpha = pr.tune_alpha(L_va, PH_va, yv, nv)
rank_te = pr.rerank_scores(L_te, PH_te, alpha)
print(f"RF count acc (test) = {(k_rf == nt.clip(1,5)).mean():.4f} | tuned alpha = {alpha}")

C = n_cls; yti = (yt > 0.5).astype(int); ktrue = yti.sum(1)
def decode(score, k):
    yp = np.zeros((len(score), C), int)
    for i in range(len(score)): yp[i, np.argsort(score[i])[::-1][:int(k[i])]] = 1
    return yp
def report(name, score, k):
    em = per_noc_em(yti, decode(score, k), nt)
    print(f"{name:30s} overall {em[0]:.4f} | N3 {em[3]:.4f}  N4 {em[4]:.4f}  N5 {em[5]:.4f}")

print("\n=== decode comparison (rows: ranking x count) ===")
report("P-rank   + RF-count",   P_te,    k_rf)
report("phi-rank + RF-count",   rank_te, k_rf)
report("P-rank   + ORACLE-k",   P_te,    ktrue)
report("phi-rank + ORACLE-k",   rank_te, ktrue)

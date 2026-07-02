"""
probes_n5.py — 3 no-training feasibility probes on an existing checkpoint (p3), full test, NOC5:
  (1) rank of the MISSED true donor in neural P (peeling/calibration headroom)
  (2) oracle accuracy vs #shared-donors with nearest TRAIN combo (meta/coverage = distance-driven?)
  (3) kNN in pooled embedding z: do nearest train-N5 neighbors share donors? (data-as-region viable?)
"""
import json
from pathlib import Path
import numpy as np, torch
from train_set_transformer import topk_decode, DEVICE
from models.set_transformer import SetTransformerMixture
from features.enrich import enrich_tokens

D = Path("data_insilico_w"); arm = "inc4_p3_irm_seed42"

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
    return SetTransformerMixture(**kw).to(DEVICE)

cfg = json.load(open(Path("results")/arm/"metrics.json"))["config"]; n_tok = cfg.get("n_token_feats",8)
state = torch.load(Path("results")/arm/"best_model.pt", map_location=DEVICE, weights_only=True)
model = build_any(cfg, state); model.load_state_dict(state); model.eval()

def fwd_P(tok_en, mask):
    P = []
    with torch.no_grad():
        for i in range(0, len(tok_en), 256):
            P.append(torch.sigmoid(model(torch.from_numpy(tok_en[i:i+256]).to(DEVICE),
                     torch.from_numpy(mask[i:i+256]).to(DEVICE))["logits_cls"]).cpu().numpy())
    return np.concatenate(P)

def fwd_Z(tok_en, mask):
    Z = []
    with torch.no_grad():
        for i in range(0, len(tok_en), 256):
            tk = torch.from_numpy(tok_en[i:i+256]).to(DEVICE); mk = torch.from_numpy(mask[i:i+256]).to(DEVICE)
            _, H, pad = model._encode_set(tk, mk)
            Z.append(model.pma(H, pad_mask=pad).squeeze(1).cpu().numpy())
    return np.concatenate(Z)

# ---- test ----
tt = np.load(D/"tokens_test.npy").astype(np.float32); mt = np.load(D/"mask_test.npy").astype(bool)
yt = np.load(D/"y_test_set.npy").astype(int); nt = np.clip(np.load(D/"noc_test.npy").astype(int),1,5)
ent = enrich_tokens(tt, mt)[:, :, :n_tok]
P = fwd_P(ent, mt)
i5 = np.where(nt == 5)[0]

# ===== Probe 1: rank of missed true donor =====
print("== Probe 1: rank of MISSED true donor (N5 oracle-misses) ==")
ranks = []
for i in i5:
    true = set(np.where(yt[i]==1)[0].tolist())
    order = list(np.argsort(P[i])[::-1])
    top5 = set(order[:5])
    if top5 == true: continue
    for d in (true - top5):
        ranks.append(order.index(d) + 1)
ranks = np.array(ranks)
print(f"  missed donors={len(ranks)} over {len(i5)} N5 samples")
for lo,hi in [(6,6),(7,8),(9,10),(11,15),(16,45)]:
    c = ((ranks>=lo)&(ranks<=hi)).sum()
    print(f"    rank {lo:>2}-{hi:<2}: {c:>4}  ({c/max(len(ranks),1):.2f})")
print(f"  median missed-rank = {np.median(ranks):.0f}")

# ===== Probe 2: oracle acc vs #shared with nearest train combo =====
ytr = np.load(D/"y_train_set.npy").astype(int); ntr = np.clip(np.load(D/"noc_train.npy").astype(int),1,5)
train5 = [frozenset(np.where(ytr[j]==1)[0].tolist()) for j in np.where(ntr==5)[0]]
train5u = list(set(train5))
em5 = (topk_decode(P[i5], nt[i5]) == yt[i5]).all(1)
shared = []
for k, i in enumerate(i5):
    true = set(np.where(yt[i]==1)[0].tolist())
    shared.append(max(len(true & c) for c in train5u))
shared = np.array(shared)
print("\n== Probe 2: N5 oracle EM vs max #donors shared with nearest TRAIN combo ==")
print(f"  (train has {len(train5u)} unique N5 combos)")
for s in [0,1,2,3,4]:
    m = shared == s
    if m.sum(): print(f"    shared={s}: n={m.sum():>4}  oracleEM={em5[m].mean():.3f}")

# ===== Probe 3: kNN in pooled z =====
print("\n== Probe 3: kNN in embedding z (test N5 -> nearest TRAIN N5) ==")
rng = np.random.default_rng(0)
tr5idx = np.where(ntr==5)[0]; sub = rng.choice(tr5idx, size=min(5000,len(tr5idx)), replace=False)
ttr = np.load(D/"tokens_train.npy")[sub].astype(np.float32); mtr = np.load(D/"mask_train.npy")[sub].astype(bool)
entr = enrich_tokens(ttr, mtr)[:, :, :n_tok]
Ztr = fwd_Z(entr, mtr); Zte = fwd_Z(ent[i5], mt[i5])
def norm(Z): return Z / (np.linalg.norm(Z,axis=1,keepdims=True)+1e-9)
Ztr_n, Zte_n = norm(Ztr), norm(Zte)
sims = Zte_n @ Ztr_n.T                      # cosine
nn = sims.argmax(1)
tr_combos = [set(np.where(ytr[j]==1)[0].tolist()) for j in sub]
nn_shared, rand_shared = [], []
for k, i in enumerate(i5):
    true = set(np.where(yt[i]==1)[0].tolist())
    nn_shared.append(len(true & tr_combos[nn[k]]))
    rand_shared.append(len(true & tr_combos[rng.integers(len(sub))]))
print(f"  mean #shared donors: nearest-z neighbor = {np.mean(nn_shared):.2f}  | random train = {np.mean(rand_shared):.2f}")
print(f"  (max possible 4 since all test N5 combos novel; higher nearest>>random => z clusters by donor content)")

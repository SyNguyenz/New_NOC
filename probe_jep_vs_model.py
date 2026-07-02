"""
Decisive probe: is the per-donor JEP-signature score ADDITIVE to the trained model on the FULL N5 set?
  - per-donor AUC (present vs absent) over ALL N5 donors: model vs signature vs combined (untuned z-sum)
  - N5 oracle: model alone vs model + alpha*signature (alpha tuned on real VAL, eval on real TEST) -> headroom
  - complementarity: of N5 contributors the MODEL MISSES (true present but ranked outside top-k),
    what fraction does the signature score rank into top-k? (and vice versa)

No privileged info in features (signatures = observed peaks + panel only). GT used only to score.
Leakage-safe: alpha selected on VAL, reported on TEST (avoids the F34 synthetic re-rank artifact).
"""
import os, json, itertools
from pathlib import Path
import numpy as np, torch

DATA = Path(os.environ.get("STR_DATA_DIR", "data_insilico_w"))
RUN  = Path(os.environ.get("RUN", "results/inc6_maskp_seed42"))
GENO = Path(os.environ.get("STR_GENO", "data/donor_geno.npy"))
MAX_ORDER, TOPK = 3, 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def abin(a): return int(round(float(a) * 10))
def key(l, a): return (int(round(float(l))), abin(a))

# ───────── mine minimal JEPs (panel-only) ─────────
g  = np.load(GENO); gm = np.load(str(GENO).replace(".npy", "_mask.npy")).astype(bool); C = g.shape[0]
donor_items = [set(key(g[c, j, 0], g[c, j, 1]) for j in range(g.shape[1]) if gm[c, j]) for c in range(C)]
owners = {}
for c in range(C):
    for it in donor_items[c]: owners.setdefault(it, set()).add(c)
def rarity(it): return float(np.log(C / len(owners[it])))
def mine(c):
    A = sorted(donor_items[c]); priv = [it for it in A if owners[it] == {c}]
    sigs = [(it,) for it in priv]; nonp = [it for it in A if owners[it] != {c}]
    if MAX_ORDER >= 2:
        cc = [(x, y, rarity(x)+rarity(y)) for x, y in itertools.combinations(nonp, 2)
              if not ((owners[x] & owners[y]) - {c})]
        cc.sort(key=lambda t: -t[2]); sigs += [(x, y) for x, y, _ in cc[:TOPK]]
    if MAX_ORDER >= 3:
        cc = []
        for x, y, z in itertools.combinations(nonp, 3):
            if not ((owners[x]&owners[y])-{c}): continue
            if not ((owners[x]&owners[z])-{c}): continue
            if not ((owners[y]&owners[z])-{c}): continue
            if not ((owners[x]&owners[y]&owners[z])-{c}): cc.append((x, y, z, rarity(x)+rarity(y)+rarity(z)))
        cc.sort(key=lambda t: -t[3]); sigs += [(x, y, z) for x, y, z, _ in cc[:TOPK]]
    return [(s, sum(rarity(i) for i in s)) for s in sigs]
JEP = [mine(c) for c in range(C)]

def sig_score(tok, msk):
    N = tok.shape[0]; S = np.zeros((N, C))
    for i in range(N):
        v = msk[i].astype(bool)
        O = set(key(tok[i, k, 0], tok[i, k, 1]) for k in np.where(v)[0])
        for c in range(C):
            tot = 0.0
            for items, w in JEP[c]:
                if all(it in O for it in items): tot += w
            S[i, c] = tot
    return S

# ───────── model probs ─────────
from models.set_transformer import SetTransformerMixture
cfg = json.load(open(RUN / "metrics.json"))["config"]
n_tok = cfg.get("n_token_feats", 8); tp = f"tokens{n_tok}" if n_tok > 3 else "tokens"
def load(split):
    return (np.load(DATA/f"{tp}_{split}.npy").astype(np.float32), np.load(DATA/f"mask_{split}.npy"),
            np.load(DATA/f"y_{split}_set.npy").astype(bool), np.load(DATA/f"noc_{split}.npy").astype(int))
dg = dgm = None
model = SetTransformerMixture(
    n_loci=cfg.get("n_loci",24), d_locus=cfg.get("d_locus",16), d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4), n_isab=cfg.get("n_isab",2), m_inducing=cfg.get("m_inducing",32),
    n_classes=cfg.get("n_classes",45), n_noc=cfg.get("n_noc",6), dropout=cfg.get("dropout",0.1),
    cls_decoder=cfg.get("cls_decoder","pooled"), decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok, encoder=cfg.get("encoder","isab"), dec_layers=cfg.get("dec_layers",2),
    num_embed=cfg.get("num_embed","raw"), n_freq=cfg.get("n_freq",8), d_num_emb=cfg.get("d_num_emb",8),
    periodic_sigma=cfg.get("periodic_sigma",1.0), aux_heads=cfg.get("aux_heads",False),
    sparse_attn=cfg.get("sparse_attn",False),
).to(DEVICE)
sd = torch.load(RUN/"best_model.pt", map_location=DEVICE); sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
model.load_state_dict(sd, strict=False); model.eval()
@torch.no_grad()
def probs(tok, msk):
    P = []
    for i in range(0, len(tok), 256):
        o = model(torch.from_numpy(tok[i:i+256]).to(DEVICE), torch.from_numpy(msk[i:i+256].astype(bool)).to(DEVICE))
        P.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
    return np.concatenate(P)

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([pos, neg]); _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0; r = avg[inv]
    return (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

def z(a):
    s = a.std(); return (a - a.mean()) / (s if s > 1e-9 else 1.0)

# ───────── evaluate ─────────
tk_te, mk_te, y_te, noc_te = load("test")
tk_va, mk_va, y_va, noc_va = load("val")
print(f"run={RUN.name}  test N5={int((noc_te==5).sum())}  val NOC dist={np.bincount(np.clip(noc_va,1,5))[1:]}")
P_te = probs(tk_te, mk_te); S_te = sig_score(tk_te, mk_te)
P_va = probs(tk_va, mk_va); S_va = sig_score(tk_va, mk_va)

def donor_auc(P, S, y, noc, k):
    sel = np.where(noc == k)[0]
    pm, nm, ps, ns, pc, nc = [], [], [], [], [], []
    for i in sel:
        # per-sample z-normalised combine so neither feature's scale dominates
        cm = z(np.log(P[i]+1e-9) - np.log(1-P[i]+1e-9)) + z(S[i])
        for c in range(C):
            (pm if y[i,c] else nm).append(P[i,c])
            (ps if y[i,c] else ns).append(S[i,c])
            (pc if y[i,c] else nc).append(cm[c])
    return auc(pm,nm), auc(ps,ns), auc(pc,nc), len(sel)

print("\n── per-donor AUC (present vs absent), by NOC ──")
print(f"{'NOC':>4} {'model':>7} {'signat':>7} {'combo':>7}")
for k in [1,2,3,4,5]:
    am, as_, ac, n = donor_auc(P_te, S_te, y_te, noc_te, k)
    print(f"{k:>4} {am:>7.3f} {as_:>7.3f} {ac:>7.3f}")

# ── N5 oracle: model alone vs model + alpha*sig ; alpha tuned on VAL (all NOC available) ──
def oracle(P, S, y, noc, k, alpha):
    sel = np.where(noc == k)[0]; hits = 0
    for i in sel:
        score = (np.log(P[i]+1e-9)-np.log(1-P[i]+1e-9))
        if alpha != 0: score = z(score) + alpha*z(S[i])
        top = np.argsort(score)[::-1][:k]; pred = np.zeros(C, int); pred[top] = 1
        hits += int((pred == y[i]).all())
    return hits/max(1,len(sel))

alphas = [0,0.1,0.2,0.3,0.5,0.75,1.0,1.5]
# select alpha on VAL using whatever NOC it has (weight high-NOC); fall back to N>=3
val_k = [k for k in [5,4,3] if (noc_va==k).any()]
best_a, best_v = 0, -1
for a in alphas:
    v = np.mean([oracle(P_va,S_va,y_va,noc_va,k,a) for k in val_k])
    if v > best_v: best_v, best_a = v, a
print(f"\n── N5 oracle (alpha-sweep on TEST; VAL-selected alpha={best_a} from NOC{val_k}) ──")
print("  alpha:  " + "  ".join(f"{a:>4}" for a in alphas))
print("  N5 orc: " + "  ".join(f"{oracle(P_te,S_te,y_te,noc_te,5,a):.3f}" for a in alphas))
print(f"  >> model-alone N5 oracle = {oracle(P_te,S_te,y_te,noc_te,5,0):.3f}  |  "
      f"VAL-selected alpha={best_a} -> N5 oracle = {oracle(P_te,S_te,y_te,noc_te,5,best_a):.3f}")
for k in [4,3]:
    print(f"     (guard N{k}: model={oracle(P_te,S_te,y_te,noc_te,k,0):.3f} -> "
          f"alpha={best_a}: {oracle(P_te,S_te,y_te,noc_te,k,best_a):.3f})")

# ── complementarity on N5: of model-MISSED present donors, does signature rank them top-5? ──
sel5 = np.where(noc_te==5)[0]; m_miss_s_hit = m_miss = s_miss_m_hit = s_miss = 0
for i in sel5:
    topP = set(np.argsort(P_te[i])[::-1][:5]); topS = set(np.argsort(S_te[i])[::-1][:5])
    for c in np.where(y_te[i])[0]:
        if c not in topP:                       # model misses this true contributor
            m_miss += 1; m_miss_s_hit += int(c in topS)
        if c not in topS:                       # signature misses
            s_miss += 1; s_miss_m_hit += int(c in topP)
print(f"\n── N5 complementarity ──")
print(f"  model-missed true contributors recovered by signature top5: {m_miss_s_hit}/{m_miss} = {m_miss_s_hit/max(1,m_miss):.3f}")
print(f"  signature-missed true contributors recovered by model  top5: {s_miss_m_hit}/{s_miss} = {s_miss_m_hit/max(1,s_miss):.3f}")

"""
Step-by-step inference debugger for the inc6_maskp_seed42 checkpoint.

Goal: find WHERE the model loses N5 (5-contributor) mixtures at oracle top-k.
Prints actual numbers at each stage and decomposes the failure into:
  (A) reproduce the headline oracle EM (validate the pipeline matches metrics.json)
  (B) characterise the missed donor (phi rank, present alleles, prob, rank)
  (C) ENCODER vs DECODER decomposition:
        decoder readout  = logits_cls  (per-donor sparsemax cross-attn)
        encoder readout  = attr_head soft-vote over peaks (a DIFFERENT readout of the SAME H)
      + per-peak attribution: does the encoder keep a missed donor's private peak,
        or is it absorbed into a major donor?
"""
import os, json, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_insilico_w"
CKPT = ROOT / "results" / "inc6_maskp_seed42"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)

from models.set_transformer import SetTransformerMixture, sparsemax

# ── load data ────────────────────────────────────────────────────────────────
def L(n): return np.load(DATA / f"{n}.npy", allow_pickle=True)
tokens = L("tokens8_test").astype(np.float32)      # (B,160,8)
mask   = L("mask_test").astype(bool)               # (B,160)
y      = L("y_test_set").astype(np.float32)        # (B,45)
noc    = L("noc_test").astype(int)                 # (B,)
attr   = L("attr_test").astype(int)                # (B,160) per-peak donor (-1 = background)
phi    = L("phi_test").astype(np.float32)          # (B,45) mixture proportion
B = len(tokens)
print(f"test: {B} samples | NOC dist " +
      ", ".join(f"{k}:{(noc==k).sum()}" for k in sorted(set(noc.tolist()))))

# ── build model exactly as train() does (config from metrics.json) ────────────
cfg = json.load(open(CKPT / "metrics.json"))["config"]
model = SetTransformerMixture(
    n_loci=cfg["n_loci"], d_locus=cfg["d_locus"], d_model=cfg["d_model"],
    n_heads=cfg["n_heads"], n_isab=cfg["n_isab"], m_inducing=cfg["m_inducing"],
    n_classes=cfg["n_classes"], n_noc=cfg["n_noc"], dropout=cfg["dropout"],
    cls_decoder=cfg["cls_decoder"], n_token_feats=cfg["n_token_feats"],
    encoder=cfg["encoder"], num_embed=cfg["num_embed"],
    periodic_sigma=cfg["periodic_sigma"], aux_heads=cfg["aux_heads"],
    sparse_attn=cfg["sparse_attn"],
).to(DEV)
sd = torch.load(CKPT / "best_model.pt", weights_only=True, map_location=DEV)
missing, unexpected = model.load_state_dict(sd, strict=False)
model.eval()
print(f"loaded ckpt | missing={list(missing)} unexpected={list(unexpected)}")
print(f"feat_mean={model.feat_mean.cpu().numpy().round(3)}")

# ── forward, capturing H (encoded), logits_cls (decoder), attr_logits (enc readout)
@torch.no_grad()
def run():
    PROB, ATTRL, HMAX = [], [], []   # probs, attr softmax over donors, per-peak max info
    for s in range(0, B, 256):
        tk = torch.from_numpy(tokens[s:s+256]).to(DEV)
        mk = torch.from_numpy(mask[s:s+256]).to(DEV)
        x0, H, pad = model._encode_set(tk, mk)
        logits_cls = model.cls_decoder_module(H, pad_mask=pad)      # decoder readout
        attr_logits = model.attr_head(H)                            # (b,N,46) encoder readout
        PROB.append(torch.sigmoid(logits_cls).cpu().numpy())
        # per-peak donor attribution prob (softmax over 45 donor classes, drop bg class 45)
        ap = torch.softmax(attr_logits[:, :, :45], dim=-1).cpu().numpy()
        ATTRL.append(ap)
    return np.concatenate(PROB), np.concatenate(ATTRL)

probs, attr_prob = run()   # (B,45), (B,160,45)
print("forward done. probs", probs.shape, "attr_prob", attr_prob.shape)

# ── (A) reproduce oracle EM per NOC ───────────────────────────────────────────
def oracle_em_per_noc(P):
    out = {}
    for k in range(1, 6):
        idx = np.where(noc == k)[0]
        if not len(idx): continue
        ok = []
        for i in idx:
            top = np.argsort(P[i])[::-1][:k]
            pred = np.zeros(45, int); pred[top] = 1
            ok.append((pred == y[i]).all())
        out[k] = np.mean(ok)
    return out

dec_oracle = oracle_em_per_noc(probs)
print("\n=== (A) DECODER oracle EM per NOC (reproduce headline) ===")
for k, v in dec_oracle.items(): print(f"   NOC{k}: {v:.4f}")
print("   (metrics.json test_oracle N5 = 0.7876 — must match NOC5 above)")

# ── (C1) ENCODER readout: attr_head soft-vote ─────────────────────────────────
# encoder ceiling = rank donors by max/sum of per-peak attribution prob over valid peaks
def encoder_vote(reduce="max"):
    sc = np.zeros((B, 45))
    for i in range(B):
        v = mask[i]
        ap = attr_prob[i][v]                       # (n_valid, 45)
        sc[i] = ap.max(0) if reduce == "max" else ap.sum(0)
    return sc

enc_max = encoder_vote("max")
enc_sum = encoder_vote("sum")
enc_oracle_max = oracle_em_per_noc(enc_max)
enc_oracle_sum = oracle_em_per_noc(enc_sum)
print("\n=== (C1) ENCODER readout (attr_head soft-vote) oracle EM per NOC ===")
print("   reduce=MAX:", {k: round(v, 4) for k, v in enc_oracle_max.items()})
print("   reduce=SUM:", {k: round(v, 4) for k, v in enc_oracle_sum.items()})
print("   IF encoder-vote N5 >> decoder N5  -> decoder under-reads (DECODER bottleneck)")
print("   IF encoder-vote N5 ~= decoder N5  -> H lost the donor (ENCODER bottleneck)")

# ── (B+C2) characterise N5 misses + per-peak attribution of missed donor ──────
n5 = np.where(noc == 5)[0]
print(f"\n=== (B) N5 miss analysis  (n={len(n5)}) ===")
miss_rows = []
for i in n5:
    top5 = np.argsort(probs[i])[::-1][:5]
    pred = set(top5.tolist())
    true = set(np.where(y[i] == 1)[0].tolist())
    missed = true - pred                 # true donors NOT in decoder top-5
    if not missed: continue
    rank_of = {c: int(np.where(np.argsort(probs[i])[::-1] == c)[0][0]) for c in true}
    for c in missed:
        # present alleles of donor c (ground-truth peaks attributed to c)
        c_peaks = np.where((attr[i] == c) & mask[i])[0]
        n_present = len(c_peaks)
        # per-peak attribution by the ENCODER (attr_head) on donor c's own peaks
        if n_present:
            pk_pred = attr_prob[i][c_peaks][:, :45].argmax(1)   # encoder's donor call per peak
            kept = (pk_pred == c).mean()                        # frac kept by encoder
            # where do the lost ones go? to a TRUE major donor?
            absorbed_to_major = np.mean([p in true and p != c for p in pk_pred])
        else:
            kept = absorbed_to_major = np.nan
        miss_rows.append(dict(
            i=int(i), donor=int(c), phi=float(phi[i, c]),
            phi_rank=int((phi[i] > phi[i, c]).sum()),         # 0=biggest..4=faintest among the 5
            prob=float(probs[i, c]), dec_rank=rank_of[c],
            n_present=n_present, enc_keep=kept, absorbed_major=absorbed_to_major,
            enc_vote_rank=int(np.where(np.argsort(enc_max[i])[::-1] == c)[0][0]),
        ))

import statistics as st
print(f"total missed-donor instances across N5: {len(miss_rows)}")
def col(name): return [r[name] for r in miss_rows if not (isinstance(r[name], float) and np.isnan(r[name]))]
print(f"  missed donor phi:        median={np.median(col('phi')):.4f}  "
      f"min={min(col('phi')):.4f}  max={max(col('phi')):.4f}")
pr = np.array([r['phi_rank'] for r in miss_rows])
print(f"  missed donor phi_rank (0=loudest..4=faintest of the 5): " +
      ", ".join(f"{k}:{(pr==k).sum()}" for k in range(5)))
print(f"  missed donor decoder prob:   median={np.median(col('prob')):.4f}")
print(f"  missed donor decoder rank:   median={np.median(col('dec_rank')):.1f}  "
      f"(5=just outside top-5)")
print(f"  #present alleles of missed:  median={np.median(col('n_present')):.1f}  "
      f"min={min(col('n_present'))}")
print(f"\n  --- ENCODER per-peak attribution of the MISSED donor's own peaks ---")
print(f"  enc_keep (frac of donor's peaks the encoder attributes to that donor): "
      f"median={np.median(col('enc_keep')):.3f}  mean={np.mean(col('enc_keep')):.3f}")
print(f"  absorbed_to_a_true_major (frac of donor's peaks attributed to another true donor): "
      f"mean={np.mean(col('absorbed_major')):.3f}")
evr = np.array([r['enc_vote_rank'] for r in miss_rows])
print(f"\n  --- can the ENCODER readout recover the donor the DECODER missed? ---")
print(f"  encoder-vote rank of decoder-missed donor: "
      f"in top5={ (evr<5).mean():.3f}  median rank={np.median(evr):.1f}")
print(f"    (high 'in top5' => decoder bottleneck; low => encoder also lost it)")

# show 5 hardest examples
print("\n  --- 8 example missed donors (faintest first) ---")
for r in sorted(miss_rows, key=lambda r: r['phi'])[:8]:
    print(f"   s{r['i']:4d} donor{r['donor']:2d} phi={r['phi']:.4f}(rank{r['phi_rank']}) "
          f"decP={r['prob']:.3f} decRank={r['dec_rank']} "
          f"nPeaks={r['n_present']} encKeep={r['enc_keep']:.2f} "
          f"absorbMajor={r['absorbed_major']:.2f} encVoteRank={r['enc_vote_rank']}")

"""
Causal test of the 'encoder washes the faint minor into the majors' mechanism.

For each N5 sample with a decoder-missed minor c:
  intervention = MASK OUT the observed peaks belonging (ground-truth attr) to the
  2 LOUDEST major donors, then re-encode + re-decode.
If c's probability / rank recovers strongly, the majors were causally suppressing c
in the ENCODER (height-dominated attention competition) -> encoder mechanism confirmed.

Also: per-peak linear separability of the minor's private FAINT peaks in the RAW
projected tokens x0 vs the ENCODED set H (does ISAB context-mixing destroy it?).
"""
import json, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_insilico_w"; CKPT = ROOT / "results" / "inc6_maskp_seed42"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture

def L(n): return np.load(DATA / f"{n}.npy", allow_pickle=True)
tokens=L("tokens8_test").astype(np.float32); mask=L("mask_test").astype(bool)
y=L("y_test_set").astype(np.float32); noc=L("noc_test").astype(int)
attr=L("attr_test").astype(int); phi=L("phi_test").astype(np.float32)
B=len(tokens)

cfg=json.load(open(CKPT/"metrics.json"))["config"]
model=SetTransformerMixture(n_loci=cfg["n_loci"],d_locus=cfg["d_locus"],d_model=cfg["d_model"],
    n_heads=cfg["n_heads"],n_isab=cfg["n_isab"],m_inducing=cfg["m_inducing"],n_classes=cfg["n_classes"],
    n_noc=cfg["n_noc"],dropout=cfg["dropout"],cls_decoder=cfg["cls_decoder"],n_token_feats=cfg["n_token_feats"],
    encoder=cfg["encoder"],num_embed=cfg["num_embed"],periodic_sigma=cfg["periodic_sigma"],
    aux_heads=cfg["aux_heads"],sparse_attn=cfg["sparse_attn"]).to(DEV)
model.load_state_dict(torch.load(CKPT/"best_model.pt",weights_only=True,map_location=DEV)); model.eval()

@torch.no_grad()
def probs_for(tok, msk):
    P=[]
    for s in range(0,len(tok),256):
        tk=torch.from_numpy(tok[s:s+256]).to(DEV); mk=torch.from_numpy(msk[s:s+256]).to(DEV)
        _,H,pad=model._encode_set(tk,mk)
        P.append(torch.sigmoid(model.cls_decoder_module(H,pad_mask=pad)).cpu().numpy())
    return np.concatenate(P)

base = probs_for(tokens, mask)
n5 = np.where(noc==5)[0]

# collect missed minors
missed=[]
for i in n5:
    top5=set(np.argsort(base[i])[::-1][:5].tolist())
    for c in np.where(y[i]==1)[0]:
        if c not in top5: missed.append((int(i),int(c)))
print(f"N5 missed (donor,sample) instances: {len(missed)}")

# build intervened copies: for each missed (i,c), mask peaks of the 2 loudest majors
tok2 = tokens.copy(); msk2 = mask.copy()
# we process per missed instance but several share a sample; do per-sample union of loud-major peaks
from collections import defaultdict
bysamp = defaultdict(list)
for i,c in missed: bysamp[i].append(c)

rows=[]
for i, cs in bysamp.items():
    true=np.where(y[i]==1)[0]
    order=true[np.argsort(phi[i,true])[::-1]]      # loudest first
    majors=order[:2].tolist()                      # 2 loudest donors
    drop = np.isin(attr[i], majors) & mask[i]      # their peaks
    mi = mask[i].copy(); mi[drop]=False
    # store for batched rerun: we make a single-sample batch later
    rows.append((i, cs, majors, mi, int(drop.sum())))

# batched rerun with masks zeroed
tokB = np.stack([tokens[i] for i,_,_,_,_ in rows])
mskB = np.stack([mi for _,_,_,mi,_ in rows])
intP = probs_for(tokB, mskB)

print("\n=== causal: prob & rank of the MISSED minor, BEFORE vs AFTER masking the 2 loudest majors ===")
dp=[]; dr=[]; recov=0; tot=0
for bi,(i,cs,majors,mi,ndrop) in enumerate(rows):
    for c in cs:
        p0=base[i,c]; p1=intP[bi,c]
        r0=int(np.where(np.argsort(base[i])[::-1]==c)[0][0])
        r1=int(np.where(np.argsort(intP[bi])[::-1]==c)[0][0])
        dp.append(p1-p0); dr.append(r1-r0); tot+=1
        # 'recovered' = now in top-3 (only 3 true donors remain after dropping 2 majors)
        if r1<3: recov+=1
dp=np.array(dp); dr=np.array(dr)
print(f"  instances: {tot}")
print(f"  prob  BEFORE median={np.median([base[i,c] for i,cs,_,_,_ in rows for c in cs]):.3f}")
print(f"  prob  AFTER  median={np.median([intP[bi,c] for bi,(i,cs,_,_,_) in enumerate(rows) for c in cs]):.3f}")
print(f"  delta prob:  median={np.median(dp):+.3f}  mean={dp.mean():+.3f}  (>0 in {100*(dp>0).mean():.0f}%)")
print(f"  delta rank:  median={np.median(dr):+.1f}  (negative = moved UP the ranking)")
print(f"  recovered into top-3 after removing 2 majors: {recov}/{tot} = {recov/tot:.2f}")
print("  (strong positive delta => the loud majors were causally suppressing the minor in the encoder)")

# ── per-peak linear separability: raw x0 vs encoded H ─────────────────────────
# For a missed minor's PRIVATE FAINT peaks, is the peak embedding closer to donor c's
# query in x0 (raw) than in H (encoded)? We use the donor decoder's score_w as the donor
# direction and measure cosine of each private peak's embedding to score_w[c] vs to the
# loudest major's score_w. If H pulls the peak toward the major (vs x0), context-mixing
# absorbed it.
g=np.load(ROOT/"data"/"donor_geno.npy"); gm=np.load(ROOT/"data"/"donor_geno_mask.npy").astype(bool)
def key(lo,al): return lo.astype(int)*1000+np.round(al*10).astype(int)
ref_keys=[set(key(g[c,gm[c],0],g[c,gm[c],1]).tolist()) for c in range(45)]
score_w = model.cls_decoder_module.score_w.detach().cpu().numpy()  # (45,128)
score_w = score_w/ (np.linalg.norm(score_w,axis=1,keepdims=True)+1e-9)

@torch.no_grad()
def embeds(i):
    tk=torch.from_numpy(tokens[i:i+1]).to(DEV); mk=torch.from_numpy(mask[i:i+1]).to(DEV)
    x0,H,_=model._encode_set(tk,mk)
    return x0[0].cpu().numpy(), H[0].cpu().numpy()

cos_x0_c=[]; cos_H_c=[]; cos_x0_M=[]; cos_H_M=[]
for i,c in missed[:200]:
    true=np.where(y[i]==1)[0]; M=int(true[np.argmax(phi[i,true])])  # loudest major
    obs_keys=key(tokens[i,mask[i],0],tokens[i,mask[i],1])
    others=set().union(*[ref_keys[o] for o in true if o!=c])
    privmask=np.array([ (k_ in ref_keys[c]) and (k_ not in others) for k_ in obs_keys ])
    if privmask.sum()==0: continue
    idx=np.where(mask[i])[0][privmask]
    x0,H=embeds(i)
    xv=x0[idx]; hv=H[idx]
    xv=xv/(np.linalg.norm(xv,axis=1,keepdims=True)+1e-9); hv=hv/(np.linalg.norm(hv,axis=1,keepdims=True)+1e-9)
    cos_x0_c+= (xv@score_w[c]).tolist();  cos_H_c+= (hv@score_w[c]).tolist()
    cos_x0_M+= (xv@score_w[M]).tolist();  cos_H_M+= (hv@score_w[M]).tolist()

print("\n=== per-peak cosine of the missed minor's PRIVATE peaks to donor directions ===")
print(f"  RAW x0:   to-own-donor={np.mean(cos_x0_c):+.3f}   to-loud-major={np.mean(cos_x0_M):+.3f}   "
      f"(own-major gap {np.mean(cos_x0_c)-np.mean(cos_x0_M):+.3f})")
print(f"  ENCODED H:to-own-donor={np.mean(cos_H_c):+.3f}   to-loud-major={np.mean(cos_H_M):+.3f}   "
      f"(own-major gap {np.mean(cos_H_c)-np.mean(cos_H_M):+.3f})")
print("  (gap shrinking / flipping from x0 to H => ISAB context-mixing pulls the private peak toward the major)")

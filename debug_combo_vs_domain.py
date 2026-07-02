"""
After the 3-way retrain: is the REMAINING N5 gap (decoder ~0.81 vs the 0.94 in-domain
ceiling) domain, encoder, or COMBINATORIAL-GENERALIZATION of the readout?

For each checkpoint, encode the SAME real test with that model, then fit a per-peak
linear probe (combo-invariant by construction) on HALF the real test (real attr labels),
eval the other half -> in-domain ceiling. Compare to the model's decoder N5.
  ceiling >> decoder  -> info IS in H; decoder fails to GENERALIZE the readout across
                         combos (combo-readout gap) -> lever = combo-invariant readout.
  ceiling ~= decoder  -> H itself lost combo-separability -> encoder/representation limit.
Also prints TRAIN vs DEV oracle (the pure combinatorial-generalization gap, difficulty-matched).
"""
import json, numpy as np, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
tok=L("tokens8_test").astype(np.float32); mk=L("mask_test").astype(bool)
y=L("y_test_set").astype(np.float32); noc=L("noc_test").astype(int); at=L("attr_test").astype(int)
B=len(tok)
rng=np.random.RandomState(0); perm=rng.permutation(B); fit_i,ev_i=perm[:B//2],perm[B//2:]

def build(ckpt):
    cfg=json.load(open(ckpt/"metrics.json"))["config"]
    m=SetTransformerMixture(n_loci=cfg["n_loci"],d_locus=cfg["d_locus"],d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],n_isab=cfg["n_isab"],m_inducing=cfg["m_inducing"],n_classes=cfg["n_classes"],
        n_noc=cfg["n_noc"],dropout=cfg["dropout"],cls_decoder=cfg["cls_decoder"],n_token_feats=cfg["n_token_feats"],
        encoder=cfg["encoder"],num_embed=cfg["num_embed"],periodic_sigma=cfg["periodic_sigma"],
        aux_heads=cfg["aux_heads"],sparse_attn=cfg["sparse_attn"]).to(DEV)
    m.load_state_dict(torch.load(ckpt/"best_model.pt",weights_only=True,map_location=DEV)); m.eval()
    return m

@torch.no_grad()
def encode(m):
    H=np.zeros((B,tok.shape[1],128),np.float32); P=np.zeros((B,45),np.float32)
    for s in range(0,B,256):
        tk=torch.from_numpy(tok[s:s+256]).to(DEV); mb=torch.from_numpy(mk[s:s+256]).to(DEV)
        _,h,pad=m._encode_set(tk,mb); H[s:s+256]=h.cpu().numpy()
        P[s:s+256]=torch.sigmoid(m.cls_decoder_module(h,pad_mask=pad)).cpu().numpy()
    return H,P

def n5_oracle(P,idxs):
    ok=[]
    for i in idxs:
        if noc[i]!=5: continue
        top=np.argsort(P[i])[::-1][:5]; pr=np.zeros(45,int); pr[top]=1; ok.append((pr==y[i]).all())
    return float(np.mean(ok)) if ok else float("nan")

def probe_ceiling(H):
    Xs,Ys=[],[]
    for i in fit_i:
        v=mk[i]&(at[i]>=0)
        if v.sum(): Xs.append(H[i][v]); Ys.append(at[i][v])
    clf=LogisticRegression(max_iter=300,C=1.0).fit(np.concatenate(Xs),np.concatenate(Ys)); cl=clf.classes_
    P=np.zeros((B,45))
    for i in ev_i:
        v=mk[i]
        if v.sum(): pp=clf.predict_proba(H[i][v]); sc=np.zeros(45); sc[cl]=pp.max(0); P[i]=sc
    return n5_oracle(P,ev_i)

for arm in ["genprop_orig_seed42","genprop_real_seed42","genprop_wide_seed42"]:
    ck=ROOT/"results"/arm; M=json.load(open(ck/"metrics.json"))
    m=build(ck); H,P=encode(m)
    dec_ev=n5_oracle(P,ev_i)                       # decoder N5 on eval half (apples-to-apples)
    ceil=probe_ceiling(H)                           # in-domain combo-invariant readout ceiling
    g=M["generalization"]
    print(f"\n{arm}")
    print(f"  decoder N5 (eval half)            : {dec_ev:.3f}")
    print(f"  in-domain per-peak probe ceiling  : {ceil:.3f}   (gap above decoder = combo-readout headroom)")
    print(f"  TRAIN N5 oracle {g['train_oracle']['5']:.3f}  ->  DEV N5 {g['dev_oracle']['5']:.3f}  "
          f"(combo-gen gap, difficulty-matched = {g['train_oracle']['5']-g['dev_oracle']['5']:+.3f})")

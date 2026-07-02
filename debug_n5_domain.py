"""
Disambiguate the 0.756 (decoder) -> 0.944 (in-domain linear probe on H) N5 gap:
is it (a) the readout ARCHITECTURE (linear per-peak >> sparsemax per-donor), or
       (b) DOMAIN SHIFT (decoder trained on SYNTHETIC train, test is REAL)?

Fit the SAME per-peak linear probe on encoded H of the SYNTHETIC TRAIN set, then
evaluate soft-vote oracle EM on the REAL test set.
  train-on-synth probe N5 ~ 0.79  => gap is DOMAIN SHIFT (decoder fine, both fail on real)
  train-on-synth probe N5 >> 0.79  => readout ARCHITECTURE matters (decoder under-reads H)
"""
import json, numpy as np, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"; CKPT=ROOT/"results"/"inc6_maskp_seed42"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture

def L(n): return np.load(DATA/f"{n}.npy", allow_pickle=True)
cfg=json.load(open(CKPT/"metrics.json"))["config"]
model=SetTransformerMixture(n_loci=cfg["n_loci"],d_locus=cfg["d_locus"],d_model=cfg["d_model"],
    n_heads=cfg["n_heads"],n_isab=cfg["n_isab"],m_inducing=cfg["m_inducing"],n_classes=cfg["n_classes"],
    n_noc=cfg["n_noc"],dropout=cfg["dropout"],cls_decoder=cfg["cls_decoder"],n_token_feats=cfg["n_token_feats"],
    encoder=cfg["encoder"],num_embed=cfg["num_embed"],periodic_sigma=cfg["periodic_sigma"],
    aux_heads=cfg["aux_heads"],sparse_attn=cfg["sparse_attn"]).to(DEV)
model.load_state_dict(torch.load(CKPT/"best_model.pt",weights_only=True,map_location=DEV)); model.eval()

@torch.no_grad()
def encode(tok,msk,want_probs=False):
    H=np.zeros((len(tok),tok.shape[1],cfg["d_model"]),np.float32); PB=None
    if want_probs: PB=np.zeros((len(tok),45),np.float32)
    for s in range(0,len(tok),256):
        tk=torch.from_numpy(tok[s:s+256]).to(DEV); mk=torch.from_numpy(msk[s:s+256]).to(DEV)
        _,h,pad=model._encode_set(tk,mk); H[s:s+256]=h.cpu().numpy()
        if want_probs: PB[s:s+256]=torch.sigmoid(model.cls_decoder_module(h,pad_mask=pad)).cpu().numpy()
    return (H,PB) if want_probs else H

# real test
tk_te=L("tokens8_test").astype(np.float32); mk_te=L("mask_test").astype(bool)
y_te=L("y_test_set").astype(np.float32); noc_te=L("noc_test").astype(int)
H_te,base=encode(tk_te,mk_te,want_probs=True)

# synthetic train subsample (balanced-ish over NOC), for fitting the probe
tk_tr=L("tokens8_train").astype(np.float32); mk_tr=L("mask_train").astype(bool)
at_tr=L("attr_train").astype(int); noc_tr=L("noc_train").astype(int)
rng=np.random.RandomState(0)
sub=[]
for k in range(1,6):
    ids=np.where(noc_tr==k)[0]; rng.shuffle(ids); sub+=ids[:2000].tolist()
sub=np.array(sub); rng.shuffle(sub)
print(f"fit probe on {len(sub)} synthetic-train samples; eval on {len(tk_te)} real-test")
H_tr=encode(tk_tr[sub],mk_tr[sub])

def peakset(H,mk,at,idxs=None):
    Xs,Ys=[],[]; rng_=range(len(H)) if idxs is None else idxs
    for i in rng_:
        v=mk[i]&(at[i]>=0)
        if v.sum()==0: continue
        Xs.append(H[i][v]); Ys.append(at[i][v])
    return np.concatenate(Xs),np.concatenate(Ys)

Xtr,Ytr=peakset(H_tr,mk_tr[sub],at_tr[sub])
clf=LogisticRegression(max_iter=300,C=1.0).fit(Xtr,Ytr)
classes=clf.classes_
P=np.zeros((len(tk_te),45))
for i in range(len(tk_te)):
    v=mk_te[i]
    if v.sum()==0: continue
    pp=clf.predict_proba(H_te[i][v]); sc=np.zeros(45); sc[classes]=pp.max(0); P[i]=sc

def oracle(Pr):
    out={}
    for k in range(1,6):
        idx=np.where(noc_te==k)[0]; ok=[]
        for i in idx:
            top=np.argsort(Pr[i])[::-1][:k]; pred=np.zeros(45,int); pred[top]=1
            ok.append((pred==y_te[i]).all())
        out[k]=round(float(np.mean(ok)),4)
    return out

print("\n=== probe FIT ON SYNTHETIC TRAIN H, eval on REAL TEST ===")
print("  DECODER (model)            N5 oracle:", oracle(base))
print("  per-peak linear probe (H)  N5 oracle:", oracle(P))
print("\n  if probe(train-on-synth) N5 >> decoder N5 -> readout ARCHITECTURE (decoder under-reads H)")
print("  if probe(train-on-synth) N5 ~= decoder N5 -> DOMAIN SHIFT synth->real (decoder fine)")

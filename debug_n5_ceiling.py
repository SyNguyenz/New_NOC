"""
Encoder-output ceiling: what is the BEST achievable per-donor readout of the
encoded set H?  If even an oracle linear probe on H ceilings near the decoder's
0.79 on N5, the information is gone from H -> ENCODER bottleneck.  If it reaches
~0.95, H carries the info and the per-donor decoder is under-reading -> DECODER.

Probe = per-peak L2-logistic regression (peak embedding -> attr donor), fit on
HALF the test samples' peaks, soft-voted (max over a sample's peaks) on the OTHER
half, scored by oracle top-k EM per NOC.  Split is by SAMPLE (no peak leakage).
Compared against the same readout on the RAW projected tokens x0 (pre-ISAB) — if
x0 >= H on N5, ISAB context-mixing is destroying faint-minor separability.
"""
import json, numpy as np, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parent
DATA = ROOT/"data_insilico_w"; CKPT = ROOT/"results"/"inc6_maskp_seed42"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture

def L(n): return np.load(DATA/f"{n}.npy", allow_pickle=True)
tokens=L("tokens8_test").astype(np.float32); mask=L("mask_test").astype(bool)
y=L("y_test_set").astype(np.float32); noc=L("noc_test").astype(int); attr=L("attr_test").astype(int)
B=len(tokens)
cfg=json.load(open(CKPT/"metrics.json"))["config"]
model=SetTransformerMixture(n_loci=cfg["n_loci"],d_locus=cfg["d_locus"],d_model=cfg["d_model"],
    n_heads=cfg["n_heads"],n_isab=cfg["n_isab"],m_inducing=cfg["m_inducing"],n_classes=cfg["n_classes"],
    n_noc=cfg["n_noc"],dropout=cfg["dropout"],cls_decoder=cfg["cls_decoder"],n_token_feats=cfg["n_token_feats"],
    encoder=cfg["encoder"],num_embed=cfg["num_embed"],periodic_sigma=cfg["periodic_sigma"],
    aux_heads=cfg["aux_heads"],sparse_attn=cfg["sparse_attn"]).to(DEV)
model.load_state_dict(torch.load(CKPT/"best_model.pt",weights_only=True,map_location=DEV)); model.eval()

@torch.no_grad()
def encode_all():
    X0,HH,PB=[],[],[]
    for s in range(0,B,256):
        tk=torch.from_numpy(tokens[s:s+256]).to(DEV); mk=torch.from_numpy(mask[s:s+256]).to(DEV)
        x0,H,pad=model._encode_set(tk,mk)
        PB.append(torch.sigmoid(model.cls_decoder_module(H,pad_mask=pad)).cpu().numpy())
        X0.append(x0.cpu().numpy().astype(np.float32)); HH.append(H.cpu().numpy().astype(np.float32))
    return np.concatenate(X0),np.concatenate(HH),np.concatenate(PB)
x0all,Hall,base=encode_all()
print("encoded. decoder N5 oracle (ref):", end=" ")
def oracle(P):
    out={}
    for k in range(1,6):
        idx=np.where(noc==k)[0]; ok=[]
        for i in idx:
            top=np.argsort(P[i])[::-1][:k]; pred=np.zeros(45,int); pred[top]=1
            ok.append((pred==y[i]).all())
        out[k]=round(float(np.mean(ok)),4)
    return out
print(oracle(base))

rng=np.random.RandomState(0)
perm=rng.permutation(B); half=B//2
fit_idx,ev_idx=perm[:half],perm[half:]

def peak_dataset(feat, idxs):
    Xs,Ys=[],[]
    for i in idxs:
        v=mask[i]&(attr[i]>=0)
        if v.sum()==0: continue
        Xs.append(feat[i][v]); Ys.append(attr[i][v])
    return np.concatenate(Xs),np.concatenate(Ys)

def probe(feat, tag):
    Xtr,Ytr=peak_dataset(feat,fit_idx)
    clf=LogisticRegression(max_iter=300,C=1.0,n_jobs=-1)
    clf.fit(Xtr,Ytr)
    classes=clf.classes_
    # soft-vote on eval half: per donor, max over a sample's peaks of predicted prob
    P=np.zeros((B,45))
    for i in ev_idx:
        v=mask[i]
        if v.sum()==0: continue
        pp=clf.predict_proba(feat[i][v])   # (n_valid, n_classes_seen)
        sc=np.zeros(45)
        sc[classes]=pp.max(0)
        P[i]=sc
    out={}
    for k in range(1,6):
        idx=[i for i in ev_idx if noc[i]==k]; ok=[]
        for i in idx:
            top=np.argsort(P[i])[::-1][:k]; pred=np.zeros(45,int); pred[top]=1
            ok.append((pred==y[i]).all())
        out[k]=round(float(np.mean(ok)),4) if ok else None
    print(f"  {tag}: oracle EM per NOC (eval half) = {out}")
    return P

def oracle_subset(P, idxs):
    out={}
    for k in range(1,6):
        ok=[]
        for i in idxs:
            if noc[i]!=k: continue
            top=np.argsort(P[i])[::-1][:k]; pred=np.zeros(45,int); pred[top]=1
            ok.append((pred==y[i]).all())
        out[k]=round(float(np.mean(ok)),4) if ok else None
    return out

print("\n=== oracle per-peak linear probe ceiling (eval half) ===")
print("  DECODER   : oracle EM per NOC (eval half) =", oracle_subset(base, ev_idx))
probe(Hall, "ENCODED H ")
probe(x0all,"RAW x0    ")
print("\nInterpretation:")
print("  H-probe N5 ~ decoder N5 (~0.79)  => info gone from H -> ENCODER bottleneck")
print("  H-probe N5 >> decoder N5         => decoder under-reads H -> DECODER bottleneck")
print("  x0-probe >= H-probe on N5        => ISAB mixing destroys faint-minor separability")

"""
Q3: what does the holistic DECODER actually have over the per-peak attr_head for ID?
If attr_head with a BETTER aggregation (sum/logsumexp instead of crude max) matches or
beats the decoder on EVERY NOC, then the decoder's only edge was learned aggregation
(which we can give a per-peak readout) -> a single principled readout is viable.
If the decoder still wins somewhere (e.g. N4 skewed), it has genuine set/joint reasoning.
"""
import json, numpy as np, torch
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
tok=L("tokens8_test").astype(np.float32); mk=L("mask_test").astype(bool)
y=L("y_test_set").astype(np.float32); noc=L("noc_test").astype(int)
B=len(tok)
def build(ck):
    c=json.load(open(ck/"metrics.json"))["config"]
    m=SetTransformerMixture(n_loci=c["n_loci"],d_locus=c["d_locus"],d_model=c["d_model"],n_heads=c["n_heads"],
        n_isab=c["n_isab"],m_inducing=c["m_inducing"],n_classes=c["n_classes"],n_noc=c["n_noc"],dropout=c["dropout"],
        cls_decoder=c["cls_decoder"],n_token_feats=c["n_token_feats"],encoder=c["encoder"],num_embed=c["num_embed"],
        periodic_sigma=c["periodic_sigma"],aux_heads=c["aux_heads"],sparse_attn=c["sparse_attn"]).to(DEV)
    m.load_state_dict(torch.load(ck/"best_model.pt",weights_only=True,map_location=DEV)); m.eval(); return m
@torch.no_grad()
def enc(m):
    P=np.zeros((B,45),np.float32); A=np.zeros((B,tok.shape[1],45),np.float32); Alog=np.zeros_like(A)
    for s in range(0,B,256):
        tk_=torch.from_numpy(tok[s:s+256]).to(DEV); mb=torch.from_numpy(mk[s:s+256]).to(DEV)
        _,h,pad=m._encode_set(tk_,mb)
        P[s:s+256]=torch.sigmoid(m.cls_decoder_module(h,pad_mask=pad)).cpu().numpy()
        la=m.attr_head(h)[:,:,:45]
        A[s:s+256]=torch.softmax(la,-1).cpu().numpy()
        Alog[s:s+256]=torch.log_softmax(la,-1).cpu().numpy()
    return P,A,Alog
def agg(A,mode,Alog=None):
    sc=np.zeros((B,45))
    for i in range(B):
        v=mk[i]
        if not v.sum(): continue
        a=A[i][v]
        if mode=="max": sc[i]=a.max(0)
        elif mode=="sum": sc[i]=a.sum(0)
        elif mode=="mean": sc[i]=a.mean(0)
        elif mode=="top3": sc[i]=np.sort(a,0)[::-1][:3].sum(0)
        elif mode=="lse": sc[i]=np.log(np.exp(Alog[i][v]).sum(0)+1e-9)  # logsumexp of log-probs = log(sum prob)
    return sc
def pn(P):
    out={}
    for k in range(1,6):
        idx=np.where(noc==k)[0]; e=[]
        for i in idx:
            top=np.argsort(P[i])[::-1][:k]; pr=np.zeros(45,int); pr[top]=1; e.append((pr==y[i]).all())
        out[k]=round(float(np.mean(e)),3)
    return out
for arm in ["genprop_orig_seed42","genprop_real_seed42"]:
    m=build(ROOT/"results"/arm); P,A,Alog=enc(m)
    print(f"\n===== {arm} =====")
    print(f"  decoder        : {pn(P)}")
    print(f"  attr MAX       : {pn(agg(A,'max'))}")
    print(f"  attr SUM       : {pn(agg(A,'sum'))}")
    print(f"  attr MEAN      : {pn(agg(A,'mean'))}")
    print(f"  attr TOP3-sum  : {pn(agg(A,'top3'))}")
    print(f"  attr LOGSUMEXP : {pn(agg(A,'lse',Alog))}")

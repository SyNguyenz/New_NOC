"""
Cheap (no-retrain) test of the lever: does a DEPLOYABLE combo-invariant readout of H pull
the decoder's N5 toward the 0.93 in-domain ceiling, without hurting N1-N4?

Readouts of the SAME encoded H, per checkpoint:
  decoder      : model's per-donor sparsemax (baseline)
  attr_vote    : attr_head per-peak softmax, max-over-peaks  (combo-invariant, SYNTH-trained, deployable)
  valprobe     : per-peak logistic fit on REAL VAL H         (combo-invariant, REAL-calibrated, deployable)
  ens          : z-norm(decoder) + z-norm(valprobe)          (deployable ensemble)
  ceiling      : per-peak logistic fit on REAL TEST H (CV)   (UPPER BOUND, uses test labels — not deployable)
Reports per-NOC oracle EM (k = true NOC). N1-N4 = no-regression guard.
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
tkv=L("tokens8_val").astype(np.float32); mkv=L("mask_val").astype(bool); atv=L("attr_val").astype(int)
B=len(tok); rng=np.random.RandomState(0); perm=rng.permutation(B); fit_i,ev_i=set(perm[:B//2]),perm[B//2:]

def build(ck):
    c=json.load(open(ck/"metrics.json"))["config"]
    m=SetTransformerMixture(n_loci=c["n_loci"],d_locus=c["d_locus"],d_model=c["d_model"],n_heads=c["n_heads"],
        n_isab=c["n_isab"],m_inducing=c["m_inducing"],n_classes=c["n_classes"],n_noc=c["n_noc"],dropout=c["dropout"],
        cls_decoder=c["cls_decoder"],n_token_feats=c["n_token_feats"],encoder=c["encoder"],num_embed=c["num_embed"],
        periodic_sigma=c["periodic_sigma"],aux_heads=c["aux_heads"],sparse_attn=c["sparse_attn"]).to(DEV)
    m.load_state_dict(torch.load(ck/"best_model.pt",weights_only=True,map_location=DEV)); m.eval(); return m

@torch.no_grad()
def enc(m,T,Mk):
    n=len(T); H=np.zeros((n,T.shape[1],128),np.float32); P=np.zeros((n,45),np.float32); A=np.zeros((n,T.shape[1],45),np.float32)
    for s in range(0,n,256):
        tk_=torch.from_numpy(T[s:s+256]).to(DEV); mb=torch.from_numpy(Mk[s:s+256]).to(DEV)
        _,h,pad=m._encode_set(tk_,mb); H[s:s+256]=h.cpu().numpy()
        P[s:s+256]=torch.sigmoid(m.cls_decoder_module(h,pad_mask=pad)).cpu().numpy()
        A[s:s+256]=torch.softmax(m.attr_head(h)[:,:,:45],-1).cpu().numpy()
    return H,P,A

def vote(A,Mk):
    n=len(A); sc=np.zeros((n,45))
    for i in range(n):
        v=Mk[i]
        if v.sum(): sc[i]=A[i][v].max(0)
    return sc

def fit_probe(Htr,Mktr,Attr,idxs=None):
    Xs,Ys=[],[]; idxs=range(len(Htr)) if idxs is None else idxs
    for i in idxs:
        v=Mktr[i]&(Attr[i]>=0)
        if v.sum(): Xs.append(Htr[i][v]); Ys.append(Attr[i][v])
    clf=LogisticRegression(max_iter=300,C=1.0).fit(np.concatenate(Xs),np.concatenate(Ys))
    return clf
def apply_probe(clf,H,Mk):
    cl=clf.classes_; n=len(H); P=np.zeros((n,45))
    for i in range(n):
        v=Mk[i]
        if v.sum(): pp=clf.predict_proba(H[i][v]); P[i,cl]=pp.max(0)
    return P
def znorm(M): return (M-M.mean(1,keepdims=True))/(M.std(1,keepdims=True)+1e-9)

def per_noc_oracle(P, idxs=None):
    idxs=range(B) if idxs is None else idxs
    out={}
    for k in range(1,6):
        ok=[i for i in idxs if noc[i]==k]
        if not ok: continue
        e=[]
        for i in ok:
            top=np.argsort(P[i])[::-1][:k]; pr=np.zeros(45,int); pr[top]=1; e.append((pr==y[i]).all())
        out[k]=round(float(np.mean(e)),3)
    return out

for arm in ["genprop_orig_seed42","genprop_real_seed42"]:
    ck=ROOT/"results"/arm; m=build(ck)
    H,P,A=enc(m,tok,mk); Hv,_,_=enc(m,tkv,mkv)
    attr_vote=vote(A,mk)
    clf_val=fit_probe(Hv,mkv,atv); valp=apply_probe(clf_val,H,mk)
    clf_ce=fit_probe(H,mk,at,idxs=fit_i); ceil=apply_probe(clf_ce,H,mk)
    print(f"\n===== {arm} =====")
    print(f"  decoder         : {per_noc_oracle(P)}")
    print(f"  attr_vote       : {per_noc_oracle(attr_vote)}   (combo-inv, synth, DEPLOYABLE)")
    print(f"  valprobe        : {per_noc_oracle(valp)}   (combo-inv, real-VAL, DEPLOYABLE)")
    print(f"  ens dec+val     : {per_noc_oracle(znorm(P)+znorm(valp))}   (DEPLOYABLE)")
    print(f"  ens dec+attr    : {per_noc_oracle(znorm(P)+znorm(attr_vote))}   (DEPLOYABLE, no real labels)")
    print(f"  ens dec+attr+val: {per_noc_oracle(znorm(P)+znorm(attr_vote)+znorm(valp))}   (DEPLOYABLE)")
    # decoder-DOMINANT weightings: decoder anchors the ranking, combo-inv readouts only nudge.
    print(f"  ens 2dec+attr   : {per_noc_oracle(2*znorm(P)+znorm(attr_vote))}   (decoder-anchored)")
    print(f"  ens 3dec+attr+val:{per_noc_oracle(3*znorm(P)+znorm(attr_vote)+znorm(valp))}   (decoder-anchored)")
    # NOC-gated: pure decoder for N<=3 (where it's already ~ceiling), ensemble only for N4-5.
    gated=np.where((noc>=4)[:,None], znorm(P)+znorm(attr_vote)+znorm(valp), znorm(P))
    print(f"  gated(N>=4 ens) : {per_noc_oracle(gated)}   (zero change at N1-3 by construction)")
    print(f"  ceiling         : {per_noc_oracle(ceil, ev_i)}   (test-fit CV, eval-half, UPPER BOUND)")

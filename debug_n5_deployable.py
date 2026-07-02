"""
Deployable check: fit the per-peak linear readout of H on the REAL VAL set (1366,
held-out real data available at model-selection time, has attr labels), evaluate on
the REAL TEST set. If N5 oracle >> 0.79, recalibrating the readout on a little real
data is a real, deployable lever for the N5 wall (no test labels used).
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
def encode(tok,msk,wp=False):
    H=np.zeros((len(tok),tok.shape[1],cfg["d_model"]),np.float32); PB=np.zeros((len(tok),45),np.float32) if wp else None
    for s in range(0,len(tok),256):
        tk=torch.from_numpy(tok[s:s+256]).to(DEV); mk=torch.from_numpy(msk[s:s+256]).to(DEV)
        _,h,pad=model._encode_set(tk,mk); H[s:s+256]=h.cpu().numpy()
        if wp: PB[s:s+256]=torch.sigmoid(model.cls_decoder_module(h,pad_mask=pad)).cpu().numpy()
    return (H,PB) if wp else H

tk_te=L("tokens8_test").astype(np.float32); mk_te=L("mask_test").astype(bool)
y_te=L("y_test_set").astype(np.float32); noc_te=L("noc_test").astype(int)
tk_va=L("tokens8_val").astype(np.float32); mk_va=L("mask_val").astype(bool); at_va=L("attr_val").astype(int)
print(f"real val fit set: {len(tk_va)} | real test eval: {len(tk_te)}")
print("val NOC dist:", {k:int((L('noc_val').astype(int)==k).sum()) for k in range(1,6)})
print("val attr has real labels?", (at_va>=0).any(), "valid peak labels:", int((at_va>=0).sum()))

H_va=encode(tk_va,mk_va); H_te,base=encode(tk_te,mk_te,wp=True)
Xs,Ys=[],[]
for i in range(len(H_va)):
    v=mk_va[i]&(at_va[i]>=0)
    if v.sum(): Xs.append(H_va[i][v]); Ys.append(at_va[i][v])
X=np.concatenate(Xs); Y=np.concatenate(Ys)
clf=LogisticRegression(max_iter=400,C=1.0).fit(X,Y); cl=clf.classes_
P=np.zeros((len(tk_te),45))
for i in range(len(tk_te)):
    v=mk_te[i]
    if v.sum(): pp=clf.predict_proba(H_te[i][v]); sc=np.zeros(45); sc[cl]=pp.max(0); P[i]=sc
def oracle(Pr):
    out={}
    for k in range(1,6):
        idx=np.where(noc_te==k)[0]; ok=[]
        for i in idx:
            top=np.argsort(Pr[i])[::-1][:k]; pred=np.zeros(45,int); pred[top]=1; ok.append((pred==y_te[i]).all())
        out[k]=round(float(np.mean(ok)),4)
    return out
print("\n=== readout fit on REAL VAL H, eval on REAL TEST ===")
print("  DECODER (model)           :", oracle(base))
print("  per-peak probe (val-fit)  :", oracle(P))

# also: ENSEMBLE the val-fit probe with the decoder (avg of normalized ranks of probs)
def znorm(M): return (M-M.mean(1,keepdims=True))/(M.std(1,keepdims=True)+1e-9)
ens=znorm(base)+znorm(P)
print("  decoder + val-probe ensemble:", oracle(ens))

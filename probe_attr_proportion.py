"""
phi_head is broken (corr -0.23) but attr_head is GOOD (0.894). So DERIVE a per-donor height/proportion
from attribution: attr_h[c] = sum of peak heights attributed to donor c. Does THIS (good-head-derived)
signal separate the model's decoy from its missed-true better than phi (0.614)?  This = the inference
signal an analysis-by-synthesis recon would use, built on the head that actually works.
"""
import numpy as np
D="results/inc13_B_distill_seed42"; DA="data_insilico_w"
tk=np.load(f"{DA}/tokens8_test.npy"); mk=np.load(f"{DA}/mask_test.npy").astype(bool)
attr_p=np.load(f"{D}/attr_pred_test.npy"); y_p=np.load(f"{D}/y_test_pred.npy").astype(bool)
y_t=np.load(f"{D}/y_test_true.npy").astype(bool); noc=np.load(f"{DA}/noc_test.npy"); C=45

H=np.expm1(tk[:,:,2])                      # peak heights (RFU)
def auc(pos,neg):
    pos,neg=np.asarray(pos,float),np.asarray(neg,float)
    if not len(pos) or not len(neg): return float("nan")
    a=np.concatenate([pos,neg]); _,inv,cnt=np.unique(a,return_inverse=True,return_counts=True)
    cs=np.cumsum(cnt); rk=((cs-cnt+cs+1)/2.0)[inv]
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

# attr-derived per-donor height (and normalized proportion)
def attr_signal(i):
    ah=np.zeros(C); v=mk[i]
    for k in np.where(v)[0]:
        c=attr_p[i,k]
        if 0<=c<C: ah[c]+=H[i,k]
    return ah, ah/max(ah.sum(),1e-6)

for NV in [5,4]:
    sel=np.where(noc==NV)[0]
    mt_h=[]; dc_h=[]; mt_n=[]; dc_n=[]; mt_cnt=[]; dc_cnt=[]
    for i in sel:
        ah,an=attr_signal(i)
        cnt=np.bincount(attr_p[i][mk[i]][attr_p[i][mk[i]]>=0], minlength=C)
        miss=np.where(y_t[i]&~y_p[i])[0]; dec=np.where(~y_t[i]&y_p[i])[0]
        for c in miss: mt_h.append(ah[c]); mt_n.append(an[c]); mt_cnt.append(cnt[c])
        for c in dec:  dc_h.append(ah[c]); dc_n.append(an[c]); dc_cnt.append(cnt[c])
    print(f"=== N{NV} attr-derived signal on model errors (n_miss={len(mt_h)}, n_decoy={len(dc_h)}) ===")
    print(f"  attr_height : miss mean={np.mean(mt_h):7.1f}  decoy mean={np.mean(dc_h):7.1f}  AUC={auc(mt_h,dc_h):.3f}")
    print(f"  attr_propor : miss mean={np.mean(mt_n):.4f}   decoy mean={np.mean(dc_n):.4f}    AUC={auc(mt_n,dc_n):.3f}")
    print(f"  #peaks attr : miss mean={np.mean(mt_cnt):.2f}     decoy mean={np.mean(dc_cnt):.2f}      AUC={auc(mt_cnt,dc_cnt):.3f}")
    print(f"  refs: phi=0.614/0.664, signature=0.50, height-combined upper bound=0.69\n")

"""
Is inference-time recon viable? It needs the phi_head/attr_head to be (a) ACCURATE and
(b) COMPLEMENTARY to the cls decision (i.e., on the model's OWN errors, does phi rank the
missed-true ABOVE the decoy?). If phi just follows the cls decision, inference-recon adds nothing.

Uses inc13_B_distill saved predictions (no model reload). GT used only to score.
"""
import numpy as np
D="results/inc13_B_distill_seed42"; DA="data_insilico_w"
phi_p=np.load(f"{D}/phi_pred_test.npy"); attr_p=np.load(f"{D}/attr_pred_test.npy")
y_p=np.load(f"{D}/y_test_pred.npy").astype(bool); y_t=np.load(f"{D}/y_test_true.npy").astype(bool)
phi_t=np.load(f"{DA}/phi_test.npy"); attr_t=np.load(f"{DA}/attr_test.npy"); noc=np.load(f"{DA}/noc_test.npy")

def auc(pos,neg):
    pos,neg=np.asarray(pos,float),np.asarray(neg,float)
    if not len(pos) or not len(neg): return float("nan")
    a=np.concatenate([pos,neg]); _,inv,cnt=np.unique(a,return_inverse=True,return_counts=True)
    cs=np.cumsum(cnt); rk=((cs-cnt+cs+1)/2.0)[inv]
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

# ── (1) phi-head accuracy ──
pres=y_t; absent=~y_t
corr=np.corrcoef(phi_p[pres], phi_t[pres])[0,1]
mae_pres=np.abs(phi_p[pres]-phi_t[pres]).mean()
print("=== phi-head accuracy ===")
print(f"  present donors: corr(phi_pred,phi_true)={corr:.3f}  MAE={mae_pres:.3f}  (true mean phi={phi_t[pres].mean():.3f})")
print(f"  absent donors : mean phi_pred={phi_p[absent].mean():.4f}  (should be ~0)")

# ── (2) attr-head accuracy (peaks with a real attributed owner) ──
real=attr_t>=0
acc_attr=(attr_p[real]==attr_t[real]).mean()
print(f"\n=== attr-head accuracy ===\n  per-peak attribution acc (attributed peaks) = {acc_attr:.3f}  (n={real.sum()})")

# ── (3) N5: is phi COMPLEMENTARY on the model's errors? missed-true vs decoy ──
for NV in [5,4]:
    sel=np.where(noc==NV)[0]
    mt_phi=[]; dc_phi=[]; mt_phiT=[]
    for i in sel:
        miss=np.where(y_t[i]&~y_p[i])[0]      # true present, model said absent
        dec =np.where(~y_t[i]&y_p[i])[0]      # true absent,  model said present
        for c in miss: mt_phi.append(phi_p[i,c]); mt_phiT.append(phi_t[i,c])
        for c in dec:  dc_phi.append(phi_p[i,c])
    a=auc(mt_phi,dc_phi)
    print(f"\n=== N{NV} phi on model's errors (n_miss={len(mt_phi)}, n_decoy={len(dc_phi)}) ===")
    print(f"  phi_pred:  missed-true mean={np.mean(mt_phi):.4f}   decoy mean={np.mean(dc_phi):.4f}")
    print(f"  true phi of missed-true = {np.mean(mt_phiT):.4f}  (these ARE faint contributors)")
    print(f"  AUC(phi_pred: missed-true vs decoy) = {a:.3f}   [0.5=phi blind/follows-cls | >0.5=complementary]")
    print(f"  reference: signature presence=0.50, height-combined upper bound=0.69")

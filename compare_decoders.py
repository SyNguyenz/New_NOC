"""
compare_decoders.py — Compare pooled vs per-donor ST + baselines on no-leak test.
Includes partial-credit diagnostic + probe of test combo {46,47,48}.

Usage: python compare_decoders.py
"""
import numpy as np, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data"
meta=json.loads((DATA/"meta_set.json").read_text())
known=meta["known_donors"]
noc=np.load(DATA/"noc_test.npy")

def metrics(pred_dir, label, probs_file=None):
    d=Path(pred_dir)
    if not (d/"y_test_pred.npy").exists():
        print(f"\n[{label}] not found ({d})"); return
    yt=np.load(d/"y_test_true.npy"); yp=np.load(d/"y_test_pred.npy")
    noc_l = np.load(d/"noc_test.npy") if (d/"noc_test.npy").exists() else noc
    tp=((yt==1)&(yp==1)).sum(1); fp=((yt==0)&(yp==1)).sum(1)
    n_true=yt.sum(1); n_pred=yp.sum(1)
    em=(yt==yp).all(1)
    mp=noc_l>=2
    print(f"\n=== {label} ===")
    print(f"  Overall EM={em.mean():.4f}  multi-person: recall={tp[mp].sum()/max(n_true[mp].sum(),1):.3f} "
          f"prec={tp[mp].sum()/max((tp[mp]+fp[mp]).sum(),1):.3f} avg_pred={n_pred[mp].mean():.2f}(true {n_true[mp].mean():.2f})")
    for t in range(1,6):
        m=noc_l==t
        if m.sum()==0: continue
        print(f"    NOC={t}: EM={em[m].mean():.3f} (n={int(m.sum())})")

print("="*60)
print("  DECODER COMPARISON (no-leak novel-combo test)")
print("="*60)
metrics(ROOT/"results/set_transformer",          "ST pooled (original)")
metrics(ROOT/"results/set_transformer_perdonor", "ST per-donor (NEW)")
metrics(ROOT/"results/baseline_lr",              "Logistic Regression")
metrics(ROOT/"results/baseline_xgb",             "XGBoost")

# Probe per-donor ST on {46,47,48} if probs available
pd_dir = ROOT/"results/set_transformer_perdonor"
if (pd_dir/"probs_test.npy").exists():
    probs=np.load(pd_dir/"probs_test.npy"); yt=np.load(pd_dir/"y_test_true.npy")
    m3=np.where(noc==3)[0]
    from collections import Counter
    cnt=Counter()
    for i in m3:
        for j in np.where(probs[i]>=0.5)[0]: cnt[known[j]]+=1
    print(f"\n  [per-donor ST] NOC=3 most-predicted (true={[known[j] for j in np.where(yt[m3[0]]>0.5)[0]]}):")
    for dn,c in cnt.most_common(8):
        print(f"    D{dn}: {c}/{len(m3)} ({c/len(m3)*100:.0f}%)")

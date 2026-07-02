"""
eval_topk.py — Evaluate a trained ST checkpoint with NOC-constrained top-k decoding.

Loads checkpoint, runs inference on REAL no-leak test, then reports Exact Match under:
  - threshold (best from val)
  - oracle top-k (k = true NOC)   [ranking-quality ceiling]
  - noc-head top-k (k = model NOC head argmax)
  - count top-k (k = #donors above 0.5)

Usage:
  python eval_topk.py --subdir set_transformer_perdonor --cls_decoder per_donor
  python eval_topk.py --subdir set_transformer --cls_decoder pooled
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
import sys

ROOT = Path(__file__).resolve().parent; DATA = ROOT/"data"
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DS(Dataset):
    def __init__(s, split):
        s.t=torch.from_numpy(np.load(DATA/f"tokens_{split}.npy"))
        s.m=torch.from_numpy(np.load(DATA/f"mask_{split}.npy"))
    def __len__(s): return len(s.t)
    def __getitem__(s,i): return s.t[i], s.m[i]


def infer(model, split):
    ds=DS(split); probs=[]; noc_logits=[]
    model.eval()
    with torch.no_grad():
        for t,m in DataLoader(ds,batch_size=256):
            out=model(t.to(DEV),m.to(DEV))
            probs.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
            noc_logits.append(out["logits_noc"].cpu().numpy())
    return np.concatenate(probs), np.concatenate(noc_logits)


def per_noc_em(yp, yt, noc):
    em=(yt==yp).all(1)
    return [em.mean()]+[em[noc==t].mean() if (noc==t).sum() else float('nan') for t in range(1,6)]


def topk_pred(probs, k_arr):
    yp=np.zeros_like(probs, dtype=int)
    for i in range(len(probs)):
        k=int(max(1,min(5,k_arr[i])))
        yp[i, np.argsort(probs[i])[::-1][:k]]=1
    return yp


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--subdir", required=True)
    ap.add_argument("--cls_decoder", default="pooled", choices=["pooled","per_donor","additive"])
    ap.add_argument("--decoder_source", default="encoded", choices=["encoded","raw","local"])
    ap.add_argument("--config", default=str(ROOT/"configs"/"set_transformer.json"))
    args=ap.parse_args()

    cfg=json.load(open(args.config))
    model=SetTransformerMixture(
        n_loci=24,d_locus=16,d_model=cfg.get("d_model",128),n_heads=cfg.get("n_heads",4),
        n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),
        n_classes=45,n_noc=6,dropout=0.1,cls_decoder=args.cls_decoder,
        decoder_source=args.decoder_source).to(DEV)
    ckpt=ROOT/"results"/args.subdir/"best_model.pt"
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    print(f"Loaded {ckpt} (cls_decoder={args.cls_decoder})")

    yt=np.load(DATA/"y_test_set.npy"); noc=np.load(DATA/"noc_test.npy")
    probs, noc_logits = infer(model, "test")
    np.save(ROOT/"results"/args.subdir/"probs_test.npy", probs)

    # val threshold search
    yv=np.load(DATA/"y_val_set.npy")
    pv,_=infer(model,"val")
    from sklearn.metrics import f1_score
    best_t,best=0.5,0
    for t in np.arange(0.1,0.85,0.05):
        f=f1_score(yv,(pv>=t).astype(int),average="macro",zero_division=0)
        if f>best: best,best_t=f,t

    k_oracle=noc
    k_nochead=noc_logits.argmax(1)
    k_count=(probs>=0.5).sum(1).clip(1,5)

    print(f"\n  {'method':<20}{'overall':>8}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
    for name,yp in [
        (f"threshold={best_t:.2f}", (probs>=best_t).astype(int)),
        ("oracle top-k",  topk_pred(probs,k_oracle)),
        ("noc-head top-k", topk_pred(probs,k_nochead)),
        ("count top-k",    topk_pred(probs,k_count)),
    ]:
        r=per_noc_em(yp,yt,noc)
        print(f"  {name:<20}"+"".join(f"{x:>7.3f}" for x in r))

    mp=noc>=2
    print(f"\n  NOC-head exact (multi-person): {np.mean(k_nochead[mp]==noc[mp]):.2f}  "
          f"within1: {np.mean(np.abs(k_nochead[mp]-noc[mp])<=1):.2f}")

if __name__=="__main__":
    main()

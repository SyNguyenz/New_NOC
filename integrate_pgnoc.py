"""
integrate_pgnoc.py — Add pgNOC continuous-model cost-curve features to the two-stage
NOC decoder and measure the deployable per-NOC EM lift on the FULL real test set.

pgNOC (gamma-weighted NNLS greedy + reference genotypes) gave AUC(4vs5)=0.815 and
NOC4 count 0.82 — far above the prob+MAC two-stage (NOC4 ~0.49). Here we feed pgNOC's
cost curve [cost_1..6, marginal drops, k*] as EXTRA features into the same two-stage
XGB classifier (stage1 NOC1-vs-multi on val, stage2 count 2-5 on in-silico train).

Compares: two-stage [prob+MAC]  vs  two-stage [prob+MAC+pgNOC].
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
from train_set_transformer import (card_features, two_stage_cardinality,
                                    topk_decode, per_noc_em)
import pgnoc

D = ROOT / "data_insilico_w"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    m = SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2,
        m_inducing=32, n_classes=45, n_noc=6, dropout=0.1, cls_decoder="hybrid",
        n_flat=590, decouple_reject=True).to(DEV)
    m.load_state_dict(torch.load(ROOT / "results/hybrid_50k_weight/best_model.pt", weights_only=True))
    m.eval(); return m


@torch.no_grad()
def get_probs(m, split):
    t = np.load(D / f"tokens_{split}.npy"); mk = np.load(D / f"mask_{split}.npy")
    xf = np.load(D / f"Xflat_{split}.npy").astype(np.float32); P = []
    for i in range(0, len(t), 512):
        P.append(torch.sigmoid(m(torch.from_numpy(t[i:i+512]).to(DEV),
            torch.from_numpy(mk[i:i+512]).to(DEV),
            torch.from_numpy(xf[i:i+512]).to(DEV))["logits_cls"]).cpu().numpy())
    return np.concatenate(P)


def pgnoc_feats(P, split, G):
    """pgNOC cost curve per sample: [cost_1..6, drops_1..5, argmin k*]."""
    H = np.expm1(np.load(D / f"Xflat_{split}.npy").astype(np.float64))
    C = np.zeros((len(P), pgnoc.KMAX))
    for i in range(len(P)):
        _, C[i] = pgnoc.estimate_noc(H[i], P[i], G)
    drops = C[:, :-1] - C[:, 1:]
    kstar = (C + 0.003 * np.arange(1, pgnoc.KMAX + 1)).argmin(1, keepdims=True) + 1
    return np.hstack([C, drops, kstar]).astype(np.float32)


def base_feats(P, split):
    return card_features(P, np.load(D / f"tokens_{split}.npy"), np.load(D / f"mask_{split}.npy"))


def main():
    t0 = time.time(); m = load_model(); G = pgnoc.build_refs()
    splits = ["train", "val", "test"]
    P = {s: get_probs(m, s) for s in splits}
    noc = {s: np.load(D / f"noc_{s}.npy").astype(int) for s in splits}
    y_te = np.load(D / "y_test_set.npy")
    print(f"probs+refs {time.time()-t0:.0f}s; computing features ...")

    t0 = time.time()
    base = {s: base_feats(P[s], s) for s in splits}
    print(f"  base (prob+MAC) {time.time()-t0:.0f}s"); t0 = time.time()
    pg = {s: pgnoc_feats(P[s], s, G) for s in splits}
    print(f"  pgNOC cost curves {time.time()-t0:.0f}s")
    comb = {s: np.hstack([base[s], pg[s]]) for s in splits}

    orc = per_noc_em(y_te, topk_decode(P["test"], noc["test"]), noc["test"])
    rows = [("oracle", orc, float("nan"))]
    kk = {}
    for name, F in [("two-stage prob+MAC", base), ("two-stage +pgNOC", comb)]:
        k = two_stage_cardinality(F["train"], noc["train"], F["val"], noc["val"], F["test"])
        kk[name] = k
        em = per_noc_em(y_te, topk_decode(P["test"], k), noc["test"])
        rows.append((name, em, float((k == noc["test"]).mean())))
    # routed: base decides when it predicts low (k<=2, keeps N2); +pgNOC when k>=3 (keeps N4/N5)
    kb = kk["two-stage prob+MAC"]; kc = kk["two-stage +pgNOC"]
    kr = np.where(kb <= 2, kb, kc)
    rows.append(("routed (base<=2|pgNOC)", per_noc_em(y_te, topk_decode(P["test"], kr), noc["test"]),
                 float((kr == noc["test"]).mean())))

    hdr = ["all", "N1", "N2", "N3", "N4", "N5"]
    print(f"\n  {'config':<22}" + "".join(f"{h:>7}" for h in hdr) + f"{'cntAcc':>8}")
    for nm, em, ca in rows:
        cs = "   —  " if np.isnan(ca) else f"{ca:>7.3f}"
        print(f"  {nm:<22}" + "".join(f"{x:>7.3f}" for x in em) + cs)

    # save routed (deployable) result + merge into checkpoint metrics
    import json
    yp = topk_decode(P["test"], kr)
    np.save(ROOT / "results/hybrid_50k_weight/y_test_pred_routed.npy", yp)
    mfile = ROOT / "results/hybrid_50k_weight/metrics.json"
    M = json.load(open(mfile))
    em_r = rows[-1][1]
    M["decode"] = "routed_pgnoc"
    M["em_routed_pgnoc"] = round(float(em_r[0]), 4)
    M["per_noc_routed_pgnoc"] = {str(j): round(float(em_r[j]), 4) for j in range(1, 6)}
    M["routed_count_acc"] = round(float((kr == noc["test"]).mean()), 4)
    json.dump(M, open(mfile, "w"), indent=2)
    print(f"\n  saved routed pred + merged metrics -> results/hybrid_50k_weight/")


if __name__ == "__main__":
    main()

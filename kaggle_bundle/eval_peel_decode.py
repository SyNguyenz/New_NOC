"""
eval_peel_decode.py — Increment 5 eval (NEW). Gated recursive-peel decode for a trained arm,
on the in-silico DEV (held-out combos, the F29 judge) and REAL test. Writes a "peel" block into
the arm's results/<sub>/metrics.json: per-NOC ORACLE (count given) for PLAIN top-k vs GATED PEEL,
across a tau sweep. Plain top-k = the no-regression guard (N1/2/3) and isolates decode vs training.

Decode (per sample, K=true NOC): keep neural picks with prob>=tau FIXED; fill the remaining
uncertain slots by subtracting the picked donors (NNLS on reference G, linear RFU) and re-scoring
the residual with the SAME model (now trained on residuals → not OOD). Matches build_residual_aug's
residual definition exactly. tau=0 => plain top-k (neural baseline); tau=1 => pure peel.

Usage: python eval_peel_decode.py <results_subdir> <data_dir> [--taus 0,0.7,0.8,0.9]
       (data_dir = CLEAN base dir, e.g. data_w — provides test/dev tokens + single-source G)
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np, torch
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_set_transformer import DEVICE, build_pgnoc_refs
from models.set_transformer import SetTransformerMixture
from features.enrich import enrich_tokens
if not os.environ.get("STR_DATA_DIR"):
    os.environ["STR_DATA_DIR"] = sys.argv[2] if len(sys.argv) > 2 else "data_w"
from make_insilico import xflat_to_tokens


def build_model(cfg):
    return SetTransformerMixture(
        n_loci=24, d_locus=16, d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 4),
        n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32), n_classes=45, n_noc=6,
        dropout=cfg.get("dropout", 0.1), cls_decoder=cfg.get("cls_decoder", "per_donor"),
        decoder_source=cfg.get("decoder_source", "encoded"), n_token_feats=cfg.get("n_token_feats", 8),
        encoder=cfg.get("encoder", "isab"), dec_layers=cfg.get("dec_layers", 2),
        num_embed=cfg.get("num_embed", "raw"), n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
        periodic_sigma=cfg.get("periodic_sigma", 1.0), aux_heads=cfg.get("aux_heads", False),
        sparse_attn=cfg.get("sparse_attn", False)).to(DEVICE)


def main(sub, data_dir, taus):
    D = Path(data_dir); res = ROOT / "results" / sub
    cfg = json.load(open(res / "metrics.json"))["config"]
    n_tok = cfg.get("n_token_feats", 8); tp = f"tokens{n_tok}" if n_tok > 3 else "tokens"
    model = build_model(cfg)
    model.load_state_dict(torch.load(res / "best_model.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    # single-source reference G from CLEAN train (linear relative profiles)
    G = build_pgnoc_refs(np.load(D / "Xflat_train.npy").astype(np.float64),
                         np.load(D / "y_train_set.npy").astype(np.float32),
                         np.load(D / "noc_train.npy").astype(np.int64))

    @torch.no_grad()
    def score_tokens(en, mk):
        out = []
        for i in range(0, len(en), 256):
            o = model(torch.from_numpy(en[i:i+256]).to(DEVICE), torch.from_numpy(mk[i:i+256]).to(DEVICE))
            out.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def score_residual(resid_lin):
        rflat = np.log1p(np.clip(resid_lin, 0, None)).astype(np.float32)
        t3, m3, _, _ = xflat_to_tokens(rflat)
        en = enrich_tokens(t3[None], m3[None])[:, :, :n_tok]
        o = model(torch.from_numpy(en).to(DEVICE), torch.from_numpy(m3[None]).to(DEVICE))
        return torch.sigmoid(o["logits_cls"])[0].cpu().numpy()

    def gated_peel(mixlin, order, probs, K, tau):
        picks = [int(d) for d in order if probs[d] >= tau][:K]
        if not picks:
            picks = [int(order[0])]
        while len(picks) < K:
            A = G[picks].T; phi, _ = nnls(A, mixlin); resid = np.clip(mixlin - A @ phi, 0, None)
            if resid.sum() < 1e-9:
                for d in order:
                    if d not in picks: picks.append(int(d)); break
                continue
            pr = score_residual(resid); pr[picks] = -1e9; picks.append(int(np.argmax(pr)))
        return set(picks)

    def eval_split(split):
        if not (D / f"mask_{split}.npy").exists():
            return None
        if (D / f"{tp}_{split}.npy").exists():
            en = np.load(D / f"{tp}_{split}.npy").astype(np.float32)
        else:   # fall back: enrich raw tokens on the fly
            en = enrich_tokens(np.load(D / f"tokens_{split}.npy").astype(np.float32),
                               np.load(D / f"mask_{split}.npy").astype(bool))[:, :, :n_tok]
        mk = np.load(D / f"mask_{split}.npy").astype(bool)
        y = np.load(D / f"y_{split}_set.npy").astype(int)
        noc = np.clip(np.load(D / f"noc_{split}.npy").astype(int), 1, 5)
        Xf = np.load(D / f"Xflat_{split}.npy").astype(np.float64)
        P = score_tokens(en, mk)
        block = {"n_per_noc": {int(k): int((noc == k).sum()) for k in range(1, 6)}}
        # plain top-k oracle per NOC (no-regression baseline)
        def per_noc(decode_set_fn):
            ok = {k: [] for k in range(1, 6)}
            for i in range(len(P)):
                S = set(np.where(y[i] == 1)[0]); K = int(noc[i]); order = list(np.argsort(P[i])[::-1])
                ok[K].append(S == decode_set_fn(i, order, K))
            return {int(k): round(float(np.mean(v)), 4) for k, v in ok.items() if v}
        block["plain_oracle"] = per_noc(lambda i, order, K: set(order[:K]))
        block["peel_oracle"] = {}
        for tau in taus:
            block["peel_oracle"][str(tau)] = per_noc(
                lambda i, order, K, tau=tau: gated_peel(np.expm1(Xf[i]), order, P[i], K, tau))
        return block

    peel = {"taus": taus}
    for sp in ("dev", "test"):
        b = eval_split(sp)
        if b is not None:
            peel[sp] = b
    M = json.load(open(res / "metrics.json")); M["peel"] = peel
    json.dump(M, open(res / "metrics.json", "w"), indent=2)
    # console summary (focus N4/N5, the wall)
    print(f"\n== Increment 5 PEEL decode — {sub} ==")
    for sp in ("dev", "test"):
        if sp not in peel: continue
        pl = peel[sp]["plain_oracle"]
        print(f"[{sp}] plain top-k oracle: " + "  ".join(f"N{k}={pl.get(k)}" for k in range(1, 6)))
        for tau in taus:
            po = peel[sp]["peel_oracle"][str(tau)]
            print(f"[{sp}] peel tau={tau}: " + "  ".join(f"N{k}={po.get(k)}" for k in range(1, 6)))
    print(f"wrote peel block → {res/'metrics.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sub"); ap.add_argument("data_dir")
    ap.add_argument("--taus", default="0,0.7,0.8,0.9")
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",")]
    main(a.sub, a.data_dir, taus)

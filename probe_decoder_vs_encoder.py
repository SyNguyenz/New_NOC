"""
probe_decoder_vs_encoder.py — reconcile F33 ("decoder under-read ~.13 > encoder ~.21") with the
ceiling probe ("soft-vote == decoder"). Measures decoder vs encoder-soft-vote at THREE levels on the
FULL real test, per NOC, so we know WHICH level the two sessions were each talking about:

  (A) full-set oracle EM          : all k donors correct in top-k         (what ceiling-probe used)
  (B) faint-minor recall          : the FAINTEST present donor in top-k   (what F33 likely used)
  (C) present-donor recall        : avg fraction of present donors in top-k

decoder   score = sigmoid(logits_cls)
softvote  score = sum_peaks softmax(attr)[:, :C]   (encoder per-peak readout)
softmax2  score = max_peaks  softmax(attr)[:, :C]   (alt aggregation, in case F33 used max)

Usage: python probe_decoder_vs_encoder.py [run_dir ...]
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from models.set_transformer import SetTransformerMixture

DATA = Path("data_insilico_w")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS = sys.argv[1:] or ["results/inc2_2d_sparse_seed42", "results/inc9_b4_sink_seed42"]

tok = np.load(DATA / "tokens8_test.npy").astype(np.float32)
msk = np.load(DATA / "mask_test.npy")
y   = np.load(DATA / "y_test_set.npy")
noc = np.clip(np.load(DATA / "noc_test.npy").astype(int), 1, 5)
phi = np.load(DATA / "phi_test.npy").astype(np.float32)
C = y.shape[1]


def build(cfg):
    return SetTransformerMixture(
        n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16), d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4), n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
        n_classes=cfg.get("n_classes", 45), n_noc=cfg.get("n_noc", 6), dropout=cfg.get("dropout", 0.1),
        cls_decoder=cfg.get("cls_decoder", "per_donor"), decoder_source=cfg.get("decoder_source", "encoded"),
        n_token_feats=cfg.get("n_token_feats", 8), encoder=cfg.get("encoder", "isab"),
        dec_layers=cfg.get("dec_layers", 2), num_embed=cfg.get("num_embed", "raw"),
        n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
        periodic_sigma=cfg.get("periodic_sigma", 1.0), aux_heads=cfg.get("aux_heads", False),
        sparse_attn=cfg.get("sparse_attn", False), attn_sink=int(cfg.get("attn_sink", 0) or 0),
        donor_recon=cfg.get("donor_recon", False),
    ).to(DEVICE)


@torch.no_grad()
def scores(rd):
    cfg = json.load(open(Path(rd) / "metrics.json"))["config"]
    m = build(cfg)
    sd = torch.load(Path(rd) / "best_model.pt", map_location=DEVICE)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    m.load_state_dict(sd, strict=False); m.eval()
    DEC, SV, MX = [], [], []
    for s in range(0, len(tok), 128):
        T = torch.from_numpy(tok[s:s+128]).to(DEVICE); M = torch.from_numpy(msk[s:s+128]).to(DEVICE)
        o = m(T, M)
        DEC.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
        a = F.softmax(o["logits_attr"], dim=-1)[:, :, :C] * M.unsqueeze(-1)
        SV.append(a.sum(1).cpu().numpy())
        MX.append(a.max(1).values.cpu().numpy())
    return np.concatenate(DEC), np.concatenate(SV), np.concatenate(MX)


def metrics(score):
    """returns dict noc-> (oracleEM, faintRecall, presentRecall)"""
    out = {}
    for k in range(1, 6):
        sel = np.where(noc == k)[0]
        if len(sel) == 0:
            continue
        em = fr = pr = 0
        for j in sel:
            top = set(np.argsort(score[j])[::-1][:k].tolist())
            pres = np.where(y[j] == 1)[0]
            faint = pres[int(np.argmin(phi[j][pres]))]
            em += int(set(pres.tolist()) == top)
            fr += int(faint in top)
            pr += len(top & set(pres.tolist())) / k
        n = len(sel)
        out[k] = (em / n, fr / n, pr / n)
    return out


names = [Path(r).name for r in RUNS]
R = {}
for rd in RUNS:
    dec, sv, mx = scores(rd)
    R[Path(rd).name] = {"decoder": metrics(dec), "softvote_sum": metrics(sv), "softvote_max": metrics(mx)}

LBL = [("full-set ORACLE EM", 0), ("FAINT-MINOR recall", 1), ("present-donor recall", 2)]
for nm in names:
    print(f"\n############### {nm} ###############")
    for lbl, idx in LBL:
        print(f"\n  -- {lbl}")
        print(f"    {'NOC':<5}{'decoder':>12}{'softvote_sum':>14}{'softvote_max':>14}{'sv_sum-dec':>13}")
        for k in range(1, 6):
            d = R[nm]["decoder"].get(k)
            ss = R[nm]["softvote_sum"].get(k)
            sm = R[nm]["softvote_max"].get(k)
            if d is None:
                continue
            print(f"    {k:<5}{d[idx]:>12.3f}{ss[idx]:>14.3f}{sm[idx]:>14.3f}{ss[idx]-d[idx]:>13.3f}")

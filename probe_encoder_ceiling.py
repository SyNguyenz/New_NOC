"""
probe_encoder_ceiling.py — is the ENCODER at its information ceiling, or is the decoder under-reading?

For each run, on the FULL real test, per NOC, compute two ORACLE EM numbers (true count handed in):
  decoder   = top-k of sigmoid(logits_cls)            -> what the per-donor decoder extracts
  softvote  = top-k of sum_peaks softmax(attr)[:, :C] -> what the encoder's per-peak readout contains

softvote reads ID straight off the encoder's per-peak attribution (the most direct encoder readout,
F33's encoder-information proxy). If softvote >> decoder at high NOC, the encoder already holds more
than the decoder uses -> the binding constraint is decoder under-read, encoder is (relatively) at ceiling.
If softvote ~= decoder and both low, the encoder itself is still the ceiling (needs more de-smoothing).

Usage:  python probe_encoder_ceiling.py [run_dir ...]
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from models.set_transformer import SetTransformerMixture

DATA = Path("data_insilico_w")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS = sys.argv[1:] or [
    "results/inc2_2d_sparse_seed42",
    "results/inc9_b1_recon_seed42",
    "results/inc9_b4_sink_seed42",
]

tok = np.load(DATA / "tokens8_test.npy").astype(np.float32)
msk = np.load(DATA / "mask_test.npy")
y   = np.load(DATA / "y_test_set.npy")
noc = np.clip(np.load(DATA / "noc_test.npy").astype(int), 1, 5)
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


def oracle_em(score, y, noc):
    out = {}
    for k in range(1, 6):
        sel = np.where(noc == k)[0]
        if len(sel) == 0:
            continue
        em = 0
        for j in sel:
            top = np.argsort(score[j])[::-1][:k]
            pred = np.zeros(C, int); pred[top] = 1
            em += (pred == y[j]).all()
        out[k] = em / len(sel)
    return out


@torch.no_grad()
def run(rd):
    cfg = json.load(open(Path(rd) / "metrics.json"))["config"]
    m = build(cfg)
    sd = torch.load(Path(rd) / "best_model.pt", map_location=DEVICE)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    m.load_state_dict(sd, strict=False); m.eval()
    DEC, SV = [], []
    for s in range(0, len(tok), 128):
        T = torch.from_numpy(tok[s:s+128]).to(DEVICE); M = torch.from_numpy(msk[s:s+128]).to(DEVICE)
        o = m(T, M)
        DEC.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
        a = F.softmax(o["logits_attr"], dim=-1)[:, :, :C]          # (b,N,C) drop background col
        a = a * M.unsqueeze(-1)                                     # zero padded peaks
        SV.append(a.sum(1).cpu().numpy())                           # (b,C) summed per-donor evidence
    return np.concatenate(DEC), np.concatenate(SV)


print(f"full real test n={len(tok)}  NOC dist={ {k:int((noc==k).sum()) for k in range(1,6)} }\n")
res = {}
for rd in RUNS:
    dec, sv = run(rd)
    res[Path(rd).name] = (oracle_em(dec, y, noc), oracle_em(sv, y, noc))

names = [Path(r).name for r in RUNS]
for tag, idx in [("DECODER oracle EM", 0), ("ENCODER soft-vote oracle EM", 1)]:
    print(f"== {tag} ==")
    print(f"  {'NOC':<5}" + "".join(f"{n[-16:]:>18}" for n in names))
    for k in range(1, 6):
        print(f"  {k:<5}" + "".join(f"{res[n][idx].get(k, float('nan')):>18.3f}" for n in names))
    print()

print("== soft-vote MINUS decoder (encoder headroom the decoder leaves on the table) ==")
print(f"  {'NOC':<5}" + "".join(f"{n[-16:]:>18}" for n in names))
for k in range(1, 6):
    print(f"  {k:<5}" + "".join(f"{res[n][1].get(k,float('nan'))-res[n][0].get(k,float('nan')):>18.3f}" for n in names))

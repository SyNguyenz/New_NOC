"""
probe_underread_ci.py — how reliable is the "decoder under-read ~.06 at N5"?

For each base seed, on FULL real test, per NOC, decompose the decoder-vs-encoder-readout gap:
  dec      = decoder oracle EM
  svmax    = encoder soft-vote (max-pool attr) oracle EM
  union    = sample correct if decoder OR svmax correct  (recoverable by SOME no-retrain readout)
  b        = #samples decoder WRONG but svmax RIGHT   (decoder leaves these on the table)
  c        = #samples decoder RIGHT but svmax WRONG   (svmax loses these)
  net      = (b - c)/n  == svmax_EM - dec_EM     (the headline "~.06")
  mcnemar  = (|b-c|-1)^2 / (b+c)                  (chi2, ~3.84 = p.05, ~6.6 = p.01)

Aggregated mean / min / max across seeds -> a 3-seed read on whether the gap is real or seed noise.
Usage: python probe_underread_ci.py [run_dir ...]   (default = 3 base seeds)
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
    "results/inc2_2d_sparse_seed43",
    "results/inc2_2d_sparse_seed44",
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


@torch.no_grad()
def correctness(rd):
    """returns dec_correct, svmax_correct : bool arrays over test (oracle top-k)."""
    cfg = json.load(open(Path(rd) / "metrics.json"))["config"]
    m = build(cfg)
    sd = torch.load(Path(rd) / "best_model.pt", map_location=DEVICE)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    m.load_state_dict(sd, strict=False); m.eval()
    DEC, MX = [], []
    for s in range(0, len(tok), 128):
        T = torch.from_numpy(tok[s:s+128]).to(DEVICE); M = torch.from_numpy(msk[s:s+128]).to(DEVICE)
        o = m(T, M)
        DEC.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
        a = F.softmax(o["logits_attr"], dim=-1)[:, :, :C] * M.unsqueeze(-1)
        MX.append(a.max(1).values.cpu().numpy())
    dec, mx = np.concatenate(DEC), np.concatenate(MX)
    dc = np.zeros(len(tok), bool); mc = np.zeros(len(tok), bool)
    for j in range(len(tok)):
        k = noc[j]; yt = set(np.where(y[j] == 1)[0].tolist())
        dc[j] = set(np.argsort(dec[j])[::-1][:k].tolist()) == yt
        mc[j] = set(np.argsort(mx[j])[::-1][:k].tolist()) == yt
    return dc, mc


PER = {k: {m: [] for m in ["dec", "svmax", "union", "b", "c", "net", "mcnemar"]} for k in (3, 4, 5)}
for rd in RUNS:
    dc, mc = correctness(rd)
    print(f"\n{Path(rd).name}")
    for k in (3, 4, 5):
        sel = noc == k; n = sel.sum()
        d = dc[sel]; m = mc[sel]
        b = int((~d & m).sum()); c = int((d & ~m).sum())
        rec = dict(dec=d.mean(), svmax=m.mean(), union=(d | m).mean(),
                   b=b, c=c, net=(b - c) / n,
                   mcnemar=((abs(b - c) - 1) ** 2 / (b + c)) if (b + c) else 0.0)
        for kk, vv in rec.items():
            PER[k][kk].append(vv)
        print(f"  N{k} (n={n}): dec={rec['dec']:.3f} svmax={rec['svmax']:.3f} union={rec['union']:.3f} "
              f"| b(dec_wrong,svmax_right)={b} c(dec_right,svmax_wrong)={c} net={rec['net']:+.3f} mcnemar={rec['mcnemar']:.1f}")

print("\n================ 3-seed summary (mean [min..max]) ================")
for k in (3, 4, 5):
    def ms(key):
        v = PER[k][key]; return f"{np.mean(v):.3f} [{min(v):.3f}..{max(v):.3f}]"
    print(f"\nNOC{k}:")
    print(f"  decoder EM         : {ms('dec')}")
    print(f"  svmax  EM          : {ms('svmax')}")
    print(f"  union  EM (recover): {ms('union')}")
    print(f"  net (svmax-dec)    : {ms('net')}")
    print(f"  union-dec (gross)  : {np.mean([PER[k]['union'][i]-PER[k]['dec'][i] for i in range(len(RUNS))]):.3f}"
          f" [{min(PER[k]['union'][i]-PER[k]['dec'][i] for i in range(len(RUNS))):.3f}.."
          f"{max(PER[k]['union'][i]-PER[k]['dec'][i] for i in range(len(RUNS))):.3f}]")
    print(f"  mcnemar chi2       : {ms('mcnemar')}  (3.84=p.05, 6.63=p.01)")

"""
probe_entanglement.py — EVAL-ONLY feasibility probe for DIRECTED disentangling (the post-VIB question).

VIB (undirected KL bottleneck) traded down: it cut train memorization but lost signal too. The refined
hypothesis: low-phi minor signal is ENTANGLED with combo-context in the per-donor latent. A truly
per-donor (compositional) latent D_d should encode ONLY donor d's own evidence — it should NOT be able
to predict WHO ELSE is in the mixture (d's own alleles do not determine the other donors' identities).

ENTANGLEMENT METRIC = co-contributor decodability. Fit a linear map  D_d -> {is donor j present, j!=d}
on TRAIN; AUROC above chance = the donor latent has absorbed combo identity (leakage / memorization
substrate). Compare TRAIN (seen combos) vs DEV (novel combos), and minorw (no conditioning) vs
repA geno_query (genotype-conditioned queries — the candidate DIRECTED lever).

  * high co-contributor AUROC, train>>dev          => latent memorizes the combo (entanglement confirmed).
  * geno_query AUROC < minorw AUROC                 => directed conditioning DISENTANGLES -> GO.
  * geno_query AUROC ~ minorw AUROC                 => this directed lever does not disentangle -> NO-GO.

Usage: python probe_entanglement.py
"""
import json
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

DATA = Path("data_insilico_w")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture

tok = np.load(DATA / "tokens8_train.npy").astype(np.float32)
msk = np.load(DATA / "mask_train.npy"); ymat = np.load(DATA / "y_train_set.npy").astype(np.float32)
noc = np.load(DATA / "noc_train.npy").astype(int)

def dev_mask_seed0(y, noc, combo_frac=0.15, noc1_frac=0.06, seed=0):
    rng = np.random.default_rng(seed); noc = np.clip(noc.astype(int), 1, 5); N = len(noc); m = np.zeros(N, bool)
    for k in [2, 3, 4, 5]:
        idx = np.where(noc == k)[0]; combos = {}
        for i in idx: combos.setdefault(tuple(np.where(y[i] == 1)[0].tolist()), []).append(i)
        uniq = list(combos); rng.shuffle(uniq)
        for c in uniq[:max(1, int(round(len(uniq) * combo_frac)))]: m[combos[c]] = True
    idx1 = np.where(noc == 1)[0]
    m[rng.choice(idx1, size=int(round(len(idx1) * noc1_frac)), replace=False)] = True
    return m
dmask = dev_mask_seed0(ymat, noc)

def build(run):
    cfg = json.load(open(Path("results") / run / "metrics.json"))["config"]
    dg = dgm = None
    if cfg.get("geno_query", False):
        gp = DATA / "donor_geno.npy"
        if not gp.exists(): gp = Path("data") / "donor_geno.npy"
        dg = torch.from_numpy(np.load(gp).astype(np.float32))
        dgm = torch.from_numpy(np.load(gp.parent / "donor_geno_mask.npy"))
    m = SetTransformerMixture(
        n_loci=cfg.get("n_loci",24), d_locus=cfg.get("d_locus",16), d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4), n_isab=cfg.get("n_isab",2), m_inducing=cfg.get("m_inducing",32),
        n_classes=cfg.get("n_classes",45), n_noc=cfg.get("n_noc",6), dropout=cfg.get("dropout",0.1),
        cls_decoder=cfg.get("cls_decoder","pooled"), decoder_source=cfg.get("decoder_source","encoded"),
        n_token_feats=cfg.get("n_token_feats",8), encoder=cfg.get("encoder","isab"), dec_layers=cfg.get("dec_layers",2),
        num_embed=cfg.get("num_embed","raw"), n_freq=cfg.get("n_freq",8), d_num_emb=cfg.get("d_num_emb",8),
        periodic_sigma=cfg.get("periodic_sigma",1.0), aux_heads=cfg.get("aux_heads",False),
        d_proj=cfg.get("d_proj",64), sparse_attn=cfg.get("sparse_attn",False),
        geno_query=cfg.get("geno_query",False), donor_geno=dg, donor_geno_mask=dgm,
        vib=cfg.get("vib",False),
    ).to(DEVICE)
    sd = torch.load(Path("results")/run/"best_model.pt", map_location=DEVICE)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    m.load_state_dict(sd, strict=False); m.eval()
    return m, cfg

@torch.no_grad()
def donor_latents(model, cfg, idx, bs=256):
    """Return per-donor latent D (len(idx), 45, d) = the PerDonorDecoder pre-score latent + sigmoid logits."""
    dec = model.cls_decoder_module
    Ds, Ps = [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i+bs]
        t = torch.from_numpy(tok[b]).to(DEVICE); mk = torch.from_numpy(msk[b]).to(DEVICE)
        x0, H, pad = model._encode_set(t, mk)
        src = x0 if cfg.get("decoder_source") == "raw" else H
        q = dec.donor_queries
        if cfg.get("geno_query", False):
            q = q + model._encode_geno().unsqueeze(0)
        D = q.expand(t.size(0), -1, -1)
        for mab in dec.layers:
            D = mab(D, src, key_padding_mask=pad)
        if getattr(dec, "vib", False):
            D = dec.to_mu(D)                       # eval-mode latent (no sampling)
        logits = torch.einsum("bnd,nd->bn", D, dec.score_w) + dec.score_b
        Ds.append(D.cpu().numpy()); Ps.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(Ds), np.concatenate(Ps)

def gather(model, cfg, idx, noc_keep=(4,5), cap=4000):
    sel = idx[np.isin(noc[idx], noc_keep)]
    if len(sel) > cap: sel = np.random.default_rng(0).choice(sel, cap, replace=False)
    D, P = donor_latents(model, cfg, sel)
    Xd, Yco = [], []          # latent of each TRUE donor ; its co-contributor 45-vec (self zeroed)
    for j, gi in enumerate(sel):
        tc = np.where(ymat[gi] == 1)[0]
        for d in tc:
            co = ymat[gi].copy(); co[d] = 0.0
            Xd.append(D[j, d]); Yco.append(co)
    return np.array(Xd), np.array(Yco)

def entanglement(run, label):
    model, cfg = build(run)
    tr_idx = np.where(~dmask)[0]; dv_idx = np.where(dmask)[0]
    Xtr, Ytr = gather(model, cfg, tr_idx)
    Xdv, Ydv = gather(model, cfg, dv_idx)
    # fit linear co-contributor predictor on train latents
    rg = Ridge(alpha=10.0).fit(Xtr, Ytr)
    def auc(X, Y, Yhat):
        # micro-AUROC over off-self entries (cols that are ever positive)
        cols = np.where(Y.sum(0) > 0)[0]
        yt = Y[:, cols].ravel(); ys = Yhat[:, cols].ravel()
        return roc_auc_score(yt, ys)
    a_tr = auc(Xtr, Ytr, rg.predict(Xtr)); a_dv = auc(Xdv, Ydv, rg.predict(Xdv))
    print(f"  {label:24s}  co-contributor AUROC: train {a_tr:.3f} | dev {a_dv:.3f}   (0.5=disentangled)")
    return a_tr, a_dv

print("ENTANGLEMENT = can donor d's latent predict WHO ELSE is in the mix? (>0.5 = combo leaked into latent)\n")
entanglement("inc6_minorw_seed42",   "minorw (no conditioning)")
entanglement("inc3_repA_genoq_seed42","repA geno_query (directed)")
print("\nREAD: geno_query AUROC < minorw => directed conditioning disentangles -> GO. ~equal => NO-GO.")

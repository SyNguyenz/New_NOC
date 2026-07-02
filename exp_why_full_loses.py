"""Why does the FULL EuroForMix-class efm_rerank NOT beat the crude uniform phi_rerank? Two candidate causes:
(a) generator data is bad / efm mis-specified, or (b) LOP structure: the rerank score = z(cls)+a*z(log phi)
is a PRODUCT-OF-EXPERTS; the phi channel's value scales with its INDEPENDENCE from cls, not its accuracy.
Test: measure, per sample, each phi's FIDELITY to true phi (is it the better deconvolution?) AND its
REDUNDANCY with cls (corr). If efm is MORE accurate but MORE redundant -> (b) explains the paradox: a
better phi that agrees more with cls adds less in the pool. (Not a fake gain, not bad data.)"""
import numpy as np, importlib.util
from pathlib import Path
from sklearn.metrics import roc_auc_score
PROJ = Path("."); DATA = PROJ / "data_insilico_w"; GENO = PROJ / "data"; CACHE = PROJ / "cache_insilico"; K = 45
def lm(n, p): s = importlib.util.spec_from_file_location(n, str(p)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pr = lm("pr", PROJ / "inc22_clean" / "phi_rerank.py"); ef = lm("ef", PROJ / "efm_rerank.py")
g = np.load(GENO / "donor_geno.npy").astype(np.float32); gm = np.load(GENO / "donor_geno_mask.npy")
def load(sp):
    d = {k: np.load(DATA / f"{k}_{sp}.npy") for k in ["tokens8", "mask", "phi", "noc"]}
    d["y"] = (d["phi"] > 0).astype(np.float32); d["cls"] = np.load(CACHE / f"cls_{sp}.npy")
    szp = DATA / f"size_{sp}.npy"; d["size"] = np.load(szp).astype(np.float64) if szp.exists() else np.zeros(d["mask"].shape)
    return d
te = load("test")
UNI = pr.deconv_phi(te["tokens8"], te["mask"].astype(bool), g, gm, n_iters=12)            # crude uniform
EFM = ef.deconv_efm(te["tokens8"], te["mask"].astype(bool), te["size"], g, gm, xi=0.08)     # full EuroForMix-class

def corr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9: return np.nan
    return np.corrcoef(a, b)[0, 1]
def stats(phi, focus):
    red, fid, auc = [], [], []
    for i in np.where(np.isin(te["noc"], focus))[0]:
        lp = np.log(phi[i] + 1e-6)
        red.append(corr(lp, te["cls"][i]))                       # redundancy with cls
        fid.append(corr(phi[i], te["phi"][i]))                   # fidelity to TRUE phi
        yi = (te["y"][i] > 0.5).astype(int)
        if 0 < yi.sum() < K: auc.append(roc_auc_score(yi, phi[i]))   # does phi rank present donors up?
    return np.nanmean(red), np.nanmean(fid), np.nanmean(auc)

print("=== why full(efm) does not beat crude(uniform): fidelity vs redundancy ===")
for focus, tag in [((3, 4, 5), "N3-5"), ((5,), "N5")]:
    ru, fu, au = stats(UNI, focus); re, fe, ae = stats(EFM, focus)
    print(f"\n  [{tag}]                     redundancy(corr w/ cls)   fidelity(corr w/ true phi)   AUC(phi->present)")
    print(f"    uniform (crude phi)            {ru:+.3f}                    {fu:+.3f}                   {au:.3f}")
    print(f"    efm     (full EuroForMix)      {re:+.3f}                    {fe:+.3f}                   {ae:.3f}")
    print(f"    delta (efm - uniform)          {re-ru:+.3f}  {'MORE redundant' if re>ru else 'less redundant'}     "
          f"{fe-fu:+.3f}  {'MORE accurate' if fe>fu else 'less accurate'}     {ae-au:+.3f}")
print("\n  => if efm is MORE accurate (fidelity/AUC up) but MORE redundant (corr up), the LOP dividend shrinks")
print("     -> full loses in the POOL despite being the better deconvolution. Structural, not a fake gain/bad data.")

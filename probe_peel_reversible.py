"""
Reversibility test on the inc7 encoder (real test data only — no synthetic/real confound).
For N5 samples, MASK OUT the major (top-3 by height) donors' peaks and re-run the model.
Read the model's OWN attribution head accuracy on the remaining MINOR peaks, full-context vs
major-masked, split by whether the SET head dropped that minor.

If major-masking LIFTS the dropped minors (0.30 -> high), the minor info is in the INPUT and
the encoder represents it once the majors no longer entangle it => context-entanglement that is
REVERSIBLE, not a fixed capacity/washing loss. This is the F31 probe applied to the trained inc7 model.
"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc7_masspool_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
dg=dgm=None
if cfg.get("geno_query"):
    dg=torch.from_numpy(np.load(DATA/"donor_geno.npy").astype(np.float32)); dgm=torch.from_numpy(np.load(DATA/"donor_geno_mask.npy"))
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=True,sparse_attn=cfg.get("sparse_attn",False),geno_query=cfg.get("geno_query",False),
    donor_geno=dg,donor_geno_mask=dgm,vib=cfg.get("vib",False),mass_pool=cfg.get("mass_pool",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()

tk=np.load(DATA/"tokens8_test.npy")[:,:,:n_tok].astype(np.float32)
mk=np.load(DATA/"mask_test.npy").astype(bool); at=np.load(DATA/"attr_test.npy"); nc=np.load(DATA/"noc_test.npy"); LH=tk[:,:,2]
yp=np.load(RUN/"y_test_pred.npy"); yt=np.load(RUN/"y_test_true.npy")

@torch.no_grad()
def attr_pred(tokens, mask):
    out=model(torch.from_numpy(tokens).to(DEV), torch.from_numpy(mask).to(DEV))
    return out["logits_attr"].argmax(-1).cpu().numpy()

idx=np.where(nc==5)[0]
# build major-masked copy: drop peaks whose true donor is top-3 by height
mk2=mk.copy()
majors_of={}
for i in idx:
    a=at[i]; valid=np.where(a>=0)[0]
    dh={int(d):float(np.exp(LH[i][a==d]).sum()) for d in np.unique(a[valid])}
    order=sorted(dh,key=lambda d:-dh[d]); majors=set(order[:3]); majors_of[i]=(order,majors)
    for p in valid:
        if int(a[p]) in majors: mk2[i,p]=False   # hide major peaks from the encoder

ap_full=attr_pred(tk[idx], mk[idx])
ap_mask=attr_pred(tk[idx], mk2[idx])

# accuracy on MINOR peaks (rank>=3), split kept/dropped by the set head
res={("kept","full"):[],("kept","mask"):[],("drop","full"):[],("drop","mask"):[]}
for k,i in enumerate(idx):
    a=at[i]; order,majors=majors_of[i]; ro={d:r for r,d in enumerate(order)}
    for d in np.unique(a[a>=0]):
        if ro[int(d)]<3: continue          # minors only
        pk=np.where(a==d)[0]
        grp = "drop" if (yt[i,d]==1 and yp[i,d]==0) else "kept"
        res[(grp,"full")].append((ap_full[k][pk]==d).mean())
        res[(grp,"mask")].append((ap_mask[k][pk]==d).mean())

print(f"loaded {RUN.name} mass_pool={cfg.get('mass_pool')}")
print("\nN5 minor-peak attribution accuracy (model's own attr head), full-context vs MAJOR-MASKED input:")
for grp in ("kept","drop"):
    f=np.mean(res[(grp,"full")]); m=np.mean(res[(grp,"mask")]); n=len(res[(grp,"full")])
    print(f"  {grp:5s} minors (n={n:4d}): full-context={f:.3f}  ->  major-masked={m:.3f}   (delta {m-f:+.3f})")
print("\n  reversible (mask >> full) on DROPPED minors => info is in the INPUT, encoder represents it")
print("  once majors are removed = context-entanglement, NOT permanent capacity/washing loss.")

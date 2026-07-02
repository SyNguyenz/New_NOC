"""
ENCODER exact sub-mechanism: is a faint minor's H representation CONTEXT-DETERMINED (entangled with
the co-occurring donors), i.e. does the SAME donor's encoded rep move with the combo, and is it pulled
toward its co-occurring MAJORS?

Two measures on the encoder output H (per-donor mean-pooled over the donor's true peaks), N5 real test:
  (1) cross-combo SELF-stability: for a donor appearing in many samples, mean pairwise cosine of its
      H-pooled rep across appearances — as a MAJOR vs as a MINOR. Lower for minors => its rep is more
      combo-variable (context-determined) = the entanglement that hurts novel readability.
  (2) pull toward co-occurring majors: cosine(minor rep, its same-sample MAJORS' mean rep) vs
      cosine(minor rep, a RANDOM other donor's rep). Higher same-sample-major cosine => the encoder
      mixes major content into the minor's representation.
"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0,".")
from models.set_transformer import SetTransformerMixture

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc7_masspool_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"
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
print(f"loaded {RUN.name}")

tk=np.load(DATA/"tokens8_test.npy")[:,:,:n_tok].astype(np.float32); mk=np.load(DATA/"mask_test.npy").astype(bool)
at=np.load(DATA/"attr_test.npy"); noc=np.load(DATA/"noc_test.npy"); LH=tk[:,:,2]

@torch.no_grad()
def encode(idxs,bs=128):
    rows=[]  # (sample, donor, rank, Hpooled, major_pool)
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        _,H,_=model._encode_set(t,m); H=H.cpu().numpy()
        for j,gi in enumerate(sel):
            a=at[gi]; valid=np.where(a>=0)[0]
            if len(valid)==0: continue
            dh={int(d):float(np.exp(LH[gi][a==d]).sum()) for d in np.unique(a[valid])}
            order=sorted(dh,key=lambda d:-dh[d]); ro={d:r for r,d in enumerate(order)}
            reps={int(d):H[j][a==d].mean(0) for d in np.unique(a[valid])}
            maj_pool=np.mean([reps[d] for d in order[:3]],0)
            for d in np.unique(a[valid]):
                rows.append((int(gi),int(d),ro[int(d)],reps[int(d)],maj_pool))
    return rows
rows=encode(np.where(noc==5)[0])

def cos(a,b):
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

# (1) cross-combo self-stability per donor, as major vs minor
from collections import defaultdict
byd_major=defaultdict(list); byd_minor=defaultdict(list)
for _,d,r,rep,_ in rows:
    (byd_major if r<3 else byd_minor)[d].append(rep)
def mean_self_cos(byd):
    vals=[]
    for d,reps in byd.items():
        if len(reps)<2: continue
        R=np.stack(reps);
        for i in range(len(R)):
            for k in range(i+1,len(R)):
                vals.append(cos(R[i],R[k]))
    return np.mean(vals),len(vals)
maj_self,nM=mean_self_cos(byd_major); min_self,nm=mean_self_cos(byd_minor)
print("\n== (1) cross-combo SELF-stability of a donor's H rep (1=identical across combos) ==")
print(f"   as MAJOR: mean self-cosine {maj_self:.3f}  (n_pairs={nM})")
print(f"   as MINOR: mean self-cosine {min_self:.3f}  (n_pairs={nm})")
print(f"   => minor < major by {maj_self-min_self:+.3f}  (more negative = minor rep more combo-variable = context-determined)")

# (2) pull toward co-occurring majors (minors only), vs random donor
rng=np.random.default_rng(0); all_reps=[rep for _,_,_,rep,_ in rows]
pull=[]; ctrl=[]
for _,d,r,rep,mp in rows:
    if r<3: continue
    pull.append(cos(rep,mp))
    ctrl.append(cos(rep, all_reps[rng.integers(len(all_reps))]))
print("\n== (2) is the minor's rep pulled toward its co-occurring MAJORS? ==")
print(f"   cosine(minor, same-sample majors) = {np.mean(pull):.3f}")
print(f"   cosine(minor, random donor rep)   = {np.mean(ctrl):.3f}")
print(f"   => pull-minus-control {np.mean(pull)-np.mean(ctrl):+.3f}  (>0 = encoder mixes major content into the minor)")

# (3) within-sample baselines: is 0.92 just 'all peaks in a sample look alike', or are minors EXTRA-pulled?
byidx=defaultdict(list)
for gi,d,r,rep,_ in rows: byidx[gi].append((r,rep))
mm=[]; nn_=[]; xmin=[]
for gi,lst in byidx.items():
    majs=[rep for r,rep in lst if r<3]; mins=[rep for r,rep in lst if r>=3]
    for i in range(len(majs)):
        for k in range(i+1,len(majs)): mm.append(cos(majs[i],majs[k]))
    for i in range(len(mins)):
        for k in range(i+1,len(mins)): nn_.append(cos(mins[i],mins[k]))
print("\n== (3) within-SAME-sample cosine baselines (controls for shared global context) ==")
print(f"   major-major = {np.mean(mm):.3f} | minor-minor = {np.mean(nn_):.3f} | minor-major = {np.mean(pull):.3f}")
print(f"   minor's cross-combo SELF-stability = {min_self:.3f}  < its similarity to CURRENT majors {np.mean(pull):.3f}")
print("   => if minor-major >= major-major AND > minor's self-stability: the minor's rep is determined")
print("      MORE by its current co-occurring majors than by its own identity = entanglement, not artifact.")

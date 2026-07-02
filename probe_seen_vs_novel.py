"""
CLEAN seen-vs-novel combo test — both groups SYNTHETIC (same make_insilico generation), so NO
synthetic<->real confound. Isolates ONLY combo-novelty for the encoder representation.

Model-TRAIN (seen combos) vs Model-DEV (novel combos) — the exact seed=0 carve the model trained with.
Independent linear probe rep[peak]->donor id fit on a slice of model-TRAIN, evaluated on a DISJOINT
slice of model-TRAIN (seen) vs model-DEV (novel), restricted to N5 FAINT minors (rank>=3) and BINNED
by the minor/major template ratio so we compare equally-faint donors.

If seen >> novel on matched faintness  => encoder rep is COMBO-DEPENDENT (combinatorial generalization
  failure lives partly in the representation).
If seen ~= novel                       => encoder rep generalizes; the failure is downstream (decode).
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0,".")
from models.set_transformer import SetTransformerMixture

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc7_masspool_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
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
print(f"loaded {RUN.name} mass_pool={cfg.get('mass_pool')}")

# ---- exact seed=0 dev carve (== make_dev_split / measure_insilico_oracle) ----
y=np.load(DATA/"y_train_set.npy"); noc=np.load(DATA/"noc_train.npy")
tk=np.load(DATA/"tokens8_train.npy")[:,:,:n_tok].astype(np.float32); mk=np.load(DATA/"mask_train.npy").astype(bool)
at=np.load(DATA/"attr_train.npy"); LH=tk[:,:,2]
def dev_mask_seed0(y,noc,combo_frac=0.15,noc1_frac=0.06,seed=0):
    rng=np.random.default_rng(seed); noc=np.clip(noc.astype(int),1,5); N=len(noc); m=np.zeros(N,bool)
    for k in (2,3,4,5):
        idx=np.where(noc==k)[0]; combos={}
        for i in idx: combos.setdefault(tuple(np.where(y[i]==1)[0].tolist()),[]).append(i)
        uniq=list(combos); rng.shuffle(uniq)
        for c in uniq[:max(1,int(round(len(uniq)*combo_frac)))]: m[combos[c]]=True
    idx1=np.where(noc==1)[0]; m[rng.choice(idx1,size=int(round(len(idx1)*noc1_frac)),replace=False)]=True
    return m
dmask=dev_mask_seed0(y,noc)
seen_idx=np.where(~dmask)[0]; novel_idx=np.where(dmask)[0]
print(f"model-SEEN (train) {len(seen_idx)} | model-NOVEL (dev) {len(novel_idx)}  (both synthetic, seed=0 carve)")

@torch.no_grad()
def encode(idxs,bs=128):
    X=[];D=[];R=[];N=[];RATIO=[]
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        _,H,_=model._encode_set(t,m); H=H.cpu().numpy()
        for j,gi in enumerate(sel):
            a=at[gi]; valid=np.where(a>=0)[0]
            if len(valid)==0: continue
            dh={int(d):float(np.exp(LH[gi][a==d]).sum()) for d in np.unique(a[valid])}
            order=sorted(dh,key=lambda d:-dh[d]); ro={d:r for r,d in enumerate(order)}; maj=dh[order[0]]
            don=a[valid]
            X.append(H[j][valid]); D.append(don)
            R.append(np.array([ro[int(d)] for d in don])); N.append(np.full(len(valid),noc[gi]))
            RATIO.append(np.array([dh[int(d)]/maj for d in don]))
    return np.concatenate(X),np.concatenate(D).astype(int),np.concatenate(R).astype(int),np.concatenate(N).astype(int),np.concatenate(RATIO)

rng=np.random.default_rng(0)
fit_sel=rng.choice(seen_idx,size=min(7000,len(seen_idx)),replace=False)
seen_eval=rng.choice(np.setdiff1d(seen_idx,fit_sel),size=4000,replace=False)
novel_eval=novel_idx
Hf,Df,_,_,_=encode(fit_sel)
He,De,Re,Ne,RAe=encode(seen_eval)      # SEEN combos
Hn,Dn,Rn,Nn,RAn=encode(novel_eval)     # NOVEL combos
print(f"probe-fit peaks {len(Df)} | seen-eval {len(De)} | novel-eval {len(Dn)}")

clf=nn.Linear(Hf.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
Xt=torch.from_numpy(Hf).to(DEV); yt=torch.from_numpy(Df).long().to(DEV); lf=nn.CrossEntropyLoss()
for ep in range(60):
    p=torch.randperm(len(yt),device=DEV)
    for s in range(0,len(yt),8192):
        b=p[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
@torch.no_grad()
def acc(H,D,mask):
    if mask.sum()==0: return float('nan'),0
    pr=clf(torch.from_numpy(H[mask]).to(DEV)).argmax(1).cpu().numpy(); return (pr==D[mask]).mean(),int(mask.sum())

print("\n=== N5 FAINT-minor (rank>=3) donor-id readability from encoder H — SEEN vs NOVEL combo (both synthetic) ===")
print("  ratio bin = minor template / major template (match faintness across the two groups)")
print(f"  {'ratio bin':>12} | {'SEEN acc (n)':>18} | {'NOVEL acc (n)':>18}")
bins=[(0.0,0.02),(0.02,0.05),(0.05,0.10),(0.10,0.25),(0.0,1.0)]
for lo,hi in bins:
    ms=(Ne==5)&(Re>=3)&(RAe>=lo)&(RAe<hi)
    mn=(Nn==5)&(Rn>=3)&(RAn>=lo)&(RAn<hi)
    a_s,ns=acc(He,De,ms); a_n,nn_=acc(Hn,Dn,mn)
    tag=f"[{lo:.2f},{hi:.2f})" if hi<1.0 else "ALL faint"
    print(f"  {tag:>12} | {a_s:.3f} ({ns:5d})      | {a_n:.3f} ({nn_:5d})")
print("\n  seen>>novel at matched ratio => encoder rep is combo-dependent; seen~=novel => rep generalizes.")

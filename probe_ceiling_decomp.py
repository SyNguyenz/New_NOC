"""
Is N5 set-oracle 0.769 a LOW ceiling, or just (a) compounding of a strong per-donor signal and
(b) a weak soft-vote readout?

On the SAME encoder H (combo-disjoint probe):
  (A) decompose set-EM into PER-DONOR inclusion accuracy by height-rank (is the true donor in top-NOC?).
      set-EM ~= product of per-donor inclusions; if majors~1.0 and only the faintest minor leaks,
      then 0.769 is 'strong per-donor, compounded', not a weak signal.
  (B) failure anatomy: among N5 misses, how many donors are missed and at which rank.
  (C) ceiling test: replace the LINEAR soft-vote with a 2-layer MLP readout of H. If N5 set-EM jumps,
      0.769 was the readout's limit, not H's information ceiling => a better decoder can still win.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture

DATA = Path("data_insilico_w")
RUN  = Path(sys.argv[1] if len(sys.argv) > 1 else "results/inc8_v2_vicreg_inv_seed42")
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
cfg = json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
dg=dgm=None
if cfg.get("geno_query"):
    dg=torch.from_numpy(np.load(DATA/"donor_geno.npy").astype(np.float32)); dgm=torch.from_numpy(np.load(DATA/"donor_geno_mask.npy"))
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),geno_query=cfg.get("geno_query",False),
    donor_geno=dg,donor_geno_mask=dgm,vib=cfg.get("vib",False),mass_pool=cfg.get("mass_pool",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()
print(f"loaded {RUN.name}")

def load(s):
    return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
            np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))

@torch.no_grad()
def encode_peaks(tk,mk,at,idxs,bs=128):
    HH=[];DON=[];SID=[];RNK=[]
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        _,H,_=model._encode_set(t,m); H=H.cpu().numpy()
        for j,gi in enumerate(sel):
            a=at[gi]; v=np.where(a>=0)[0]
            if len(v)==0: continue
            lh=tk[gi][:,2]; dh={int(d):float(np.exp(lh[a==d]).sum()) for d in np.unique(a[v])}
            order=sorted(dh,key=lambda d:-dh[d]); ro={d:r for r,d in enumerate(order)}
            HH.append(H[j][v]); DON.append(a[v]); SID.append(np.full(len(v),gi)); RNK.append([ro[int(d)] for d in a[v]])
    return np.concatenate(HH),np.concatenate(DON).astype(int),np.concatenate(SID).astype(int),np.concatenate(RNK).astype(int)

def fit(probe,Htr,dtr,epochs=60):
    opt=torch.optim.Adam(probe.parameters(),lr=1e-2,weight_decay=1e-4)
    Xt=torch.from_numpy(Htr).to(DEV); yt=torch.from_numpy(dtr).long().to(DEV); lf=nn.CrossEntropyLoss()
    for ep in range(epochs):
        perm=torch.randperm(len(yt),device=DEV)
        for s in range(0,len(yt),8192):
            b=perm[s:s+8192]; opt.zero_grad(); lf(probe(Xt[b]),yt[b]).backward(); opt.step()
    return probe

tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0)
Htr,dtr,_,_=encode_peaks(tk_tr,mk_tr,at_tr,rng.choice(len(at_tr),size=6000,replace=False))
lin=fit(nn.Linear(Htr.shape[1],45).to(DEV),Htr,dtr)
mlp=fit(nn.Sequential(nn.Linear(Htr.shape[1],128),nn.ReLU(),nn.Linear(128,45)).to(DEV),Htr,dtr)
print(f"probes fit on {len(dtr)} peaks (linear + 2-layer MLP)")

tk,mk,at=load("test")
true_set={}; noc={}; rnk_of={}
for gi in range(len(at)):
    a=at[gi]; d=np.unique(a[a>=0])
    if len(d)==0: continue
    true_set[gi]=set(int(x) for x in d); noc[gi]=len(d)
keep=np.array(sorted(true_set))
H,don,sid,rnk=encode_peaks(tk,mk,at,keep)
# rank of each true donor per sample
for p in range(len(sid)): rnk_of[(int(sid[p]),int(don[p]))]=int(rnk[p])

@torch.no_grad()
def votes(probe):
    sm=torch.softmax(probe(torch.from_numpy(H).to(DEV)),1).cpu().numpy()
    v=defaultdict(lambda:np.zeros(45))
    for p in range(len(sid)): v[int(sid[p])]+=sm[p]
    return v

def setEM(v,k=5):
    gis=[g for g in keep if noc[g]==k]; em=[]
    for g in gis:
        pred=set(int(x) for x in np.argsort(v[g])[::-1][:k]); em.append(pred==true_set[g])
    return float(np.mean(em)),gis

vlin=votes(lin); vmlp=votes(mlp)
print("\n=== (C) CEILING test: does a stronger readout of the SAME H beat soft-vote? (N5 set-EM) ===")
for nm,v in [("linear soft-vote",vlin),("2-layer MLP    ",vmlp)]:
    em,_=setEM(v,5); print(f"   {nm}: N5 set-EM = {em:.3f}")

print("\n=== (A) per-donor INCLUSION by height-rank (linear), N5: is the true donor in top-5? ===")
gis5=[g for g in keep if noc[g]==5]
byr=defaultdict(list)
for g in gis5:
    top5=set(int(x) for x in np.argsort(vlin[g])[::-1][:5])
    for d in true_set[g]:
        byr[rnk_of[(g,d)]].append(d in top5)
prod=1.0
for r in sorted(byr):
    acc=np.mean(byr[r]); prod*=acc
    print(f"   rank r{r}{'=major' if r==0 else ('=faintest' if r==4 else '')}: inclusion={acc:.3f}  (n={len(byr[r])})")
print(f"   => product of per-donor inclusions ~= {prod:.3f}  (vs measured set-EM)")
print(f"   per-peak donor-id was ~0.77; per-DONOR inclusion is far higher (aggregation over a donor's peaks).")

print("\n=== (B) failure anatomy among N5 misses (linear) ===")
nmiss=defaultdict(int); rmiss=defaultdict(int); tot=0
for g in gis5:
    top5=set(int(x) for x in np.argsort(vlin[g])[::-1][:5]); miss=true_set[g]-top5
    if miss:
        tot+=1; nmiss[len(miss)]+=1
        for d in miss: rmiss[rnk_of[(g,d)]]+=1
print(f"   N5 samples that miss >=1 donor: {tot}/{len(gis5)}")
print("   #donors missed per failed sample: "+", ".join(f"{k}->{nmiss[k]}" for k in sorted(nmiss)))
print("   rank of the missed donor:        "+", ".join(f"r{k}->{rmiss[k]}" for k in sorted(rmiss)))

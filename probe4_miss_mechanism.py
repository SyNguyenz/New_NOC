"""
What REALLY causes the encoder's ~.21 N5 miss? (height-dominance refuted by probe3.) Test the two
leading mechanisms head-to-head on the FAINT donor of each N5 sample, using the ENCODER readout
(clean per-peak probe soft-vote = the .78 ceiling), recall = faint donor in top-5.

  (A) INTRINSIC weakness : few PRIVATE alleles present / mostly shared -> weak evidence -> miss.
      => recall should drop with #private alleles; and a donor should be CONSISTENTLY missed across combos.
  (B) COMBINATORIAL/carrier : the combo's shared-context carrier absorbs the donor's H (F32 self-corr .70).
      => recall should be COMBO-dependent (same donor recalled in some combos, missed in others);
         and miss should rise with the faint donor's pooled-H cosine to the sample carrier (mean-H).

Decisive split: WITHIN-donor recall variance. low (donors are 0/1) => intrinsic(A). high/mid => combo(B).
Plus: recall vs #private-alleles (A), recall vs carrier-cosine (B), recall vs height-ratio (control).
"""
import json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w"); RUN="results/inc2_2d_sparse_seed42"; DEV="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0); geno=load_raw_genotypes()
cfg=json.load(open(Path(RUN)/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
m=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","per_donor"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
    attn_sink=int(cfg.get("attn_sink",0) or 0),donor_recon=cfg.get("donor_recon",False)).to(DEV)
m.load_state_dict(torch.load(Path(RUN)/"best_model.pt",map_location=DEV,weights_only=True),strict=False); m.eval()

def load(s): return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
                     np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")
@torch.no_grad()
def enc(tk,mk,idx,bs=128):
    out={}
    for s in range(0,len(idx),bs):
        sel=idx[s:s+bs]; _,H,_=m._encode_set(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        H=H.cpu().numpy()
        for j,gi in enumerate(sel): out[int(gi)]=H[j]
    return out

# clean probe on train H
tk_tr,mk_tr,at_tr=load("train"); HH=[];DD=[]
for gi,H in enc(tk_tr,mk_tr,np.random.default_rng(0).choice(len(at_tr),6000,replace=False)).items():
    a=at_tr[gi]; v=np.where(a>=0)[0]; HH.append(H[v]); DD.append(a[v])
Xt=torch.from_numpy(np.concatenate(HH).astype(np.float32)).to(DEV); yt=torch.from_numpy(np.concatenate(DD).astype(int)).long().to(DEV)
clf=nn.Linear(Xt.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
for ep in range(60):
    perm=torch.randperm(len(yt),device=DEV)
    for s in range(0,len(yt),8192):
        b=perm[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
W=clf.weight.detach().cpu().numpy(); B=clf.bias.detach().cpu().numpy()
def smax(X): Z=X@W.T+B; Z-=Z.max(1,keepdims=True); E=np.exp(Z); return E/E.sum(1,keepdims=True)
print("probe fit\n")

tk,mk,at=load("test"); y=np.load(DATA/"y_test_set.npy"); noc=np.clip(np.load(DATA/"noc_test.npy").astype(int),1,5)
keep=[gi for gi in range(len(at)) if noc[gi]==5 and len(np.unique(at[gi][at[gi]>=0]))==5]
Hmap=enc(tk,mk,np.array(keep))
def private_of(gi,d,contribs):
    others=[KNOWN[o] for o in contribs if o!=d]; gX=geno.get(KNOWN[d],{}); pr=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: pr.add((L,a))
    return pr

rows=[]   # (donor, recalled, n_priv_present, priv_frac, hr, carrier_cos)
for gi in keep:
    a=at[gi]; v=np.where(a>=0)[0]
    lh=tk[gi][:,2]; hsum={int(d):float(np.exp(lh[a==d]).sum()) for d in np.unique(a[v])}
    contribs=list(hsum); tot=sum(hsum.values())
    faint=min(hsum,key=hsum.get)
    # encoder soft-vote recall (top-5)
    vote=smax(Hmap[gi][v]).sum(0); top=set(np.argsort(vote)[::-1][:5])
    rec=int(faint in top)
    # private alleles present
    pr=private_of(gi,faint,contribs); obs=set((int(tk[gi][j,0]),akey(tk[gi][j,1])) for j in v if int(a[j])==faint)
    npriv=len(pr & obs); ntot=len(obs)
    # carrier cosine: faint pooled-H vs sample mean-H
    Hs=Hmap[gi][v]; carrier=Hs.mean(0); fp=Hs[[i for i,j in enumerate(v) if int(a[j])==faint]].mean(0)
    cos=float(fp@carrier/((np.linalg.norm(fp)+1e-8)*(np.linalg.norm(carrier)+1e-8)))
    rows.append((faint,rec,npriv,npriv/max(ntot,1),hsum[faint]/tot,cos))

faints=np.array([r[0] for r in rows])
arr=np.array([(r[1],r[2],r[3],r[4],r[5]) for r in rows],dtype=float)
recall=arr[:,0]; npriv=arr[:,1]; pfrac=arr[:,2]; hr=arr[:,3]; cos=arr[:,4]

print(f"N5 faint donor-instances: {len(recall)}  overall encoder-readout recall: {recall.mean():.3f}\n")
def bins(x,rec,edges,lbl):
    print(f"  recall by {lbl}:")
    for lo,hi in zip(edges[:-1],edges[1:]):
        s=(x>=lo)&(x<hi)
        if s.sum(): print(f"    [{lo:g},{hi:g}): recall {rec[s].mean():.3f}  n={int(s.sum())}")
bins(npriv,recall,[0,1,2,3,5,100],"# PRIVATE alleles present (A intrinsic)")
bins(pfrac,recall,[0,0.25,0.5,0.75,1.01],"PRIVATE fraction (A)")
bins(hr,recall,[0,0.05,0.10,0.15,1.0],"height-ratio (control, refuted)")
bins(cos,recall,[0,0.6,0.7,0.8,0.9,1.01],"carrier cosine (B combinatorial)")

print("\n  WITHIN-DONOR recall consistency (A intrinsic => ~0/1; B combo => mid/variable):")
byd=defaultdict(list)
for d,r in zip(faints,recall): byd[d].append(r)
multi={d:v for d,v in byd.items() if len(v)>=4}
rates=np.array([np.mean(v) for v in multi.values()])
print(f"    donors with >=4 faint-appearances: {len(multi)}")
print(f"    their recall-rate distribution: mean {rates.mean():.2f}  frac in [0.2,0.8] (variable) {np.mean((rates>=0.2)&(rates<=0.8)):.2f}")
print(f"    frac ~consistent (<0.2 or >0.8): {np.mean((rates<0.2)|(rates>0.8)):.2f}")
# Bernoulli null: if recall were i.i.d. per-instance (pure combo/noise, no donor effect), expected frac-variable
p=recall.mean(); import math
print(f"    [if recall were donor-INDEPENDENT at p={p:.2f}, most ~4-sample donors would look 'variable' too]")

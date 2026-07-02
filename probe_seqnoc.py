"""
User's NOC idea (sequential top-down subtraction): order candidates by cls+phi score; add donors one
at a time; NOC = the k at which the mix is FULLY EXPLAINED (residual feasible-peak height ~0). Robust
to a decoy in top-k: if a faint true donor is displaced, its PRIVATE alleles stay UNEXPLAINED -> residual
remains -> "need one more donor". Compare to the trained card-head. tau tuned on val.
"""
import os, json
from pathlib import Path
import numpy as np, torch
from models.set_transformer import SetTransformerMixture
DA=Path("data_insilico_w"); RUN=Path(os.environ.get("RUN","results/inc13_B_distill_seed42")); G="data/donor_geno.npy"
DEVc=torch.device("cuda" if torch.cuda.is_available() else "cpu"); NITER=5; ALPHA=float(os.environ.get("ALPHA","0.75"))
def ab(a): return int(round(float(a)*10))
def kk(l,a): return (int(round(float(l))),ab(a))
g=np.load(G); gm=np.load(G.replace(".npy","_mask.npy")).astype(bool); C=g.shape[0]
dset=[set() for _ in range(C)]; carr={}
for c in range(C):
    for j in range(g.shape[1]):
        if gm[c,j]: it=kk(g[c,j,0],g[c,j,1]); dset[c].add(it); carr.setdefault(it,[]).append(c)
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats",8)
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
    dropout=0.1,cls_decoder="per_donor",decoder_source="encoded",n_token_feats=n_tok,encoder="isab++",dec_layers=2,
    num_embed="periodic",n_freq=8,d_num_emb=8,periodic_sigma=0.3,aux_heads=True,sparse_attn=True).to(DEVc)
sd=torch.load(RUN/"best_model.pt",map_location=DEVc); sd=sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd
m.load_state_dict(sd,strict=False); m.eval()
@torch.no_grad()
def fwd(t,k):
    L=[]; Cp=[]
    for i in range(0,len(t),128):
        r=m(torch.from_numpy(t[i:i+128]).to(DEVc),torch.from_numpy(k[i:i+128].astype(bool)).to(DEVc))
        L.append(r["logits_cls"].cpu().numpy()); Cp.append(r["logits_card"].argmax(1).cpu().numpy()+1)
    return np.concatenate(L),np.concatenate(Cp)
def feats(tk,mk):
    N=len(tk); PH=np.zeros((N,C)); peaklist=[]
    for i in range(N):
        pk=[(kk(tk[i,k,0],tk[i,k,1]),np.expm1(tk[i,k,2])) for k in np.where(mk[i])[0] if kk(tk[i,k,0],tk[i,k,1]) in carr]
        peaklist.append(pk)
        if not pk: continue
        n=len(pk); h=np.array([p[1] for p in pk]); S=np.full((n,C+1),-1e9)
        for r,(it,_) in enumerate(pk):
            for c in carr[it]: S[r,c]=0.0
            S[r,C]=-2.0
        phi=np.ones(C+1)/(C+1)
        for _ in range(NITER):
            z=S+np.log(phi+1e-9); z-=z.max(1,keepdims=True); A=np.exp(z); A/=A.sum(1,keepdims=True)
            w=(A[:,:C]*h[:,None]).sum(0); bg=(A[:,C]*h).sum(); phi=np.concatenate([w,[bg]])/max(w.sum()+bg,1e-9)
        PH[i]=phi[:C]
    return PH,peaklist
def z(a): s=a.std(); return (a-a.mean())/(s if s>1e-9 else 1.0)
def resid_curve(L,PH,peaklist):       # residual(k)/total for k=1..5 per sample, using cls+phi ranking
    N=len(L); R=np.ones((N,6))
    for i in range(N):
        pk=peaklist[i]
        if not pk: continue
        MODE=os.environ.get("RESID","height"); TH=float(os.environ.get("TH","50")); KMAX=int(os.environ.get("KMAX","6"))
        # decisive peaks = above stutter height AND carried by few panel donors (private-ish, the F38 signal)
        if MODE=="decisive": pkd=[(it,h) for it,h in pk if h>=TH and len(carr[it])<=KMAX]
        else: pkd=pk
        tot=(sum(h for _,h in pkd) if MODE=="height" else max(len(pkd),1))
        sc=z(L[i])+ALPHA*z(np.log(PH[i]+1e-6)); order=np.argsort(sc)[::-1]
        cov=set()
        for k in range(1,6):
            cov|=dset[order[k-1]]
            unexp=[(it,h) for it,h in pkd if it not in cov]
            R[i,k]=(sum(h for _,h in unexp) if MODE=="height" else len(unexp))/max(tot,1e-9)
    return R
def load(sp): return (np.load(DA/f"tokens{n_tok}_{sp}.npy").astype(np.float32),np.load(DA/f"mask_{sp}.npy"),np.clip(np.load(DA/f"noc_{sp}.npy"),1,5))
tkv,mkv,nv=load("val"); tkt,mkt,nt=load("test")
Lv,Cv=fwd(tkv,mkv); Lt,Ct=fwd(tkt,mkt); PHv,plv=feats(tkv,mkv); PHt,plt=feats(tkt,mkt)
Rv=resid_curve(Lv,PHv,plv); Rt=resid_curve(Lt,PHt,plt)
def seqcount(R,tau): return np.clip(np.array([next((k for k in range(1,6) if R[i,k]<tau),5) for i in range(len(R))]),1,5)
best_t,best=0.05,-1
for t in np.linspace(0.01,0.30,30):
    a=(seqcount(Rv,t)==nv).mean()
    if a>best: best,best_t=a,t
def acc(p,tr,k): s=tr==k; return (p[s]==tr[s]).mean() if s.any() else float("nan")
sq=seqcount(Rt,best_t)
print(f"=== sequential-subtraction NOC ({RUN.name}, tau={best_t:.3f} on val) vs card-head ===")
print(f"  {'NOC':>4} {'card-head':>10} {'seq-resid':>10}")
for k in [1,2,3,4,5]:
    print(f"  {k:>4} {acc(Ct,nt,k):>10.3f} {acc(sq,nt,k):>10.3f}")
print(f"  {'ALL':>4} {(Ct==nt).mean():>10.3f} {(sq==nt).mean():>10.3f}")
print(f"\n  mean residual curve (test, by true NOC) r1..r5:")
for k in [1,3,5]:
    s=nt==k; print(f"   trueNOC={k}: "+"  ".join(f"k{j}={Rt[s,j].mean():.3f}" for j in range(1,6)))

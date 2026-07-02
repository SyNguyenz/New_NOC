"""
User's two bets, small-scale TRAINED:
  (1) trained phi_head (attr = fixed genotype LOOKUP): does the phi_head's phi beat the no-train EM-uniform?
  (2) soft genotype-CONSTRAINED attr (masked softmax, trained): used as soft-split compatibility, does it
      beat uniform (production de-risk: uniform decoy-AUC 0.686 >> neural-hard-attr 0.225)?
Train ONE small model (cls + genotype-masked soft attr + phi), then compare THREE phi sources on test N5:
  A = phi_head (trained)   B = EM uniform (no head)   C = EM with trained-soft-attr as compatibility.
Metrics: decoy-AUC (missed-true vs decoy), within-N5 corr, rerank N5 oracle (alpha tuned on val).
"""
import os, math, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DA="data_insilico_w"; G="data/donor_geno.npy"; DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_N=int(os.environ.get("TRAIN_N","12000")); EPOCHS=int(os.environ.get("EPOCHS","30")); SEED=int(os.environ.get("SEED","42")); C=45; NITER=5
def ab(a): return int(round(float(a)*10))
def kk(l,a): return (int(round(float(l))),ab(a))
g=np.load(G); gm=np.load(G.replace(".npy","_mask.npy")).astype(bool)
dset=[set() for _ in range(C)]; carr={}
OWNER=np.zeros((24,1024,C),np.float32)
for c in range(C):
    for j in range(g.shape[1]):
        if gm[c,j]:
            it=kk(g[c,j,0],g[c,j,1]); dset[c].add(it); carr.setdefault(it,[]).append(c)
            li=int(round(g[c,j,0])); bi=ab(g[c,j,1])
            if 0<=li<24 and 0<=bi<1024: OWNER[li,bi,c]=1.0
OWNER_t=torch.from_numpy(OWNER).to(DEV)
def load(sp): return (np.load(f"{DA}/tokens8_{sp}.npy").astype(np.float32),np.load(f"{DA}/mask_{sp}.npy").astype(bool),
    np.load(f"{DA}/y_{sp}_set.npy").astype(np.float32),np.clip(np.load(f"{DA}/noc_{sp}.npy"),1,5),
    np.load(f"{DA}/phi_{sp}.npy").astype(np.float32),np.load(f"{DA}/attr_{sp}.npy").astype(np.int64))
tk,mk,y,noc,phi,attr=load("train")
def devmask(y,noc,frac=0.15,n1=0.06,seed=0):
    rng=np.random.default_rng(seed); m=np.zeros(len(noc),bool)
    for k in [2,3,4,5]:
        idx=np.where(noc==k)[0]; cm={}
        for i in idx: cm.setdefault(tuple(np.where(y[i]==1)[0]),[]).append(i)
        u=list(cm); rng.shuffle(u)
        for cc in u[:max(1,int(round(len(u)*frac)))]: m[cm[cc]]=True
    i1=np.where(noc==1)[0]; m[rng.choice(i1,size=int(round(len(i1)*n1)),replace=False)]=True
    return m
dm=devmask(y,noc); tri=np.where(~dm)[0]; rng=np.random.default_rng(0); tri=rng.choice(tri,size=min(TRAIN_N,len(tri)),replace=False)
def sub(ix): return tuple(a[ix] for a in (tk,mk,y,noc,phi,attr))
TRN=sub(tri); VAL=load("val"); TST=load("test")
_v=TRN[0][TRN[1]][:,2:8]; FMEAN=torch.tensor(_v.mean(0)).to(DEV); FSTD=torch.tensor(_v.std(0).clip(1e-3)).to(DEV)
print(f"train={len(TRN[0])} testN5={int((TST[3]==5).sum())} valNOC={np.bincount(VAL[3])[1:]}")

class MAB(nn.Module):
    def __init__(s,d,h): super().__init__(); s.att=nn.MultiheadAttention(d,h,batch_first=True); s.l1=nn.LayerNorm(d); s.l2=nn.LayerNorm(d); s.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
    def forward(s,q,k,kpm=None): a,_=s.att(q,k,k,key_padding_mask=kpm,need_weights=False); x=s.l1(q+a); return s.l2(x+s.ff(x))
class Per(nn.Module):
    def __init__(s,d,nf=8,sg=0.3): super().__init__(); s.c=nn.Parameter(torch.randn(nf)*sg); s.l=nn.Linear(2*nf,d)
    def forward(s,v): z=2*math.pi*s.c*v.unsqueeze(-1); return s.l(torch.cat([torch.sin(z),torch.cos(z)],-1))
class M(nn.Module):
    def __init__(s,d=64,h=4,nind=16):
        super().__init__(); s.le=nn.Embedding(26,d); s.pe=Per(d); s.cl=nn.Linear(6,d)
        s.I=nn.Parameter(torch.randn(nind,d)*0.5); s.mh=MAB(d,h); s.mx=MAB(d,h)
        s.Q=nn.Parameter(torch.empty(C,d)); nn.init.xavier_uniform_(s.Q); s.dec=MAB(d,h); s.dec2=MAB(d,h)
        s.sw=nn.Parameter(torch.empty(C,d)); nn.init.xavier_uniform_(s.sw); s.sb=nn.Parameter(torch.full((C,),-2.0))
        s.card=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,5)); s.attr=nn.Linear(d,C+1); s.phi=nn.Linear(d,C)
    def forward(s,x,m):
        kpm=~m; e=s.le(x[...,0].long().clamp(0,25))+s.pe(x[...,1])+s.cl((x[...,2:8]-FMEAN)/FSTD)
        enc=s.mx(e,s.mh(s.I.unsqueeze(0).expand(x.size(0),-1,-1),e,kpm))
        li=x[...,0].long().clamp(0,23); bi=(x[...,1]*10).round().long().clamp(0,1023)
        cmask=OWNER_t[li,bi]                                   # (B,N,C) genotype mask
        al=s.attr(enc)                                        # (B,N,C+1)
        al_m=al.clone(); al_m[...,:C]=al_m[...,:C].masked_fill(cmask==0,-1e9)  # SOFT genotype-constrained
        pooled=(enc*m.float().unsqueeze(-1)).sum(1)/m.float().sum(1,keepdim=True).clamp(min=1)
        Qb=s.Q.unsqueeze(0).expand(x.size(0),-1,-1); dec=s.dec2(s.dec(Qb,enc,kpm),enc,kpm)
        return torch.einsum("bcd,cd->bc",dec,s.sw)+s.sb, s.card(pooled), al_m, F.softplus(s.phi(pooled))
torch.manual_seed(SEED); np.random.seed(SEED); model=M().to(DEV)
opt=torch.optim.Adam(model.parameters(),lr=6e-4,weight_decay=1e-4); posw=torch.full((C,),8.0,device=DEV)
gen=torch.Generator().manual_seed(SEED)
for ep in range(EPOCHS):
    model.train(); idx=torch.randperm(len(TRN[0]),generator=gen).numpy()
    for b in range(0,len(idx),128):
        bi=idx[b:b+128]; xb=torch.from_numpy(TRN[0][bi]).to(DEV); mb=torch.from_numpy(TRN[1][bi]).to(DEV)
        yb=torch.from_numpy(TRN[2][bi]).to(DEV); nb=torch.from_numpy((TRN[3][bi]-1).astype(np.int64)).to(DEV)
        pb=torch.from_numpy(TRN[4][bi]).to(DEV); ab_=torch.from_numpy(TRN[5][bi]).to(DEV)
        lo,cd,alm,ph=model(xb,mb)
        lbl=ab_.clone(); lbl[lbl<0]=C; lbl[~mb]=-100
        loss=(F.binary_cross_entropy_with_logits(lo,yb,pos_weight=posw)+0.3*F.cross_entropy(cd,nb)
              +1.0*F.mse_loss(ph,pb)+0.5*F.cross_entropy(alm.reshape(-1,C+1),lbl.reshape(-1),ignore_index=-100))
        opt.zero_grad(); loss.backward(); opt.step()
print("trained.")

@torch.no_grad()
def infer(S):
    x,m=S[0],S[1]; L=[];PH=[];AL=[]
    for i in range(0,len(x),128):
        lo,cd,alm,ph=model(torch.from_numpy(x[i:i+128]).to(DEV),torch.from_numpy(m[i:i+128]).to(DEV))
        L.append(lo.cpu().numpy()); PH.append(ph.cpu().numpy()); AL.append(alm.cpu().numpy())
    return np.concatenate(L),np.concatenate(PH),np.concatenate(AL)
def em_phi(tk,mk,AL=None):   # AL=None -> uniform compat ; else use attr logits as compat
    N=len(tk); P=np.zeros((N,C))
    for i in range(N):
        pk=[(k,kk(tk[i,k,0],tk[i,k,1]),np.expm1(tk[i,k,2])) for k in np.where(mk[i])[0] if kk(tk[i,k,0],tk[i,k,1]) in carr]
        if not pk: continue
        n=len(pk); h=np.array([p[2] for p in pk]); S=np.full((n,C+1),-1e9)
        for r,(k,it,_) in enumerate(pk):
            for c in carr[it]: S[r,c]= 0.0 if AL is None else AL[i,k,c]
            S[r,C]= -2.0 if AL is None else AL[i,k,C]
        ph=np.ones(C+1)/(C+1)
        for _ in range(NITER):
            z=S+np.log(ph+1e-9); z-=z.max(1,keepdims=True); A=np.exp(z); A/=A.sum(1,keepdims=True)
            w=(A[:,:C]*h[:,None]).sum(0); bg=(A[:,C]*h).sum(); ph=np.concatenate([w,[bg]])/max(w.sum()+bg,1e-9)
        P[i]=ph[:C]
    return P
def auc(p,q):
    p,q=np.asarray(p,float),np.asarray(q,float)
    if not len(p) or not len(q): return float("nan")
    a=np.concatenate([p,q]); _,inv,cnt=np.unique(a,return_inverse=True,return_counts=True); cs=np.cumsum(cnt)
    rk=((cs-cnt+cs+1)/2.0)[inv]; return (rk[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(q))
def z(a): s=a.std(); return (a-a.mean())/(s if s>1e-9 else 1.0)

Lt,PHt,ALt=infer(TST); Lv,PHv,ALv=infer(VAL)
yt=TST[2].astype(bool); nt=TST[3]; phit=TST[4]; yv=VAL[2].astype(bool); nv=VAL[3]
srcs={"A:phi_head":(PHt/np.maximum(PHt.sum(1,keepdims=True),1e-9), PHv/np.maximum(PHv.sum(1,keepdims=True),1e-9)),
      "B:EM-uniform":(em_phi(TST[0],TST[1]), em_phi(VAL[0],VAL[1])),
      "C:EM-softattr":(em_phi(TST[0],TST[1],ALt), em_phi(VAL[0],VAL[1],ALv))}
def evalsrc(name,PHtest,PHval):
    mt=[];dc=[];wc=[]
    for i in np.where(nt==5)[0]:
        miss=[c for c in np.where(yt[i])[0] if c not in set(np.argsort(Lt[i])[::-1][:5])]
        dec=[c for c in np.argsort(Lt[i])[::-1][:5] if not yt[i,c]]
        for c in miss: mt.append(PHtest[i,c])
        for c in dec: dc.append(PHtest[i,c])
        cs=np.where(yt[i])[0]
        if len(cs)>=2:
            a=PHtest[i,cs]-PHtest[i,cs].mean(); b=phit[i,cs]-phit[i,cs].mean(); d=a.std()*b.std()
            wc.append((a*b).mean()/d if d>1e-9 else 0)
    def orc(L,PH,y,noc,al):
        sel=np.where(noc==5)[0]; hit=0
        for i in sel:
            sc=z(L[i])+al*z(np.log(PH[i]+1e-6)); top=np.argsort(sc)[::-1][:5]; pr=np.zeros(C,int); pr[top]=1; hit+=int((pr==y[i]).all())
        return hit/max(1,len(sel))
    alphas=[0,0.25,0.5,0.75,1.0]; ba,bv=0,-1
    for al in alphas:
        v=orc(Lv,PHval,yv,nv,al)
        if v>bv: bv,ba=v,al
    return auc(mt,dc),np.nanmean(wc),orc(Lt,PHtest,yt,nt,0),orc(Lt,PHtest,yt,nt,ba),ba
print(f"\n{'source':>14} {'decoyAUC':>9} {'withinφ':>8} {'N5 model':>9} {'N5 rerank':>10} {'α*':>4}")
for nm,(pt,pv) in srcs.items():
    da,wcv,m0,mr,a=evalsrc(nm,pt,pv)
    print(f"{nm:>14} {da:>9.3f} {wcv:>8.3f} {m0:>9.3f} {mr:>10.3f} {a:>4}")
print("\n  bar to beat = B:EM-uniform (production: decoyAUC .686, +0.07 rerank). small-scale = directional.")

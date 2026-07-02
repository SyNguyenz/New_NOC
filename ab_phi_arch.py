"""
Why is phi_head broken (corr -0.23)? Negative transfer (cls crowds it out) or missing structure?
Test 3 architectures at small scale, measure phi-corr recovery + N5:
  para   : current — attr=Linear(enc), phi=Linear(pool enc), cls=decoder(enc).  shared encoder.
  sepphi : phi gets its OWN encoder branch (decoupled). tests negative-transfer.
  casc   : attr(enc) -> phi=Linear([pool enc, attr-derived height]) -> cls conditioned on phi.
           tests "derive phi from the GOOD attr head" (Sogaard-Goldberg cascade).
Judge: phi corr (present donors) vs -0.23, attr acc, N5 dev/test oracle.
"""
import os, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DA="data_insilico_w"; DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_N=int(os.environ.get("TRAIN_N","12000")); EPOCHS=int(os.environ.get("EPOCHS","30"))
SEEDS=[int(s) for s in os.environ.get("SEEDS","42").split(",")]; ARMS=os.environ.get("ARMS","para,sepphi,casc").split(",")
C=45
def load(sp): return (np.load(f"{DA}/tokens8_{sp}.npy").astype(np.float32), np.load(f"{DA}/mask_{sp}.npy").astype(bool),
    np.load(f"{DA}/y_{sp}_set.npy").astype(np.float32), np.load(f"{DA}/noc_{sp}.npy").astype(int),
    np.load(f"{DA}/phi_{sp}.npy").astype(np.float32), np.load(f"{DA}/attr_{sp}.npy").astype(np.int64))
tk,mk,y,noc,phi,attr=load("train")
def devmask(y,noc,frac=0.15,n1=0.06,seed=0):
    rng=np.random.default_rng(seed); noc=np.clip(noc,1,5); m=np.zeros(len(noc),bool)
    for k in [2,3,4,5]:
        idx=np.where(noc==k)[0]; cm={}
        for i in idx: cm.setdefault(tuple(np.where(y[i]==1)[0]),[]).append(i)
        u=list(cm); rng.shuffle(u)
        for cc in u[:max(1,int(round(len(u)*frac)))]: m[cm[cc]]=True
    i1=np.where(noc==1)[0]; m[rng.choice(i1,size=int(round(len(i1)*n1)),replace=False)]=True
    return m
dm=devmask(y,noc); tri=np.where(~dm)[0]
rng=np.random.default_rng(0); tri=rng.choice(tri,size=min(TRAIN_N,len(tri)),replace=False)
def sub(ix): return (tk[ix],mk[ix],y[ix],noc[ix],phi[ix],attr[ix])
DEVs=sub(np.where(dm)[0]); TRN=sub(tri); TST=load("test")
print(f"train={len(TRN[0])} dev={len(DEVs[0])} (N5={int((DEVs[3]==5).sum())}) testN5={int((TST[3]==5).sum())}")
_v=TRN[0][TRN[1]][:,2:8]; FMEAN=torch.tensor(_v.mean(0)).to(DEV); FSTD=torch.tensor(_v.std(0).clip(1e-3)).to(DEV)

class MAB(nn.Module):
    def __init__(s,d,h): super().__init__(); s.att=nn.MultiheadAttention(d,h,batch_first=True); s.l1=nn.LayerNorm(d); s.l2=nn.LayerNorm(d); s.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
    def forward(s,q,k,kpm=None): a,_=s.att(q,k,k,key_padding_mask=kpm,need_weights=False); x=s.l1(q+a); return s.l2(x+s.ff(x))
class Per(nn.Module):
    def __init__(s,d,nf=8,sig=0.3): super().__init__(); s.c=nn.Parameter(torch.randn(nf)*sig); s.l=nn.Linear(2*nf,d)
    def forward(s,v): z=2*math.pi*s.c*v.unsqueeze(-1); return s.l(torch.cat([torch.sin(z),torch.cos(z)],-1))
class Enc(nn.Module):
    def __init__(s,d,h,nind): super().__init__(); s.I=nn.Parameter(torch.randn(nind,d)*0.5); s.mh=MAB(d,h); s.mx=MAB(d,h)
    def forward(s,e,kpm): H=s.mh(s.I.unsqueeze(0).expand(e.size(0),-1,-1),e,kpm); return s.mx(e,H)
class M(nn.Module):
    def __init__(s,arch,d=64,h=4,nind=16):
        super().__init__(); s.arch=arch
        s.lemb=nn.Embedding(26,d); s.per=Per(d); s.clin=nn.Linear(6,d)
        s.enc=Enc(d,h,nind)
        s.Q=nn.Parameter(torch.empty(C,d)); nn.init.xavier_uniform_(s.Q)
        s.dec=MAB(d,h); s.dec2=MAB(d,h)
        s.sw=nn.Parameter(torch.empty(C,d)); nn.init.xavier_uniform_(s.sw); s.sb=nn.Parameter(torch.full((C,),-2.0))
        s.card=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,5))
        s.attr=nn.Linear(d,C+1)
        if arch=="casc": s.phi=nn.Linear(d+C,C); s.pvec=nn.Parameter(torch.zeros(d))
        else: s.phi=nn.Linear(d,C)
        if arch=="sepphi": s.enc2=Enc(d,h,nind)
    def emb(s,x): return s.lemb(x[...,0].long().clamp(0,25))+s.per(x[...,1])+s.clin((x[...,2:8]-FMEAN)/FSTD)
    def forward(s,x,m):
        kpm=~m; e=s.emb(x); enc=s.enc(e,kpm)
        attr=s.attr(enc)                                   # (B,N,C+1)
        if s.arch=="sepphi":
            pooled2=(s.enc2(e,kpm)*m.float().unsqueeze(-1)).sum(1)/m.float().sum(1,keepdim=True).clamp(min=1)
            phi=F.softplus(s.phi(pooled2))
        pooled=(enc*m.float().unsqueeze(-1)).sum(1)/m.float().sum(1,keepdim=True).clamp(min=1)
        if s.arch=="casc":
            aw=F.softmax(attr,-1)[...,:C]                   # (B,N,C) soft attribution
            adh=(aw*x[...,2].unsqueeze(-1)*m.float().unsqueeze(-1)).sum(1)  # (B,C) attr-weighted height
            phi=F.softplus(s.phi(torch.cat([pooled,adh],-1)))
        elif s.arch=="para": phi=F.softplus(s.phi(pooled))
        Qb=s.Q.unsqueeze(0).expand(x.size(0),-1,-1)
        if s.arch=="casc": Qb=Qb + phi.unsqueeze(-1)*s.pvec  # condition donor queries on phi
        dec=s.dec(Qb,enc,kpm); dec=s.dec2(dec,enc,kpm)
        logit=torch.einsum("bcd,cd->bc",dec,s.sw)+s.sb
        return logit, s.card(pooled), attr, phi

def batches(arrs,bs,g):
    idx=torch.randperm(len(arrs[0]),generator=g).numpy()
    for i in range(0,len(idx),bs): yield [a[idx[i:i+bs]] for a in arrs]
def corr(a,b):
    a,b=a-a.mean(),b-b.mean(); d=(a.std()*b.std()); return float((a*b).mean()/d) if d>1e-9 else 0.0

def run(arch,seed):
    torch.manual_seed(seed); np.random.seed(seed); model=M(arch).to(DEV)
    opt=torch.optim.Adam(model.parameters(),lr=6e-4,weight_decay=1e-4); posw=torch.full((C,),8.0,device=DEV)
    g=torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        model.train()
        for xb,mb,yb,nb,pb,ab in batches([TRN[0],TRN[1],TRN[2],TRN[3],TRN[4],TRN[5]],128,g):
            xb=torch.from_numpy(xb).to(DEV); mb=torch.from_numpy(mb).to(DEV); yb=torch.from_numpy(yb).to(DEV)
            nb=torch.from_numpy(np.clip(nb,1,5)-1).to(DEV); pb=torch.from_numpy(pb).to(DEV); ab=torch.from_numpy(ab).to(DEV)
            logit,card,attr,phi=model(xb,mb)
            al=ab.clone(); al[al<0]=C; al[~mb]=-100                    # attr label: -1->background C, pad->ignore
            loss=(F.binary_cross_entropy_with_logits(logit,yb,pos_weight=posw)+0.3*F.cross_entropy(card,nb)
                  +1.0*F.mse_loss(phi,pb)+0.5*F.cross_entropy(attr.reshape(-1,C+1),al.reshape(-1),ignore_index=-100))
            opt.zero_grad(); loss.backward(); opt.step()
    @torch.no_grad()
    def ev(S):
        model.eval(); L=[]; PH=[]; AC=[]
        x,m,yy,nn_,pp,aa=S
        for i in range(0,len(x),256):
            lo,_,at,ph=model(torch.from_numpy(x[i:i+256]).to(DEV),torch.from_numpy(m[i:i+256]).to(DEV))
            L.append(lo.cpu().numpy()); PH.append(ph.cpu().numpy())
            ap=at.argmax(-1).cpu().numpy(); mm=m[i:i+256]; al=aa[i:i+256]
            AC.append(((ap==al)&(al>=0)&mm).sum()/max(1,((al>=0)&mm).sum()))
        L=np.concatenate(L); PH=np.concatenate(PH); nv=np.clip(nn_,1,5)
        orc={}
        for k in range(1,6):
            sel=nv==k
            if sel.any(): orc[k]=float(np.mean([ (lambda t: (np.isin(np.arange(C),t)==yy[j].astype(bool)).all())(np.argsort(L[j])[::-1][:k]) for j in np.where(sel)[0]]))
        pres=yy.astype(bool)
        return orc, corr(PH[pres],pp[pres]), float(np.mean(AC))
    od,pc,ac=ev(DEVs); ot,pct,act=ev(TST)
    return od,ot,pc,ac

print("="*72)
res={}
for arm in ARMS:
    for seed in SEEDS:
        t=time.time(); od,ot,pc,ac=run(arm,seed)
        print(f"[{arm} s{seed}] {time.time()-t:.0f}s devN5={od.get(5,0):.3f} testN5={ot.get(5,0):.3f} phiCorr={pc:+.3f} attrAcc={ac:.3f}")
    res[arm]=(od,ot,pc,ac)
print("\n"+"="*72+"\nSUMMARY (last seed)")
print(f"{'arch':>8} | {'DEV N1..N5':>30} | phiCorr  attrAcc | testN5")
for arm in ARMS:
    od,ot,pc,ac=res[arm]
    print(f"{arm:>8} | "+" ".join(f"{od.get(k,0):.3f}" for k in range(1,6))+f" | {pc:+.3f}   {ac:.3f} | {ot.get(5,0):.3f}")
print("\nJUDGE: phiCorr vs para's broken value. sepphi up => negative-transfer. casc up => structure. + N5.")

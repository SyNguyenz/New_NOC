"""
Controlled small-scale A/B: does a SET-OF-SETS representation beat flat allele-tokens when BOTH
are TRAINED at the same (small) budget? (fixes the unfair trained-vs-heuristic probe.)

Arms (identical small model/data/budget; only the toggled component differs):
  A   : baseline      flat allele tokens -> ISAB encoder -> per-donor cross-attn decoder
  B1  : +sig_head     learned soft-DNF per donor over mined minimal-JEP signatures (OR-of-ANDs),
                       added to the decoder logit (theta init 0 => reduces to A at init)
  B2  : +pair_tokens  within-locus genotype co-occurrence tokens appended to the encoder input
  B3  : B1+B2

Judge: combo-disjoint DEV N5 oracle (the wall) + guard N1-4, and real-test N5 oracle (secondary).
Directional (1 seed unless SEEDS set); a clear winner gets confirmed at full scale.
"""
import os, json, itertools, time, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DATA = Path(os.environ.get("STR_DATA_DIR", "data_insilico_w"))
GENO = Path(os.environ.get("STR_GENO", "data/donor_geno.npy"))
TRAIN_N = int(os.environ.get("TRAIN_N", "12000"))
EPOCHS  = int(os.environ.get("EPOCHS", "30"))
SEEDS   = [int(s) for s in os.environ.get("SEEDS", "42").split(",")]
ARMS    = os.environ.get("ARMS", "A,B1,B2,B3").split(",")
PAIR_TOPK = 5            # top-by-height peaks/locus to pair
MAX_ORDER, SIG_TOPK = 3, 60
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def abin(a): return int(round(float(a) * 10))
def ikey(l, a): return (int(round(float(l))), abin(a))

# ───────────────────────── mine minimal JEP signatures (panel-only) ─────────────────────────
g  = np.load(GENO); gm = np.load(str(GENO).replace(".npy", "_mask.npy")).astype(bool); C = g.shape[0]
donor_items = [set(ikey(g[c,j,0], g[c,j,1]) for j in range(g.shape[1]) if gm[c,j]) for c in range(C)]
owners = {}
for c in range(C):
    for it in donor_items[c]: owners.setdefault(it, set()).add(c)
ITEMS = sorted(owners); IIDX = {it: i for i, it in enumerate(ITEMS)}; NI = len(ITEMS)
def rar(it): return float(np.log(C / len(owners[it])))
def mine(c):
    A = sorted(donor_items[c]); priv = [it for it in A if owners[it]=={c}]; sigs=[(it,) for it in priv]
    nonp = [it for it in A if owners[it]!={c}]
    if MAX_ORDER>=2:
        cc=[(x,y,rar(x)+rar(y)) for x,y in itertools.combinations(nonp,2) if not ((owners[x]&owners[y])-{c})]
        cc.sort(key=lambda t:-t[2]); sigs+=[(x,y) for x,y,_ in cc[:SIG_TOPK]]
    if MAX_ORDER>=3:
        cc=[]
        for x,y,z in itertools.combinations(nonp,3):
            if not((owners[x]&owners[y])-{c}): continue
            if not((owners[x]&owners[z])-{c}): continue
            if not((owners[y]&owners[z])-{c}): continue
            if not((owners[x]&owners[y]&owners[z])-{c}): cc.append((x,y,z,rar(x)+rar(y)+rar(z)))
        cc.sort(key=lambda t:-t[3]); sigs+=[(x,y,z) for x,y,z,_ in cc[:SIG_TOPK]]
    return sigs
JEP = [mine(c) for c in range(C)]
MAXS = max(len(s) for s in JEP)
# padded sig tensors: index NI = dummy item (presence forced 1, neutral for product)
SIG_IDX = np.full((C, MAXS, 3), NI, np.int64); SIG_VALID = np.zeros((C, MAXS), np.float32)
for c in range(C):
    for s, items in enumerate(JEP[c]):
        SIG_VALID[c, s] = 1.0
        for t, it in enumerate(items): SIG_IDX[c, s, t] = IIDX[it]
SIG_IDX = torch.from_numpy(SIG_IDX).to(DEV); SIG_VALID = torch.from_numpy(SIG_VALID).to(DEV)
print(f"panel: {C} donors | items={NI} | sigs/donor max={MAXS} mean={np.mean([len(s) for s in JEP]):.0f}")
# OWNER (C,NI) dosage matrix for additive height reconstruction (decoupled recon arm R)
OWNER=np.zeros((C,NI),np.float32)
for c in range(C):
    for j in range(g.shape[1]):
        if gm[c,j]:
            ii=IIDX.get(ikey(g[c,j,0],g[c,j,1]))
            if ii is not None: OWNER[c,ii]+=1.0
OWNER_t=torch.from_numpy(OWNER).to(DEV)
RECON_W=float(os.environ.get("RECON_W","0.5")); RECON_RAMP=int(os.environ.get("RECON_RAMP","10"))

# ───────────────────────── data: load, carve combo-disjoint dev, subsample ─────────────────────────
def load(split):
    return (np.load(DATA/f"tokens8_{split}.npy").astype(np.float32), np.load(DATA/f"mask_{split}.npy").astype(bool),
            np.load(DATA/f"y_{split}_set.npy").astype(np.float32), np.load(DATA/f"noc_{split}.npy").astype(int))
tk, mk, y, noc = load("train")
def dev_mask_seed0(y, noc, frac=0.15, n1=0.06, seed=0):
    rng=np.random.default_rng(seed); noc=np.clip(noc,1,5); m=np.zeros(len(noc),bool)
    for k in [2,3,4,5]:
        idx=np.where(noc==k)[0]; combos={}
        for i in idx: combos.setdefault(tuple(np.where(y[i]==1)[0]),[]).append(i)
        u=list(combos); rng.shuffle(u)
        for cc in u[:max(1,int(round(len(u)*frac)))]: m[combos[cc]]=True
    i1=np.where(noc==1)[0]; m[rng.choice(i1,size=int(round(len(i1)*n1)),replace=False)]=True
    return m
dm = dev_mask_seed0(y, noc)
dev = (tk[dm], mk[dm], y[dm], noc[dm])
tri = np.where(~dm)[0]
rng = np.random.default_rng(0); tri = rng.choice(tri, size=min(TRAIN_N, len(tri)), replace=False)
trn = (tk[tri], mk[tri], y[tri], noc[tri])
tst = load("test")
print(f"train={len(trn[0])}  dev={len(dev[0])} (N5={int((dev[3]==5).sum())})  test N5={int((tst[3]==5).sum())}")

# ───────────────────────── precompute pair tokens + presence per split ─────────────────────────
def featurize(tok, msk):
    N = tok.shape[0]
    # single tokens: production 8-feat + is_pair=0  -> 9 dims
    single = np.concatenate([tok, np.zeros((N,160,1),np.float32)], -1)            # (N,160,9)
    smask  = msk.copy()
    # pair tokens: within-locus top-k by height, all pairs; 8 joint feats + is_pair=1
    Plist = []
    for i in range(N):
        v = np.where(msk[i])[0]
        rows = []
        if len(v):
            loci = tok[i,v,0]
            for L in np.unique(loci):
                idx = v[loci==L]
                if len(idx) < 2: continue
                h = tok[i,idx,2]
                idx = idx[np.argsort(h)[::-1][:PAIR_TOPK]]
                for a, b in itertools.combinations(idx, 2):
                    aa,ha = tok[i,a,1], tok[i,a,2]; bb,hb = tok[i,b,1], tok[i,b,2]
                    bal = min(ha,hb)/max(ha,hb,1e-6)
                    rows.append([L, aa, ha, bb, hb, bal, math.log1p(math.expm1(ha)+math.expm1(hb)), abs(aa-bb)])
        Plist.append(rows)
    Pmax = max((len(r) for r in Plist), default=0); Pmax = max(Pmax, 1)
    pair = np.zeros((N, Pmax, 9), np.float32); pmask = np.zeros((N, Pmax), bool)
    for i, rows in enumerate(Plist):
        for j, r in enumerate(rows):
            pair[i,j,:8] = r; pair[i,j,8] = 1.0; pmask[i,j] = True
    # presence: max log-height per item (for sig head); -100 if absent
    pres = np.full((N, NI+1), -100.0, np.float32); pres[:, NI] = 100.0   # dummy item present
    for i in range(N):
        v = np.where(msk[i])[0]
        for k in v:
            it = ikey(tok[i,k,0], tok[i,k,1])
            ii = IIDX.get(it)
            if ii is not None: pres[i,ii] = max(pres[i,ii], tok[i,k,2])
    return single, smask, pair, pmask, pres
print("featurizing..."); t0=time.time()
FZ = {n: featurize(*d[:2]) for n, d in [("train",trn),("dev",dev),("test",tst)]}
print(f"  done in {time.time()-t0:.0f}s | pair seqlen train={FZ['train'][2].shape[1]}")
# normalization stats for cols 2-7 over valid TRAIN single tokens
_s9,_sm = FZ["train"][0], FZ["train"][1]
_vt = _s9[_sm][:, 2:8]
FMEAN = torch.tensor(_vt.mean(0)); FSTD = torch.tensor(_vt.std(0).clip(1e-3))
# recon target normalization (log1p height over present panel bins) + phi for dissection
_pr=FZ["train"][4][:, :NI]; _present=_pr[_pr>-50]
RMEAN=torch.tensor(float(_present.mean())).to(DEV); RSTD=torch.tensor(float(_present.std()+1e-6)).to(DEV)
PHI_TEST=np.load(DATA/"phi_test.npy")

# ───────────────────────── minimal set-transformer ─────────────────────────
class MAB(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.att=nn.MultiheadAttention(d,h,batch_first=True)
        s.l1=nn.LayerNorm(d); s.l2=nn.LayerNorm(d); s.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
    def forward(s, q, k, kpm=None):
        a,_=s.att(q,k,k,key_padding_mask=kpm,need_weights=False); x=s.l1(q+a); return s.l2(x+s.ff(x))
class Periodic(nn.Module):                       # Gorishniy numeric embedding (the F1-F7 lever)
    def __init__(s, d, nf=8, sigma=0.3):
        super().__init__(); s.coef=nn.Parameter(torch.randn(nf)*sigma); s.lin=nn.Linear(2*nf, d)
    def forward(s, v):
        z = 2*math.pi*s.coef*v.unsqueeze(-1)
        return s.lin(torch.cat([torch.sin(z), torch.cos(z)], -1))
class SmallMix(nn.Module):
    def __init__(s, d=64, h=4, nind=16, sig=False, fmean=None, fstd=None, recon=False):
        super().__init__()
        s.locus_emb=nn.Embedding(26, d); s.allele_per=Periodic(d); s.cont_lin=nn.Linear(6, d); s.tau=nn.Embedding(2,d)
        s.register_buffer("fmean", fmean if fmean is not None else torch.zeros(6))
        s.register_buffer("fstd",  fstd  if fstd  is not None else torch.ones(6))
        s.I=nn.Parameter(torch.randn(nind,d)*0.5); s.mh=MAB(d,h); s.mx=MAB(d,h)
        s.Q=nn.Parameter(torch.empty(C,d)); nn.init.xavier_uniform_(s.Q)
        s.dec=MAB(d,h); s.dec2=MAB(d,h)                       # Query2Label L=2 cross-attn layers
        s.score_w=nn.Parameter(torch.empty(C,d)); nn.init.xavier_uniform_(s.score_w)  # PER-DONOR readout
        s.score_b=nn.Parameter(torch.full((C,), -2.5))   # RetinaNet bias init: stop ASL collapsing to all-negative
        s.card=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,5))
        s.sig=sig
        if sig:
            s.alpha=nn.Parameter(torch.tensor(3.0)); s.beta=nn.Parameter(torch.tensor(3.5))
            s.theta=nn.Parameter(torch.zeros(C, MAXS))     # init 0 => neutral at init
        s.recon=recon
        if recon:                                          # DECOUPLED recon: own weights, NOT sigmoid(cls_logit)
            s.wrec_head=nn.Linear(d,1); s.logG=nn.Parameter(torch.zeros(1))
    def forward(s, x, m, pres=None):
        kpm = ~m
        cont = (x[...,2:8] - s.fmean) / s.fstd                       # normalize logh + enriched (cols 2-7)
        e = (s.locus_emb(x[...,0].long().clamp(0,25)) + s.allele_per(x[...,1])
             + s.cont_lin(cont) + s.tau(x[...,8].long()))
        Ib = s.I.unsqueeze(0).expand(x.size(0),-1,-1)
        H = s.mh(Ib, e, kpm); enc = s.mx(e, H)                       # (B,N,d)
        Qb = s.Q.unsqueeze(0).expand(x.size(0),-1,-1)
        dec = s.dec(Qb, enc, kpm); dec = s.dec2(dec, enc, kpm)       # (B,C,d)
        logit = torch.einsum("bcd,cd->bc", dec, s.score_w) + s.score_b   # per-donor scoring
        denom = m.float().sum(1,keepdim=True).clamp(min=1)
        pooled = (enc*m.float().unsqueeze(-1)).sum(1)/denom
        card = s.card(pooled)
        if s.sig and pres is not None:
            p = torch.sigmoid(s.alpha*(pres - s.beta))               # (B,NI+1)
            p = torch.cat([p, p.new_ones(p.size(0),1)], 1) if p.size(1)==NI else p
            gp = p[:, SIG_IDX]                                       # (B,C,MAXS,3)
            sand = gp.prod(-1) * SIG_VALID.unsqueeze(0)              # (B,C,MAXS)
            slog = (sand * s.theta.unsqueeze(0)).sum(-1)            # (B,C)
            logit = logit + slog
        wrec = F.softplus(s.wrec_head(dec)).squeeze(-1) if s.recon else None   # (B,C) >=0, decoupled
        return logit, card, wrec

USE_ASL = os.environ.get("LOSS","bce")=="asl"
def asl_loss(x, y, gn=4.0, gp=0.0, clip=0.05, eps=1e-8):    # Ridnik 2021 (matches production)
    p=torch.sigmoid(x); pn=(1-p)
    if clip>0: pn=(pn+clip).clamp(max=1.0)
    lp=y*torch.log(p.clamp(min=eps)); ln=(1-y)*torch.log(pn.clamp(min=eps))
    pt=p*y + pn*(1-y); w=(1-pt)**(gp*y + gn*(1-y))
    return -((lp+ln)*w).mean()

def batches(arrs, bs, shuf, gen=None):
    N=arrs[0].shape[0]; idx=torch.randperm(N,generator=gen).numpy() if shuf else np.arange(N)
    for i in range(0,N,bs): yield [a[idx[i:i+bs]] for a in arrs]

def run_arm(arm, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    use_pair = arm in ("B2","B3"); use_sig = arm in ("B1","B3"); use_recon = arm in ("R",)
    model = SmallMix(sig=use_sig, fmean=FMEAN, fstd=FSTD, recon=use_recon).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=6e-4, weight_decay=1e-4)
    posw = torch.full((C,), 8.0, device=DEV)
    def pack(split):
        s9,sm,p9,pm,pres = FZ[split]
        if use_pair:
            X=np.concatenate([s9,p9],1); M=np.concatenate([sm,pm],1)
        else: X,M=s9,sm
        return X, M, pres
    Xtr,Mtr,Ptr = pack("train"); ytr,noctr = trn[2], trn[3]
    gen=torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        model.train()
        for xb,mb,pb,yb,nb in batches([Xtr,Mtr,Ptr,ytr,noctr],128,True,gen):
            xb=torch.from_numpy(xb).to(DEV); mb=torch.from_numpy(mb).to(DEV)
            pb=torch.from_numpy(pb).to(DEV); yb=torch.from_numpy(yb).to(DEV)
            nb=torch.from_numpy(np.clip(nb,1,5)-1).to(DEV)
            logit,card,wrec=model(xb,mb,pb if use_sig else None)
            clsl = asl_loss(logit,yb) if USE_ASL else F.binary_cross_entropy_with_logits(logit,yb,pos_weight=posw)
            loss=clsl+0.3*F.cross_entropy(card,nb)
            if use_recon:
                obs=pb[:, :NI]                                       # (B,NI) log1p height, -100 absent
                tl=torch.where(obs>-50, obs, torch.zeros_like(obs))  # target log1p height (0 if absent)
                hhat=torch.exp(model.logG)*(wrec @ OWNER_t)          # (B,NI) additive over donors (DECOUPLED)
                pl=torch.log1p(hhat.clamp(min=0))
                recon=(((pl-RMEAN)/RSTD - (tl-RMEAN)/RSTD)**2).mean()    # z-normalized log1p space
                loss=loss + RECON_W*min(1.0,(ep+1)/RECON_RAMP)*recon
            opt.zero_grad(); loss.backward(); opt.step()
    # eval
    @torch.no_grad()
    def oracle(split, arrs):
        X,M,P = pack(split); yv,nv = arrs[2], np.clip(arrs[3],1,5)
        model.eval(); L=[]
        for i in range(0,len(X),256):
            xb=torch.from_numpy(X[i:i+256]).to(DEV); mb=torch.from_numpy(M[i:i+256]).to(DEV)
            pb=torch.from_numpy(P[i:i+256]).to(DEV)
            lo,_,_=model(xb,mb,pb if use_sig else None); L.append(lo.cpu().numpy())
        L=np.concatenate(L); out={}
        for k in range(1,6):
            sel=nv==k
            if not sel.any(): continue
            em=[]
            for j in np.where(sel)[0]:
                top=np.argsort(L[j])[::-1][:k]; pr=np.zeros(C,int); pr[top]=1
                em.append((pr==yv[j]).all())
            out[k]=float(np.mean(em))
        return out
    @torch.no_grad()
    def dissect():                                          # N5 test logit by faintness rank + top-decoy
        X,M,P=pack("test"); yt,noct=tst[2].astype(bool), np.clip(tst[3],1,5); model.eval(); L=[]
        for i in range(0,len(X),256):
            lo,_,_=model(torch.from_numpy(X[i:i+256]).to(DEV),torch.from_numpy(M[i:i+256]).to(DEV),
                         torch.from_numpy(P[i:i+256]).to(DEV) if use_sig else None)
            L.append(lo.cpu().numpy())
        L=np.concatenate(L); sel=np.where(noct==5)[0]; rk=[[] for _ in range(5)]; dec=[]
        for s_ in sel:
            pres=np.where(yt[s_])[0]; order=pres[np.argsort(PHI_TEST[s_,pres])]
            for r,c in enumerate(order):
                if r<5: rk[r].append(L[s_,c])
            ab=np.where(~yt[s_])[0]; dec.append(L[s_,ab].max())
        return [float(np.mean(rk[r])) for r in range(5)], float(np.mean(dec))
    return oracle("dev",dev), oracle("test",tst), dissect()

print("\n"+"="*70)
res={}
for arm in ARMS:
    dv,te=[],[]; diss=None
    for seed in SEEDS:
        t0=time.time(); od,ot,di=run_arm(arm,seed); diss=di
        dv.append(od); te.append(ot)
        print(f"[{arm} seed{seed}] {time.time()-t0:.0f}s  dev N5={od.get(5,float('nan')):.3f} test N5={ot.get(5,float('nan')):.3f}")
    res[arm]=(dv,te,diss)

def avg(ds,k):
    v=[d.get(k,np.nan) for d in ds]; return np.nanmean(v)
print("\n"+"="*70+"\nSUMMARY (mean over seeds)")
print(f"{'arm':>4} | {'DEV oracle N1..N5':>34} | {'TEST oracle N1..N5':>34}")
for arm in ARMS:
    dv,te,_=res[arm]
    ds=" ".join(f"{avg(dv,k):.3f}" for k in range(1,6))
    ts=" ".join(f"{avg(te,k):.3f}" for k in range(1,6))
    print(f"{arm:>4} | {ds:>34} | {ts:>34}")
print("\nN5 logit dissection (last seed): r0=faintest..r4=strongest true + top-decoy")
for arm in ARMS:
    rk,dc=res[arm][2]
    print(f"  {arm:>4}: " + " ".join(f"r{r}={rk[r]:.2f}" for r in range(5)) +
          f"  decoy={dc:.2f}  margin(r0-decoy)={rk[0]-dc:+.2f}")
print("\nJUDGE: R vs A — dev N5 oracle up? + does recon RAISE true(r0..r4) / LOWER decoy / KEEP margin?")

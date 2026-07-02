"""CEILING TEST (controlled, fast) for the additive-amortized direction, in Wiedemer's ideal regime.
Synthetic ADDITIVE mixtures of known-genotype donors; train on a set of combos, TEST on HELD-OUT (novel)
combos. Clean A/B, ASL held FIXED in both arms (only variable = the additive reconstruction constraint):
  A (discriminative): encoder -> per-donor logits ; loss = ASL(logits, y)
  B (amortized-additive): SAME logits + phi_d head ; hhat(l,a)=G*sum_d sigma(logit_d)*phi_d*dosage_d ;
                          loss = ASL(logits, y) + lam*MSE(hhat, h_obs)  [MSE over present AND absent alleles]
If B >> A on held-out combos -> additivity delivers combinatorial generalization (greenlight). If B~A on
clean additive data -> the principle itself doesn't help here. Coverage: every donor appears in train combos
(Wiedemer's support condition), but the TEST combos are unseen. Reports held-out-combo exact-match @k.
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def asl(logits, y, gn=4.0, clip=0.05):
    x=torch.sigmoid(logits); xp=x; xn=(1-x)
    if clip>0: xn=(xn+clip).clamp(max=1)
    los = y*torch.log(xp.clamp(min=1e-8)) + (1-y)*(xn.clamp(min=1e-8)).log()*((x)**gn)
    return -los.mean()

def gen_world(P, L, A, k, n_train_combo, n_test_combo, per_combo, noise, dropout, seed):
    rng=np.random.RandomState(seed)
    # donor genotypes: per (donor,locus) two alleles -> dosage[d,l,a] in {0,1,2}
    dosage=np.zeros((P,L,A),np.float32)
    for d in range(P):
        for l in range(L):
            a1,a2=rng.randint(A),rng.randint(A); dosage[d,l,a1]+=1; dosage[d,l,a2]+=1
    # sample disjoint train/test combos; guarantee every donor covered in train
    def sample_combos(n, ban):
        S=set(); out=[]
        while len(out)<n:
            c=tuple(sorted(rng.choice(P,k,replace=False)))
            if c in S or c in ban: continue
            S.add(c); out.append(c)
        return out,S
    train_c,train_set=sample_combos(n_train_combo,set())
    # ensure coverage: any uncovered donor -> add a combo containing it
    covered=set(x for c in train_c for x in c)
    for d in range(P):
        if d not in covered:
            c=tuple(sorted([d]+list(rng.choice([x for x in range(P) if x!=d],k-1,replace=False))))
            train_c.append(c); train_set.add(c)
    test_c,_=sample_combos(n_test_combo,train_set)
    def build(combos, reps):
        H=[]; Y=[]
        for c in combos:
            for _ in range(reps):
                phi=rng.dirichlet(np.ones(k)).astype(np.float32)
                h=np.zeros((L,A),np.float32)
                for j,d in enumerate(c): h+=phi[j]*dosage[d]
                h*=1000.0
                if noise>0: h=h*np.exp(rng.normal(0,noise,h.shape).astype(np.float32))
                if dropout>0: h[h< (dropout*1000.0)]=0.0       # drop faint peaks -> breaks pure additivity
                y=np.zeros(P,np.float32); y[list(c)]=1
                H.append(h); Y.append(y)
        return np.stack(H), np.stack(Y)
    # train; in-domain test = FRESH mixtures of TRAIN combos (seen combos, unseen draws); novel = held-out combos
    return dosage, build(train_c,per_combo), build(train_c,2), build(test_c,per_combo), set(test_c)

def synth_pool(dosage,P,L,A,k,n,noise,dropout,ban,seed):
    """Compositional-consistency data: synthesize NOVEL-combo mixtures from the KNOWN additive forward model
    (excluding the held-out test combos), so the encoder is forced to invert the decoder on unseen compositions
    WITHOUT touching test data. = the consistency/recombination lever specialized to a fixed known decoder."""
    rng=np.random.RandomState(seed+777); H=[];Y=[]
    while len(H)<n:
        c=tuple(sorted(rng.choice(P,k,replace=False)))
        if c in ban: continue
        phi=rng.dirichlet(np.ones(k)).astype(np.float32); h=np.zeros((L,A),np.float32)
        for j,d in enumerate(c): h+=phi[j]*dosage[d]
        h*=1000.0
        if noise>0: h=h*np.exp(rng.normal(0,noise,h.shape).astype(np.float32))
        if dropout>0: h[h<dropout*1000.0]=0.0
        y=np.zeros(P,np.float32); y[list(c)]=1; H.append(h); Y.append(y)
    return np.stack(H),np.stack(Y)

class Enc(nn.Module):
    def __init__(self, L, A, P, d=128, additive=False):
        super().__init__(); self.L=L; self.A=A; self.P=P; self.additive=additive
        self.locus=nn.Embedding(L,d); self.allele=nn.Embedding(A,d); self.hin=nn.Linear(1,d)
        self.tok=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,d))
        self.post=nn.Sequential(nn.Linear(2*d,d),nn.ReLU())
        self.head=nn.Linear(d,P)
        if additive:
            self.phi=nn.Linear(d,P); self.logG=nn.Parameter(torch.tensor(0.0))
    def forward(self, H):                                   # H: (B,L,A) dense heights
        B=H.shape[0]; dev=H.device
        li=torch.arange(self.L,device=dev).view(1,self.L,1).expand(B,self.L,self.A)
        ai=torch.arange(self.A,device=dev).view(1,1,self.A).expand(B,self.L,self.A)
        logh=torch.log1p(H)
        emb=self.locus(li)+self.allele(ai)+self.hin(logh.unsqueeze(-1))
        t=self.tok(emb)                                     # (B,L,A,d)
        m=(H>0).float().unsqueeze(-1)                       # presence mask
        t=t*m; n=m.sum((1,2)).clamp(min=1)
        mean=t.sum((1,2))/n; mx=t.masked_fill(m==0,-1e9).amax((1,2))
        z=self.post(torch.cat([mean,mx],-1))
        return self.head(z), z

def run_arm(additive, dosage, tr, indom, te, P, L, A, lam, epochs, seed, recon_log=False, dropout_aware=False, tau=50.0, faint_weight=False,
            consistency=False, test_set=None, k=5, cons_noise=0.1, cons_dropout=0.0, w_cons=1.0):
    torch.manual_seed(seed)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Dz=torch.tensor(dosage,device=dev)                      # (P,L,A)
    Htr=torch.tensor(tr[0],device=dev); Ytr=torch.tensor(tr[1],device=dev)
    net=Enc(L,A,P,additive=additive).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
    N=len(Htr); bs=256; scale=Htr.mean().clamp(min=1)
    if consistency:                                          # synthesized novel-combo pool (forward model, excl. test combos)
        Hs_np,Ys_np=synth_pool(dosage,P,L,A,k,4000,cons_noise,cons_dropout,test_set or set(),seed)
        Hs=torch.tensor(Hs_np,device=dev); Ys=torch.tensor(Ys_np,device=dev); Ns=len(Hs)
    def recon_term(H,logit,z):
        w=torch.sigmoid(logit)*F.softplus(net.phi(z))
        hhat=torch.exp(net.logG)*torch.einsum('bp,pla->bla',w,Dz)
        if dropout_aware:
            pres=(H>0).float(); rl_p=((torch.log1p(hhat)-torch.log1p(H))**2)*pres
            over=F.relu(hhat-tau); rl_a=(torch.log1p(over)**2)*(1-pres); return (rl_p+rl_a).mean()
        elif recon_log:
            rl=(torch.log1p(hhat)-torch.log1p(H))**2
            if faint_weight: rl=rl*(1.0/(H/scale+0.1))
            return rl.mean()
        else: return ((hhat-H)**2).mean()/(scale**2)
    for ep in range(epochs):
        perm=torch.randperm(N,device=dev)
        for s in range(0,N,bs):
            bi=perm[s:s+bs]; H=Htr[bi]; y=Ytr[bi]
            logit,z=net(H); loss=asl(logit,y)
            if additive: loss=loss+lam*recon_term(H,logit,z)
            if consistency:                                  # L_cons: encoder must invert decoder on synthesized NOVEL combos
                js=torch.randint(0,Ns,(bs,),device=dev); Hc=Hs[js]
                lc,zc=net(Hc); closs=asl(lc,Ys[js])
                if additive: closs=closs+lam*recon_term(Hc,lc,zc)
                loss=loss+w_cons*closs
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    def em_of(split):
        H=torch.tensor(split[0],device=dev); Y=torch.tensor(split[1],device=dev)
        with torch.no_grad():
            lg,_=net(H); k=int(Y[0].sum().item())
            top=lg.topk(k,dim=1).indices
            pred=torch.zeros_like(Y); pred.scatter_(1,top,1.0)
            return (pred==Y).all(1).float().mean().item()
    return em_of(indom), em_of(te)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--P",type=int,default=30); ap.add_argument("--L",type=int,default=20); ap.add_argument("--A",type=int,default=15)
    ap.add_argument("--k",type=int,default=5); ap.add_argument("--ntr",type=int,default=300); ap.add_argument("--nte",type=int,default=120)
    ap.add_argument("--per",type=int,default=8); ap.add_argument("--noise",type=float,default=0.1); ap.add_argument("--dropout",type=float,default=0.0)
    ap.add_argument("--lam",type=float,default=0.3); ap.add_argument("--epochs",type=int,default=80); ap.add_argument("--seeds",type=int,default=3)
    ap.add_argument("--recon_log",action="store_true"); ap.add_argument("--dropout_aware",action="store_true"); ap.add_argument("--faint_weight",action="store_true")
    ap.add_argument("--consistency",action="store_true"); ap.add_argument("--cons_dropout",type=float,default=-1.0); ap.add_argument("--w_cons",type=float,default=1.0)
    a=ap.parse_args()
    cons_dropout = a.dropout if a.cons_dropout<0 else a.cons_dropout   # default: synthesize with SAME nuisance as test; set >=0 to test misspecification
    print(f"world P={a.P} L={a.L} A={a.A} k={a.k} | train {a.ntr}x{a.per} | test {a.nte} novel | noise={a.noise} dropout={a.dropout} lam={a.lam} | consistency={a.consistency} cons_dropout={cons_dropout} w_cons={a.w_cons}")
    arms=[("A",False,False),("B",True,False)]
    if a.consistency: arms+=[("A+cons",False,True),("B+cons",True,True)]
    res={nm:([],[]) for nm,_,_ in arms}
    for sd in range(42,42+a.seeds):
        dosage,tr,indom,te,test_set=gen_world(a.P,a.L,a.A,a.k,a.ntr,a.nte,a.per,a.noise,a.dropout,sd)
        tau=a.dropout*1000.0 if a.dropout>0 else 1.0
        line=f"  seed{sd}:"
        for nm,add,cons in arms:
            ii,nv=run_arm(add,dosage,tr,indom,te,a.P,a.L,a.A,a.lam,a.epochs,sd,recon_log=a.recon_log,dropout_aware=a.dropout_aware,tau=tau,
                          faint_weight=a.faint_weight,consistency=cons,test_set=test_set,k=a.k,cons_noise=a.noise,cons_dropout=cons_dropout,w_cons=a.w_cons)
            res[nm][0].append(ii); res[nm][1].append(nv); line+=f"  {nm} in{ii:.3f}/nov{nv:.3f}"
        print(line)
    base=np.array(res["A"][1])
    print(f"\nIN-DOMAIN EM:  "+" | ".join(f"{nm}={np.array(res[nm][0]).mean():.3f}" for nm,_,_ in arms))
    print(f"NOVEL-COMBO EM @k={a.k}:")
    for nm,_,_ in arms:
        nv=np.array(res[nm][1]); print(f"   {nm:7s} {nv.mean():.3f}±{nv.std():.3f}  (vs A: {nv.mean()-base.mean():+.3f})")
    print("read: B+cons is the paper-proven pair (additive + consistency). Compare to A+cons (naive augmentation, F29).")

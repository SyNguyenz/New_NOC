"""Does adding the genotype constraint (post-hoc, NO training) close the neural<->symbolic gap on shared peaks?
neural argmax  vs  neural argmax restricted to genotype-carriers  vs  symbolic soft-split."""
import importlib.util, numpy as np, torch
from pathlib import Path
PROJ = Path("."); HERE = PROJ / "inc22_clean"
CKPT = PROJ / "results/inc22_fixed_aslot_seed42/Donor-Slot_Set_Transformer.pt"
DATA = PROJ / "data_insilico_w"; GENO = PROJ / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLELE_OFF, LUT_W, K = 30, 1024, 45

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def build_owner_lut(gg, gm, n=45):
    gm = gm.bool(); o = torch.zeros(24, LUT_W, n)
    for c in range(min(n, gg.size(0))):
        for j in range(gg.size(1)):
            if gm[c, j]:
                li=int(gg[c,j,0]); ab=int(round(float(gg[c,j,1])*10))+ALLELE_OFF
                if 0<=li<24 and 0<=ab<LUT_W: o[li,ab,c]=1.0
    return o

dg = torch.from_numpy(np.load(GENO/"donor_geno.npy").astype(np.float32))
dgm = torch.from_numpy(np.load(GENO/"donor_geno_mask.npy"))
clean = load_module("st", HERE/"models"/"set_transformer.py").SetTransformerMixture(
    n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32, n_classes=45,
    dropout=0.1, n_token_feats=8, periodic_sigma=0.3, n_slot_iters=3, ot_eps=0.05, ot_iters=5,
    donor_geno=dg, donor_geno_mask=dgm, owner_lut=build_owner_lut(dg, dgm)).to(DEVICE)
sd = torch.load(CKPT, weights_only=True, map_location=DEVICE)
assert not clean.load_state_dict(sd, strict=False)[0]; clean.eval()

tok=np.load(DATA/"tokens8_test.npy").astype(np.float32); msk=np.load(DATA/"mask_test.npy")
at=np.load(DATA/"attr_test.npy"); noc=np.load(DATA/"noc_test.npy")
y_true=np.load(PROJ/"results/inc22_fixed_aslot_seed42/y_test_true.npy")
g=dg.numpy(); gm=dgm.numpy()

logits=np.zeros((tok.shape[0],tok.shape[1],K+1),np.float32)
with torch.no_grad():
    for i in range(0,tok.shape[0],256):
        logits[i:i+256]=clean(torch.from_numpy(tok[i:i+256]).to(DEVICE),
                              torch.from_numpy(msk[i:i+256]).to(DEVICE))["logits_attr"].cpu().numpy()

carr={}
for c in range(g.shape[0]):
    for j in range(g.shape[1]):
        if gm[c,j]:
            key=(int(round(g[c,j,0])),int(round(g[c,j,1]*10)))
            carr.setdefault(key,[]);
            if c not in carr[key]: carr[key].append(c)

def symbolic_owner(i,it=10):
    m=msk[i].astype(bool); idx,keys=[],[]
    for p in np.where(m)[0]:
        key=(int(round(tok[i,p,0])),int(round(tok[i,p,1]*10)))
        if key in carr: idx.append(p); keys.append(key)
    if not idx: return {}
    h=np.expm1(tok[i,idx,2].astype(np.float64)); n=len(idx); S=np.full((n,K+1),-1e9)
    for r,key in enumerate(keys):
        for c in carr[key]: S[r,c]=0.0
        S[r,K]=-2.0
    phi=np.ones(K+1)/(K+1)
    for _ in range(it):
        z=S+np.log(phi+1e-9); z-=z.max(1,keepdims=True); A=np.exp(z); A/=A.sum(1,keepdims=True)
        w=(A[:,:K]*h[:,None]).sum(0); bg=(A[:,K]*h).sum(); phi=np.concatenate([w,[bg]])/max(w.sum()+bg,1e-9)
    return {idx[r]:int(A[r].argmax()) for r in range(n)}

real=at>=0
neu=logits.argmax(-1)
rows=[]
for i in range(tok.shape[0]):
    pres=set(np.where(y_true[i]>0.5)[0]); sym=symbolic_owner(i)
    for p in np.where(real[i])[0]:
        o=int(at[i,p]); key=(int(round(tok[i,p,0])),int(round(tok[i,p,1]*10)))
        cars=carr.get(key,[])
        # genotype-masked neural argmax: restrict logits to carriers(+bg)
        ll=logits[i,p].copy(); mask_vec=np.full(K+1,-1e9);
        for c in cars: mask_vec[c]=0.0
        mask_vec[K]=0.0
        gmask_top=int((ll+mask_vec).argmax())
        ncar=sum(1 for c in cars if c in pres)
        rows.append((noc[i], ncar>=2, o, int(neu[i,p]), gmask_top, sym.get(p,-1)))

rows=np.array(rows); ncs,sh,ow,nn,gmk,sym=rows[:,0],rows[:,1].astype(bool),rows[:,2],rows[:,3],rows[:,4],rows[:,5]
v=sym>=0
def acc(m):
    m=m&v; return (nn[m]==ow[m]).mean(),(gmk[m]==ow[m]).mean(),(sym[m]==ow[m]).mean(),m.sum()
print("=== owner acc: NEURAL  vs  NEURAL+genotype-mask(post-hoc,no train)  vs  SYMBOLIC ===")
print("  group          n        neural   neural+gmask   symbolic")
for name,mk in [("PRIVATE",~sh),("SHARED",sh),("ALL",np.ones_like(sh))]:
    a,b,c,n=acc(mk); print(f"  {name:<8} {n:>8,}   {a:.3f}     {b:.3f}        {c:.3f}")
print("\n  SHARED by NOC:")
print("  NOC     n      neural   +gmask   symbolic")
for vv in [2,3,4,5]:
    a,b,c,n=acc(sh&(ncs==vv))
    if n: print(f"   {vv}   {n:>6,}    {a:.3f}    {b:.3f}    {c:.3f}")

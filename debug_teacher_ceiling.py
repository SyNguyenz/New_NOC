"""
User's question on B: "attr_head da hoc het tri thuc chua?" -- is the teacher's ceiling
(~0.82) inherent, or is it because attr_head is a HARD per-peak softmax (1 donor/peak)
that STRUCTURALLY cannot represent shared-allele co-membership (75% of alleles)?

Probe: is the co-membership knowledge already in the frozen encoder H?
  Train a per-peak MULTI-LABEL readout on H (label peak j = 1 for every PRESENT donor
  whose genotype contains peak j's allele -- legit supervision, same as training).
  Eval on held-out half, scoring all 45 donors from H only (NO privilege at eval),
  aggregate max-over-peaks -> donor presence -> N5/N4 oracle.

Compare:
  decoder            : model's sparsemax readout
  attr_head softmax  : the current teacher (hard, 1 donor/peak)
  single-label probe : logistic on H with attr labels (control: same probe, hard label)
  MULTI-label probe  : logistic on H with genotype co-membership labels (soft, credits shared)
If multi >> attr_head/single -> knowledge IS in H; attr_head under-learned it because of
the softmax single-assignment -> the teacher is improvable (fix the LEARNING region).
"""
import json, numpy as np, torch, torch.nn as nn
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
tok=L("tokens8_test").astype(np.float32); mk=L("mask_test").astype(bool)
y=L("y_test_set").astype(np.float32); noc=L("noc_test").astype(int); at=L("attr_test").astype(int)
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
B=len(tok)
gset=[set((round(float(g[c,j,0]),1),round(float(g[c,j,1]),1)) for j in range(g.shape[1]) if gm[c,j]) for c in range(45)]
# (locus,allele) -> donors who own it (over all 45)
def owners(loc,al):
    key=(round(loc,1),round(al,1)); return [c for c in range(45) if key in gset[c]]

def build(ck):
    c=json.load(open(ck/"metrics.json"))["config"]
    m=SetTransformerMixture(n_loci=c["n_loci"],d_locus=c["d_locus"],d_model=c["d_model"],n_heads=c["n_heads"],
        n_isab=c["n_isab"],m_inducing=c["m_inducing"],n_classes=c["n_classes"],n_noc=c["n_noc"],dropout=c["dropout"],
        cls_decoder=c["cls_decoder"],n_token_feats=c["n_token_feats"],encoder=c["encoder"],num_embed=c["num_embed"],
        periodic_sigma=c["periodic_sigma"],aux_heads=c["aux_heads"],sparse_attn=c["sparse_attn"]).to(DEV)
    m.load_state_dict(torch.load(ck/"best_model.pt",weights_only=True,map_location=DEV)); m.eval(); return m

@torch.no_grad()
def enc(m):
    H=np.zeros((B,tok.shape[1],128),np.float32); P=np.zeros((B,45),np.float32); A=np.zeros((B,tok.shape[1],45),np.float32)
    for s in range(0,B,256):
        tk_=torch.from_numpy(tok[s:s+256]).to(DEV); mb=torch.from_numpy(mk[s:s+256]).to(DEV)
        _,h,pad=m._encode_set(tk_,mb); H[s:s+256]=h.cpu().numpy()
        P[s:s+256]=torch.sigmoid(m.cls_decoder_module(h,pad_mask=pad)).cpu().numpy()
        A[s:s+256]=torch.softmax(m.attr_head(h)[:,:,:45],-1).cpu().numpy()
    return H,P,A

EVAL_IDX=None
def n5(P,k):
    idx=np.where(noc==k)[0]
    if EVAL_IDX is not None: idx=np.array([i for i in idx if i in EVAL_IDX])
    e=[]
    for i in idx:
        top=np.argsort(P[i])[::-1][:k]; pr=np.zeros(45,int); pr[top]=1; e.append((pr==y[i]).all())
    return round(float(np.mean(e)),3)

rng=np.random.RandomState(0); perm=rng.permutation(B); fit_i,ev_i=perm[:B//2],perm[B//2:]

def gather(idxs, multi):
    """peak features + label vec (45). multi=True: genotype co-membership among present donors.
       multi=False: single attr donor (hard)."""
    Xs,Ys=[],[]
    for i in idxs:
        present=set(np.where(y[i]>0.5)[0].tolist())
        for j in np.where(mk[i])[0]:
            lab=np.zeros(45,np.float32)
            if multi:
                for c in owners(float(tok[i,j,0]),float(tok[i,j,1])):
                    if c in present: lab[c]=1.0
            else:
                if at[i,j]>=0: lab[at[i,j]]=1.0
            Xs.append(i*1000+j); Ys.append(lab)   # placeholder; H filled later
    return np.array(Xs),np.stack(Ys)

def train_readout(H, multi):
    # collect fit peaks
    Xf,Yf=[],[]
    for i in fit_i:
        present=set(np.where(y[i]>0.5)[0].tolist())
        for j in np.where(mk[i])[0]:
            lab=np.zeros(45,np.float32)
            if multi:
                for c in owners(float(tok[i,j,0]),float(tok[i,j,1])):
                    if c in present: lab[c]=1.0
            else:
                if at[i,j]>=0: lab[at[i,j]]=1.0
            Xf.append(H[i,j]); Yf.append(lab)
    Xf=torch.tensor(np.stack(Xf),device=DEV); Yf=torch.tensor(np.stack(Yf),device=DEV)
    W=nn.Linear(128,45).to(DEV); opt=torch.optim.Adam(W.parameters(),lr=1e-2,weight_decay=1e-4)
    pw=torch.tensor(20.0,device=DEV)
    for ep in range(300):
        opt.zero_grad(); lo=W(Xf)
        loss=nn.functional.binary_cross_entropy_with_logits(lo,Yf,pos_weight=pw); loss.backward(); opt.step()
    return W

@torch.no_grad()
def apply_readout(W,H,idxs):
    P=np.zeros((B,45))
    for i in idxs:
        v=np.where(mk[i])[0]
        if len(v)==0: continue
        pr=torch.sigmoid(W(torch.tensor(H[i,v],device=DEV))).cpu().numpy()  # (nv,45)
        P[i]=pr.max(0)
    return P

m=build(ROOT/"results"/"inc6_maskp_seed42"); H,P,A=enc(m)
# attr_head teacher (max) on eval half
attr=np.zeros((B,45))
for i in ev_i:
    v=mk[i]
    if v.sum(): attr[i]=A[i][v].max(0)
Wsl=train_readout(H,multi=False); Psl=apply_readout(Wsl,H,ev_i)
Wml=train_readout(H,multi=True);  Pml=apply_readout(Wml,H,ev_i)

EVAL_IDX=set(ev_i.tolist())
def ev(P): return f"N4={n5(P,4)}  N5={n5(P,5)}"
print("base encoder H, eval-half readouts (no eval-time privilege):")
print(f"  decoder              {ev(P)}")
print(f"  attr_head softmax    {ev(attr)}      <- current teacher (hard 1-donor/peak)")
print(f"  single-label probe   {ev(Psl)}      <- control: logistic on H, hard attr label")
print(f"  MULTI-label probe    {ev(Pml)}      <- genotype co-membership (credits shared alleles)")

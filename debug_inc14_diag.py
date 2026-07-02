"""Root-cause diagnosis (NO retrain) for Inc14 A-v2 / B-v2 on the real test set.

B-v2 (dev N5 +.029, test flat): decompose into TEACHER quality x TRANSFER.
  - did ml_attr head LEARN co-membership? (its multi-label readout N5 vs the .96 probe ceiling, vs softmax attr)
  - did the decoder ABSORB it? (decoder N5 vs ml_attr-teacher N5 gap)
A-v2 (everything down, train N5 .952->.795): is it COLLAPSE or honest over-regularization?
  - rep-invariance to additive-subtraction (did MMD bite?) base vs A-v2
  - per-donor rep DISCRIMINABILITY: mean cosine between DIFFERENT donors' reps (->1 = collapsed)
"""
import json, numpy as np, torch
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
from train_set_transformer import subtract_height
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
tok=L("tokens8_test").astype(np.float32); mk=L("mask_test").astype(bool)
y=L("y_test_set").astype(np.float32); noc=L("noc_test").astype(int)
phi=L("phi_test").astype(np.float32)
B=len(tok)
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy"); OFF=30; W=1024
owner_lut=torch.zeros(24,W,45)
for c in range(45):
    for j in range(g.shape[1]):
        if gm[c,j]:
            li=int(g[c,j,0]); ab=int(round(float(g[c,j,1])*10))+OFF
            if 0<=li<24 and 0<=ab<W: owner_lut[li,ab,c]=1.0
owner_lut=owner_lut.to(DEV)
def gather_owner(t):
    loc=t[:,:,0].long().clamp(0,23); ab=(torch.round(t[:,:,1]*10).long()+OFF).clamp(0,W-1); return owner_lut[loc,ab]

def build(ck, ml_attr=False):
    c=json.load(open(ck/"metrics.json"))["config"]
    m=SetTransformerMixture(n_loci=c["n_loci"],d_locus=c["d_locus"],d_model=c["d_model"],n_heads=c["n_heads"],
        n_isab=c["n_isab"],m_inducing=c["m_inducing"],n_classes=c["n_classes"],n_noc=c["n_noc"],dropout=c["dropout"],
        cls_decoder=c["cls_decoder"],n_token_feats=c["n_token_feats"],encoder=c["encoder"],num_embed=c["num_embed"],
        periodic_sigma=c["periodic_sigma"],aux_heads=c["aux_heads"],sparse_attn=c["sparse_attn"],
        ml_attr=bool(c.get("ml_attr",False))).to(DEV)
    m.load_state_dict(torch.load(ck/"best_model.pt",weights_only=True,map_location=DEV)); m.eval(); return m

def n5(P,k):
    idx=np.where(noc==k)[0]; e=[]
    for i in idx:
        top=np.argsort(P[i])[::-1][:k]; pr=np.zeros(45,int); pr[top]=1; e.append((pr==y[i]).all())
    return round(float(np.mean(e)),3)

@torch.no_grad()
def readouts(m):
    P=np.zeros((B,45)); soft=np.zeros((B,45)); mlv=np.zeros((B,45))
    for s in range(0,B,256):
        tk=torch.from_numpy(tok[s:s+256]).to(DEV); mb=torch.from_numpy(mk[s:s+256]).to(DEV)
        out=m(tk,mb)
        P[s:s+256]=torch.sigmoid(out["logits_cls"]).cpu().numpy()
        a=torch.softmax(out["logits_attr"][:,:,:45],-1).masked_fill(~mb.bool().unsqueeze(-1),0.0)
        soft[s:s+256]=a.max(1).values.cpu().numpy()
        if "logits_mlattr" in out:
            ml=torch.sigmoid(out["logits_mlattr"]).masked_fill(~mb.bool().unsqueeze(-1),0.0)
            mlv[s:s+256]=ml.max(1).values.cpu().numpy()
    return P,soft,mlv

print("="*80); print("B-v2 — TEACHER quality x TRANSFER  (test oracle per NOC)"); print("="*80)
for nm,ck,ml in [("base","inc6_maskp_seed42",False),("B-v2","inc14_Bv2_seed42",True)]:
    m=build(ROOT/"results"/ck,ml); P,soft,mlv=readouts(m)
    print(f"\n{nm}")
    print(f"  decoder            N4={n5(P,4)}  N5={n5(P,5)}")
    print(f"  attr_head softmax  N4={n5(soft,4)}  N5={n5(soft,5)}")
    if ml: print(f"  ml_attr MULTI-lab  N4={n5(mlv,4)}  N5={n5(mlv,5)}   <- the B-v2 teacher (vs .96 in-domain ceiling)")

print("\n"+"="*80); print("A-v2 — COLLAPSE vs honest regularization"); print("="*80)
@torch.no_grad()
def encode_reps(m, T, Mk):
    R=np.zeros((len(T),45,128),np.float32)
    for s in range(0,len(T),256):
        tk=torch.from_numpy(T[s:s+256]).to(DEV); mb=torch.from_numpy(Mk[s:s+256]).to(DEV)
        _,h,pad=m._encode_set(tk,mb); m.cls_decoder_module(h,pad_mask=pad)
        R[s:s+256]=m.cls_decoder_module.last_reps.cpu().numpy()
    return R
def subtract_cf(idxs):
    T=tok.copy()
    yk=torch.from_numpy(y).to(DEV); ph=torch.from_numpy(phi).to(DEV)
    tk=torch.from_numpy(tok).to(DEV); mb=torch.from_numpy(mk).to(DEV)
    owner=gather_owner(tk); present=(yk>0.5)
    w=owner*ph.clamp(min=0).unsqueeze(1)*present.float().unsqueeze(1)
    rng=np.random.RandomState(0); c_i=torch.zeros(B,dtype=torch.long,device=DEV)
    keep=present.clone()
    for i in range(B):
        pr=np.where(y[i]>0.5)[0]
        if len(pr)>=2:
            c=pr[rng.randint(len(pr))]; c_i[i]=c; keep[i]=present[i].clone(); keep[i,c]=False
        else: keep[i]=False
    wsum=w.sum(-1).clamp(min=1e-6)
    wc=w.gather(-1,c_i.view(-1,1,1).expand(-1,w.size(1),1)).squeeze(-1)
    mult=(1.0-(wc/wsum)).clamp(0,1)
    t_sub=subtract_height(tk,mb,mult).cpu().numpy()
    return t_sub, keep.cpu().numpy()
t_sub,keep=subtract_cf(None)
for nm,ck in [("base","inc6_maskp_seed42"),("A-v2","inc14_Av2_seed42")]:
    m=build(ROOT/"results"/ck,False)
    Rm=encode_reps(m,tok,mk); Rc=encode_reps(m,t_sub,mk)
    idh=slice(0,64)
    # invariance: kept-donor identity-rep cosine main vs additive-subtracted (higher=more invariant)
    inv=[]; disc=[]
    for k in [4,5]:
        ii=np.where(noc==k)[0]
        cs=[];
        for i in ii:
            kk=np.where(keep[i])[0]
            if len(kk)==0: continue
            a=Rm[i][kk][:,idh]; b=Rc[i][kk][:,idh]
            cs.append(np.mean(np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-9)))
        inv.append((k,round(float(np.mean(cs)),4)))
    # discriminability: mean cosine between DIFFERENT present donors' reps (->1 = collapsed)
    for k in [5]:
        ii=np.where(noc==k)[0]; ds=[]
        for i in ii:
            pr=np.where(y[i]>0.5)[0]
            if len(pr)<2: continue
            r=Rm[i][pr][:,idh]; r=r/(np.linalg.norm(r,axis=1,keepdims=True)+1e-9)
            cm=r@r.T; ds.append((cm.sum()-len(pr))/(len(pr)*(len(pr)-1)))
        disc=round(float(np.mean(ds)),4)
    print(f"  {nm:6s} | kept-donor id-rep invariance to ADDITIVE-subtract {inv} | "
          f"inter-donor id-rep cosine N5 (collapse if ->1) = {disc}")

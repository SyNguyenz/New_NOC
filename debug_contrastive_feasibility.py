"""No-train feasibility for the CONTRASTIVE cross-combo direction.

The contrastive objective would make donor d's per-donor rep INVARIANT across combos (align same-donor)
and SEPARATE compatible-absent donors (hard negatives). Its deployable readout = nearest-DONOR-PROTOTYPE.
Simulate that NOW on the (non-contrastive) base encoder:
  prototype_d = mean L2-normalized last_reps[:,d,:] over mixtures where d is PRESENT.
  score(test i, donor d) = cosine(last_reps[i,d], prototype_d) -> top-k -> N5 oracle.
Build prototypes from REAL VAL (pure combo-invariance test, all 45 donors appear) and from SYNTH train.
  proto readout > decoder .788  -> the combo-invariant donor latent is ALREADY partly there; contrastive
                                   (which sharpens alignment + pushes hard negatives) has headroom -> GO.
  proto ~ decoder               -> latent present but not better; contrastive must sharpen -> weak.
  proto << decoder              -> rep is combo-entangled; contrastive must build it -> uncertain.
Also: hard-negative check — do compatible-absent donors get LOWER proto score than present (vs decoder)?
"""
import json, numpy as np, torch
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
gset=[set((round(float(g[c,j,0]),1),round(float(g[c,j,1]),1)) for j in range(g.shape[1]) if gm[c,j]) for c in range(45)]

def build(ck):
    c=json.load(open(ck/"metrics.json"))["config"]
    m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
      dropout=0.1,cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",
      periodic_sigma=c["periodic_sigma"],aux_heads=True,sparse_attn=True).to(DEV)
    m.load_state_dict(torch.load(ck/"best_model.pt",weights_only=True,map_location=DEV)); m.eval(); return m

@torch.no_grad()
def encode(m,split,cap=None):
    tok=L(f"tokens8_{split}").astype(np.float32); mk=L(f"mask_{split}").astype(bool); y=L(f"y_{split}_set").astype(np.float32)
    if cap and len(tok)>cap:
        idx=np.random.RandomState(0).permutation(len(tok))[:cap]; tok,mk,y=tok[idx],mk[idx],y[idx]
    n=len(tok); R=np.zeros((n,45,128),np.float32); P=np.zeros((n,45),np.float32)
    for s in range(0,n,256):
        tk=torch.from_numpy(tok[s:s+256]).to(DEV); mb=torch.from_numpy(mk[s:s+256]).to(DEV)
        _,h,pad=m._encode_set(tk,mb)
        P[s:s+256]=torch.sigmoid(m.cls_decoder_module(h,pad_mask=pad)).cpu().numpy()
        R[s:s+256]=m.cls_decoder_module.last_reps.cpu().numpy()
    Rn=R/ (np.linalg.norm(R,axis=-1,keepdims=True)+1e-9)
    return Rn,P,y,tok,mk

def prototypes(Rn,y):
    proto=np.zeros((45,128),np.float32)
    for d in range(45):
        idx=np.where(y[:,d]>0.5)[0]
        if len(idx): proto[d]=Rn[idx,d].mean(0)
    return proto/(np.linalg.norm(proto,axis=-1,keepdims=True)+1e-9)

m=build(ROOT/"results"/"inc6_maskp_seed42")
Rv,_,yv,_,_=encode(m,"val")
Rs,_,ys,_,_=encode(m,"train",cap=12000)
Rt,Pt,yt,tokt,mkt=encode(m,"test")
noct=L("noc_test").astype(int); B=len(Rt)
proto_v=prototypes(Rv,yv); proto_s=prototypes(Rs,ys)
def score(proto): return np.einsum("ndc,dc->nd",Rt,proto)   # (B,45) cosine to each donor's prototype
Sv=score(proto_v); Ss=score(proto_s)
def n5(Q,k=5):
    idx=np.where(noct==k)[0]; e=[]
    for i in idx:
        t=np.argsort(Q[i])[::-1][:k]; pr=np.zeros(45,int); pr[t]=1; e.append((pr==yt[i]).all())
    return round(float(np.mean(e)),3)
print("real test oracle (k=true NOC):")
print(f"  decoder                 N4={n5(Pt,4)}  N5={n5(Pt,5)}   (baseline)")
print(f"  proto from REAL VAL      N4={n5(Sv,4)}  N5={n5(Sv,5)}   (pure combo-invariance test)")
print(f"  proto from SYNTH train   N4={n5(Ss,4)}  N5={n5(Ss,5)}   (deployable: synth-built)")
# combo-invariance of the rep: nearest-prototype donor-classification accuracy on present donors (test)
def proto_acc(S):
    hit=tot=0
    for i in range(B):
        for d in np.where(yt[i]>0.5)[0]:
            # is d's own prototype the top match among all 45 for slot d's rep? (rank of true donor)
            tot+=1; hit += (np.argmax(S[i])==d) if False else 0
    return None
# hard-negative: present vs compatible-absent under proto(VAL) vs decoder, N5
def obskeys(i): return set((round(float(tokt[i,j,0]),1),round(float(tokt[i,j,1]),1)) for j in np.where(mkt[i])[0])
pres_s=[]; absc_s=[]; pres_d=[]; absc_d=[]
for i in np.where(noct==5)[0]:
    ok=obskeys(i); pr=set(np.where(yt[i]>0.5)[0].tolist())
    for d in range(45):
        comp=len(gset[d]&ok)/max(len(gset[d]),1)
        if d in pr: pres_s.append(Sv[i,d]); pres_d.append(Pt[i,d])
        elif comp>=0.8: absc_s.append(Sv[i,d]); absc_d.append(Pt[i,d])
print("\nN5 present vs compatible-ABSENT (higher present-minus-absent margin = better false-friend rejection):")
print(f"  proto(VAL): present={np.mean(pres_s):.3f}  compat-absent={np.mean(absc_s):.3f}  margin={np.mean(pres_s)-np.mean(absc_s):.3f}")
print(f"  decoder   : present={np.mean(pres_d):.3f}  compat-absent={np.mean(absc_d):.3f}  margin={np.mean(pres_d)-np.mean(absc_d):.3f}")

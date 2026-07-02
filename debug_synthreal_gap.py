"""THE decisive no-train question: the real-test N5 headroom — is it SYNTH->REAL (readout doesn't transfer)
or combo-generalization (info not in H)? And can REAL-VAL calibration (we HAVE labeled real val) close it?

On the BASE synth-trained encoder H (frozen), compare readouts of the SAME H:
  decoder (synth-trained)            : the model's own per-donor readout
  single-label probe, fit on REAL VAL: logistic on H, attr labels, fit on real val -> eval real test
  MULTI-label probe,  fit on REAL VAL: genotype co-membership, fit on real val -> eval real test  (DEPLOYABLE:
                                       we have real val labels + donor genotypes; no test labels used)
  MULTI-label probe,  fit on REAL TEST-half (CV): in-domain ceiling (upper bound)
If real-VAL multi-label >> decoder -> the wall is the READOUT's synth->real transfer (info IS in H);
   lever = real-val-calibrated multi-label readout (cheap, deployable, no generator change).
If real-VAL multi-label ~ decoder -> H lost it / val too small -> the gap is deeper (generator/encoder).
"""
import json, numpy as np, torch, torch.nn as nn
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy"); OFF=30; W=1024
gset=[set((round(float(g[c,j,0]),1),round(float(g[c,j,1]),1)) for j in range(g.shape[1]) if gm[c,j]) for c in range(45)]
def owners_of(loc,al):
    key=(round(loc,1),round(al,1)); return [c for c in range(45) if key in gset[c]]

def build(ck):
    c=json.load(open(ck/"metrics.json"))["config"]
    m=SetTransformerMixture(n_loci=c["n_loci"],d_locus=c["d_locus"],d_model=c["d_model"],n_heads=c["n_heads"],
        n_isab=c["n_isab"],m_inducing=c["m_inducing"],n_classes=c["n_classes"],n_noc=c["n_noc"],dropout=c["dropout"],
        cls_decoder=c["cls_decoder"],n_token_feats=c["n_token_feats"],encoder=c["encoder"],num_embed=c["num_embed"],
        periodic_sigma=c["periodic_sigma"],aux_heads=c["aux_heads"],sparse_attn=c["sparse_attn"]).to(DEV)
    m.load_state_dict(torch.load(ck/"best_model.pt",weights_only=True,map_location=DEV)); m.eval(); return m

@torch.no_grad()
def enc(m,split):
    tok=L(f"tokens8_{split}").astype(np.float32); mk=L(f"mask_{split}").astype(bool)
    n=len(tok); H=np.zeros((n,tok.shape[1],128),np.float32); P=np.zeros((n,45),np.float32)
    for s in range(0,n,256):
        tk=torch.from_numpy(tok[s:s+256]).to(DEV); mb=torch.from_numpy(mk[s:s+256]).to(DEV)
        _,h,pad=m._encode_set(tk,mb); H[s:s+256]=h.cpu().numpy()
        P[s:s+256]=torch.sigmoid(m.cls_decoder_module(h,pad_mask=pad)).cpu().numpy()
    return H,P,tok,mk

def labels(tok,mk,y,attr,multi):
    """per-peak label rows + their H index list, built lazily by caller."""
    pass

def train_readout(H,tok,mk,y,attr,idxs,multi):
    Xs,Ys=[],[]
    for i in idxs:
        present=set(np.where(y[i]>0.5)[0].tolist())
        for j in np.where(mk[i])[0]:
            lab=np.zeros(45,np.float32)
            if multi:
                for c in owners_of(float(tok[i,j,0]),float(tok[i,j,1])):
                    if c in present: lab[c]=1.0
            else:
                if attr[i,j]>=0: lab[attr[i,j]]=1.0
            Xs.append(H[i,j]); Ys.append(lab)
    X=torch.tensor(np.stack(Xs),device=DEV); Y=torch.tensor(np.stack(Ys),device=DEV)
    Wt=nn.Linear(128,45).to(DEV); opt=torch.optim.Adam(Wt.parameters(),lr=1e-2,weight_decay=1e-4)
    pw=torch.tensor(20.0,device=DEV)
    for _ in range(300):
        opt.zero_grad(); l=nn.functional.binary_cross_entropy_with_logits(Wt(X),Y,pos_weight=pw); l.backward(); opt.step()
    return Wt
@torch.no_grad()
def apply_readout(Wt,H,mk,idxs):
    P=np.zeros((len(H),45))
    for i in idxs:
        v=np.where(mk[i])[0]
        if len(v): P[i]=torch.sigmoid(Wt(torch.tensor(H[i,v],device=DEV))).cpu().numpy().max(0)
    return P

m=build(ROOT/"results"/"inc6_maskp_seed42")
Hv,_,tokv,mkv=enc(m,"val");  yv=L("y_val_set").astype(np.float32);  atv=L("attr_val").astype(int)
Ht,Pt,tokt,mkt=enc(m,"test"); yt=L("y_test_set").astype(np.float32); att=L("attr_test").astype(int); noct=L("noc_test").astype(int)
B=len(Ht)
def n5(P,k,idxs=None):
    idx=np.where(noct==k)[0]; idx=[i for i in idx if (idxs is None or i in idxs)]
    e=[]
    for i in idx:
        t=np.argsort(P[i])[::-1][:k]; pr=np.zeros(45,int); pr[t]=1; e.append((pr==yt[i]).all())
    return round(float(np.mean(e)),3) if e else None

rng=np.random.RandomState(0); perm=rng.permutation(B); fit_i,ev_i=perm[:B//2],perm[B//2:]; ev_set=set(ev_i.tolist())
print("BASE encoder H (synth-trained, frozen) — readouts on REAL test:")
print(f"  decoder (synth)              N4={n5(Pt,4)}  N5={n5(Pt,5)}")
Wsl_v=train_readout(Hv,tokv,mkv,yv,atv,range(len(Hv)),multi=False)
Wml_v=train_readout(Hv,tokv,mkv,yv,atv,range(len(Hv)),multi=True)
Psl_v=apply_readout(Wsl_v,Ht,mkt,range(B)); Pml_v=apply_readout(Wml_v,Ht,mkt,range(B))
print(f"  single-label, fit REAL VAL   N4={n5(Psl_v,4)}  N5={n5(Psl_v,5)}   (deployable: real val labels)")
print(f"  MULTI-label,  fit REAL VAL   N4={n5(Pml_v,4)}  N5={n5(Pml_v,5)}   (deployable: real val + genotypes)")
Wml_t=train_readout(Ht,tokt,mkt,yt,att,fit_i,multi=True)
Pml_t=apply_readout(Wml_t,Ht,mkt,ev_i)
print(f"  MULTI-label,  fit REAL TEST  N4={n5(Pml_t,4,ev_set)}  N5={n5(Pml_t,5,ev_set)}   (in-domain CEILING, eval-half)")

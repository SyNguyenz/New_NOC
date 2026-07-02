"""
Are levers A and B each "triet de" (saturated / fully solving their own problem)
BEFORE we combine them?

A (counterfactual rep-invariance): its PROBLEM was collapse; rep-level fixed that.
  But did the loss actually CHANGE anything? Probe: remove ONE present donor's peaks,
  re-encode, measure cosine of the KEPT donors' per-donor reps (full vs ablated).
    A invariance >> base  -> loss bit; mechanism worked.
    A invariance ~= base  -> loss too weak (under-leveraged) -> raise weight.   <- user's hunch
  Then cross with N5: if invariance rose but N5 flat -> premise weak (invariance != N5).

B (distill attr_head -> decoder): its PROBLEM was decoder under-reads vs combo-invariant
  attr-vote. Probe the student(decoder)-vs-teacher(attr vote) GAP per NOC.
    base: decoder << teacher (the headroom B targets).
    B: did decoder close the gap to its teacher?
       gap ~= 0  -> SATURATED; teacher is the ceiling -> to push B, build a BETTER teacher.
       gap > 0   -> NOT saturated -> push distill harder (weight/ramp).
  Also report alt teacher aggregations (sum/logsumexp/top3) = candidate stronger teachers.
"""
import json, numpy as np, torch
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
tok=L("tokens8_test").astype(np.float32); mk=L("mask_test").astype(bool)
y=L("y_test_set").astype(np.float32); noc=L("noc_test").astype(int); at=L("attr_test").astype(int)
B=len(tok)

def build(ck):
    c=json.load(open(ck/"metrics.json"))["config"]
    m=SetTransformerMixture(n_loci=c["n_loci"],d_locus=c["d_locus"],d_model=c["d_model"],n_heads=c["n_heads"],
        n_isab=c["n_isab"],m_inducing=c["m_inducing"],n_classes=c["n_classes"],n_noc=c["n_noc"],dropout=c["dropout"],
        cls_decoder=c["cls_decoder"],n_token_feats=c["n_token_feats"],encoder=c["encoder"],num_embed=c["num_embed"],
        periodic_sigma=c["periodic_sigma"],aux_heads=c["aux_heads"],sparse_attn=c["sparse_attn"]).to(DEV)
    m.load_state_dict(torch.load(ck/"best_model.pt",weights_only=True,map_location=DEV)); m.eval(); return m

@torch.no_grad()
def enc(m):
    P=np.zeros((B,45),np.float32); A=np.zeros((B,tok.shape[1],45),np.float32)
    for s in range(0,B,256):
        tk_=torch.from_numpy(tok[s:s+256]).to(DEV); mb=torch.from_numpy(mk[s:s+256]).to(DEV)
        _,h,pad=m._encode_set(tk_,mb)
        P[s:s+256]=torch.sigmoid(m.cls_decoder_module(h,pad_mask=pad)).cpu().numpy()
        A[s:s+256]=torch.softmax(m.attr_head(h)[:,:,:45],-1).cpu().numpy()
    return P,A

def vote(A,mode="max"):
    sc=np.zeros((B,45))
    for i in range(B):
        v=mk[i]
        if not v.sum(): continue
        a=A[i][v]
        if mode=="max": sc[i]=a.max(0)
        elif mode=="sum": sc[i]=a.sum(0)
        elif mode=="top3": sc[i]=np.sort(a,0)[::-1][:3].sum(0)
        elif mode=="noisyor": sc[i]=1.0-np.prod(1.0-a,0)   # P(>=1 peak is this donor); valid [0,1] BCE target
    return sc

def n5(P,k):
    idx=np.where(noc==k)[0]; e=[]
    for i in idx:
        top=np.argsort(P[i])[::-1][:k]; pr=np.zeros(45,int); pr[top]=1; e.append((pr==y[i]).all())
    return round(float(np.mean(e)),3)

@torch.no_grad()
def rep_invariance(m, ks=(4,5)):
    """For each sample: remove one present donor's attributed peaks, re-encode,
    cosine of KEPT donors' per-donor reps (full vs ablated). Mean per NOC."""
    rng=np.random.RandomState(0); out={}
    for k in ks:
        idx=np.where(noc==k)[0]; coss=[]
        for s in range(0,len(idx),256):
            bi=idx[s:s+256]
            tk_=torch.from_numpy(tok[bi]).to(DEV); mb=torch.from_numpy(mk[bi]).to(DEV)
            _,h,pad=m._encode_set(tk_,mb); m.cls_decoder_module(h,pad_mask=pad)
            reps_full=m.cls_decoder_module.last_reps.clone()       # (b,45,d)
            mb_cf=mb.clone(); kept=np.zeros((len(bi),45),bool)
            for j,i in enumerate(bi):
                present=np.where(y[i]>0.5)[0]
                c=present[rng.randint(len(present))]              # random present donor to remove
                rm=(at[i]==c)&mk[i]
                if rm.sum()==0 or (mk[i].sum()-rm.sum())==0:
                    kept[j]=False; continue
                mb_cf[j, np.where(rm)[0]]=False
                kp=np.zeros(45,bool); kp[present]=True; kp[c]=False; kept[j]=kp
            _,h2,pad2=m._encode_set(tk_,mb_cf.bool()); m.cls_decoder_module(h2,pad_mask=pad2)
            reps_cf=m.cls_decoder_module.last_reps                 # (b,45,d)
            cos=torch.nn.functional.cosine_similarity(reps_full,reps_cf,dim=-1).cpu().numpy()  # (b,45)
            for j in range(len(bi)):
                if kept[j].any(): coss.append(cos[j][kept[j]].mean())
        out[k]=round(float(np.mean(coss)),4)
    return out

arms={"base":"inc6_maskp_seed42","A_cfinv":"inc13_A_cfinv_seed42",
      "B_distill":"inc13_B_distill_seed42","C_ladder":"inc13_C_ladder_seed42"}
print("="*78)
print("PART 1 -- B saturation: decoder(student) vs attr-vote(teacher) N5/N4 oracle")
print("="*78)
for name,sub in arms.items():
    m=build(ROOT/"results"/sub); P,A=enc(m)
    tea=vote(A,"max"); tea_s=vote(A,"sum"); tea_t=vote(A,"top3"); tea_n=vote(A,"noisyor")
    print(f"\n{name}")
    print(f"  decoder         N4={n5(P,4)}  N5={n5(P,5)}")
    print(f"  teacher max     N4={n5(tea,4)}  N5={n5(tea,5)}   (gap N5 = {n5(tea,5)-n5(P,5):+.3f})")
    print(f"  teacher noisyOR N4={n5(tea_n,4)}  N5={n5(tea_n,5)}   <- valid [0,1] distill target")
    print(f"  teacher sum     N4={n5(tea_s,4)}  N5={n5(tea_s,5)}")
    print(f"  teacher top3    N4={n5(tea_t,4)}  N5={n5(tea_t,5)}")

print("\n"+"="*78)
print("PART 2 -- A mechanism: kept-donor rep cosine under co-donor removal (higher=more invariant)")
print("="*78)
for name,sub in arms.items():
    m=build(ROOT/"results"/sub); inv=rep_invariance(m)
    print(f"  {name:12s}  N4={inv[4]}  N5={inv[5]}")

"""
On PRODUCTION (inc6_maskp): true-donor evidence is high (AUC .985). But the N5 wall is DECOYS —
absent donors the model ranks INTO top-k, displacing a faint true donor. The decisive question for
the sig-head: on the model's OWN error cases, does the signature score rank the MISSED-TRUE above
the DECOY?  If yes -> sig-head is COMPLEMENTARY (its additive logit would flip the error). If the
sig score is just as fooled (decoy >= missed-true) -> sig-head is REDUNDANT, won't help full-scale.

Groups (per N5 sample, by model top-k=true-NOC):
  true_hit   : true donor IN model top-k
  true_miss  : true donor NOT in top-k          <- the faint minor the model loses
  decoy      : absent donor IN model top-k        <- the false positive that displaced it
  absent_rest: absent donor not in top-k
Measure production logit AND signature score for each group; then the pairwise (miss,decoy) contrast.
"""
import os, json, itertools
from pathlib import Path
import numpy as np, torch

DATA = Path(os.environ.get("STR_DATA_DIR", "data_insilico_w"))
RUN  = Path(os.environ.get("RUN", "results/inc6_maskp_seed42"))
GENO = Path(os.environ.get("STR_GENO", "data/donor_geno.npy"))
MAX_ORDER, TOPK = 3, 60
DEVc = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ab(a): return int(round(float(a)*10))
def kk(l,a): return (int(round(float(l))), ab(a))

# ---- mine minimal JEP signatures (panel only) ----
g=np.load(GENO); gm=np.load(str(GENO).replace(".npy","_mask.npy")).astype(bool); C=g.shape[0]
ditems=[set(kk(g[c,j,0],g[c,j,1]) for j in range(g.shape[1]) if gm[c,j]) for c in range(C)]
own={}
for c in range(C):
    for it in ditems[c]: own.setdefault(it,set()).add(c)
def rar(it): return float(np.log(C/len(own[it])))
def mine(c):
    A=sorted(ditems[c]); sigs=[(it,) for it in A if own[it]=={c}]; nonp=[it for it in A if own[it]!={c}]
    cc=[(x,y,rar(x)+rar(y)) for x,y in itertools.combinations(nonp,2) if not((own[x]&own[y])-{c})]
    cc.sort(key=lambda t:-t[2]); sigs+=[(x,y) for x,y,_ in cc[:TOPK]]
    cc=[]
    for x,y,z in itertools.combinations(nonp,3):
        if not((own[x]&own[y])-{c}): continue
        if not((own[x]&own[z])-{c}): continue
        if not((own[y]&own[z])-{c}): continue
        if not((own[x]&own[y]&own[z])-{c}): cc.append((x,y,z,rar(x)+rar(y)+rar(z)))
    cc.sort(key=lambda t:-t[3]); sigs+=[(x,y,z) for x,y,z,_ in cc[:TOPK]]
    return [(s,sum(rar(i) for i in s)) for s in sigs]
JEP=[mine(c) for c in range(C)]

def sig_score(tok,msk):
    N=tok.shape[0]; S=np.zeros((N,C))
    for i in range(N):
        O=set(kk(tok[i,k,0],tok[i,k,1]) for k in np.where(msk[i])[0])
        for c in range(C):
            t=0.0
            for items,w in JEP[c]:
                if all(it in O for it in items): t+=w
            S[i,c]=t
    return S

# ---- production model -> raw logits ----
from models.set_transformer import SetTransformerMixture
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats",8); tp=f"tokens{n_tok}"
def load(sp): return (np.load(DATA/f"{tp}_{sp}.npy").astype(np.float32),np.load(DATA/f"mask_{sp}.npy"),
                      np.load(DATA/f"y_{sp}_set.npy").astype(bool),np.load(DATA/f"noc_{sp}.npy").astype(int))
model=SetTransformerMixture(
    n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),
    n_classes=45,n_noc=6,dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","per_donor"),
    decoder_source=cfg.get("decoder_source","encoded"),n_token_feats=n_tok,encoder=cfg.get("encoder","isab++"),
    dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","periodic"),n_freq=cfg.get("n_freq",8),
    d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",0.3),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False)).to(DEVc)
sd=torch.load(RUN/"best_model.pt",map_location=DEVc); sd=sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd
model.load_state_dict(sd,strict=False); model.eval()
@torch.no_grad()
def logits(tok,msk):
    L=[]
    for i in range(0,len(tok),256):
        o=model(torch.from_numpy(tok[i:i+256]).to(DEVc),torch.from_numpy(msk[i:i+256].astype(bool)).to(DEVc))
        L.append(o["logits_cls"].cpu().numpy())
    return np.concatenate(L)

tk,mk,y,noc=load("test")
print(f"run={RUN.name}  N5={(noc==5).sum()}")
Lg=logits(tk,mk); Sg=sig_score(tk,mk)

def auc(pos,neg):
    pos,neg=np.asarray(pos),np.asarray(neg)
    if not len(pos) or not len(neg): return float("nan")
    a=np.concatenate([pos,neg]); _,inv,cnt=np.unique(a,return_inverse=True,return_counts=True)
    cs=np.cumsum(cnt); rk=((cs-cnt+cs+1)/2.0)[inv]
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

for NV in [5,4]:
    sel=np.where(noc==NV)[0]
    grp={g:{"L":[],"S":[]} for g in ["true_hit","true_miss","decoy","absent_rest"]}
    pair_prodflip=pair_sigflip=pair_n=0
    for i in sel:
        top=set(np.argsort(Lg[i])[::-1][:NV]); tru=set(np.where(y[i])[0])
        miss=sorted(tru-top); dec=sorted((set(range(C))-tru)&top)
        for c in range(C):
            if y[i,c] and c in top: g="true_hit"
            elif y[i,c]:            g="true_miss"
            elif c in top:          g="decoy"
            else:                   g="absent_rest"
            grp[g]["L"].append(Lg[i,c]); grp[g]["S"].append(Sg[i,c])
        for mt in miss:
            for dc in dec:
                pair_n+=1
                pair_prodflip+=int(Lg[i,dc] >  Lg[i,mt])   # production: decoy beats missed-true (the error)
                pair_sigflip += int(Sg[i,mt] >  Sg[i,dc])   # signature : missed-true beats decoy (the repair)
    print(f"\n===== N{NV}  (samples={len(sel)}) =====")
    print(f"{'group':>12} {'n':>6} {'prod_logit':>11} {'sig_score':>10}")
    for gname in ["true_hit","true_miss","decoy","absent_rest"]:
        L=grp[gname]["L"]; S=grp[gname]["S"]
        print(f"{gname:>12} {len(L):>6} {np.mean(L):>11.3f} {np.mean(S):>10.3f}")
    # AUC on the HARD discrimination the model gets wrong: missed-true vs decoy
    a_prod=auc(grp["true_miss"]["L"], grp["decoy"]["L"])
    a_sig =auc(grp["true_miss"]["S"], grp["decoy"]["S"])
    print(f"  AUC(missed-true vs decoy):  production={a_prod:.3f}   signature={a_sig:.3f}   (0.5=blind)")
    print(f"  pairwise (missed-true,decoy)={pair_n}:  production ranks decoy>true {pair_prodflip/max(1,pair_n):.3f} (=the error)"
          f"  |  signature ranks true>decoy {pair_sigflip/max(1,pair_n):.3f} (=repair potential)")

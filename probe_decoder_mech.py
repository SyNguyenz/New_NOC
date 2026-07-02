"""
Decoder mechanism: is the drop of a readable faint minor 'UNDER-READ' (decoder suppresses the faint
donor's score, an arbitrary filler takes the slot) or 'COMBO-MEMORIZATION' (decoder confidently ranks
a NON-contributor that co-occurs with the present majors in TRAIN, displacing the true minor)?

On the 38 pure-DECODER-miss faint minors (H isolates them, decoder drops them), measure:
  (A) decoder rank (among 45) of the true minor -> shallow miss vs buried.
  (B) train co-occurrence with the 3 majors:  for the DISPLACER (false donor in decoder top-5) vs the
      TRUE minor.  cooc(x|M) = mean_{m in M} P(x present | m present) from TRAIN sets.
        displacer_cooc >> minor_cooc  => decoder preferred a memorized partner = COMBO-MEMORIZATION.
        displacer_cooc ~= minor_cooc  => no partner preference = UNDER-READ (minor suppressed, filler).
  (C) decoder score of the true minor: is it near the floor (suppressed) or just below threshold?
Control = both-KEEP minors.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc8_v2_vicreg_inv_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
    vib=cfg.get("vib",False),mass_pool=cfg.get("mass_pool",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()
print(f"loaded {RUN.name}")

def load(s):
    return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
            np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))

# ---- train co-occurrence (conditional P(x present | m present)) ----
_,_,at_tr=load("train")
freq=np.zeros(45); co=np.zeros((45,45))
for gi in range(len(at_tr)):
    d=np.unique(at_tr[gi]); d=d[d>=0].astype(int)
    for a in d:
        freq[a]+=1
        for b in d: co[a,b]+=1
def cond(x,M):  # mean_{m in M} P(x|m)
    vs=[co[x,m]/freq[m] for m in M if freq[m]>0]
    return float(np.mean(vs)) if vs else 0.0

# ---- per-peak probe on train H (to rebuild cells) ----
@torch.no_grad()
def encode_full(tk,mk,idxs,bs=128):
    out={}
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        _,H,_=model._encode_set(t,m); H=H.cpu().numpy()
        for j,gi in enumerate(sel): out[int(gi)]=H[j]
    return out
tk_tr,mk_tr,_=load("train"); rng=np.random.default_rng(0)
Hm=encode_full(tk_tr,mk_tr,rng.choice(len(at_tr),size=6000,replace=False))
HH=[];DD=[]
for gi,H in Hm.items():
    a=at_tr[gi]; v=np.where(a>=0)[0]; HH.append(H[v]); DD.append(a[v])
clf=nn.Linear(128,45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
Xt=torch.from_numpy(np.concatenate(HH).astype(np.float32)).to(DEV); yt=torch.from_numpy(np.concatenate(DD).astype(int)).long().to(DEV)
lf=nn.CrossEntropyLoss()
for ep in range(60):
    perm=torch.randperm(len(yt),device=DEV)
    for s in range(0,len(yt),8192):
        b=perm[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
W=clf.weight.detach().cpu().numpy(); B=clf.bias.detach().cpu().numpy()
def smrows(X):
    Z=X@W.T+B; Z-=Z.max(1,keepdims=True); E=np.exp(Z); return E/E.sum(1,keepdims=True)
print("probe fit + co-occurrence built")

# ---- test ----
tk,mk,at=load("test")
samp={}
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: continue
    lh=tk[gi][:,2]; info={}
    for d in np.unique(a[v]): info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
    order=sorted(info,key=lambda d:-info[d]["h"])
    for r,d in enumerate(order): info[d]["rank"]=r
    samp[gi]={"info":info,"noc":len(order)}
keep=[g for g in samp if samp[g]["noc"]==5]
Hte=encode_full(tk,mk,np.array(keep))
@torch.no_grad()
def mscores(idxs,bs=256):
    out=np.zeros((len(idxs),45),np.float32)
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; o=model(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        out[s:s+len(sel)]=torch.sigmoid(o["logits_cls"]).cpu().numpy()
    return out
ms=mscores(np.array(keep)); ms_map={g:ms[i] for i,g in enumerate(keep)}

cells=defaultdict(list)
for g in keep:
    a=at[g]; v=np.where(a>=0)[0]; vote=smrows(Hte[g][v]).sum(0)
    ptop=set(np.argsort(vote)[::-1][:5].tolist()); dtop=set(np.argsort(ms_map[g])[::-1][:5].tolist())
    d=[dd for dd in samp[g]["info"] if samp[g]["info"][dd]["rank"]==4][0]
    cells[(d in ptop, d in dtop)].append(g)

def analyze(gs,lab):
    ranks=[]; minor_sc=[]; disp_sc=[]; minor_cooc=[]; disp_cooc=[]; disp_is_false=0; ndisp=0
    for g in gs:
        info=samp[g]["info"]; true=set(info.keys())
        d=[dd for dd in info if info[dd]["rank"]==4][0]
        M=[dd for dd in info if info[dd]["rank"]<3]
        sc=ms_map[g]; order=np.argsort(sc)[::-1]
        ranks.append(int(np.where(order==d)[0][0])+1)   # 1-based decoder rank of true minor
        minor_sc.append(float(sc[d])); minor_cooc.append(cond(d,M))
        top5=set(order[:5].tolist()); displacers=[x for x in top5 if x not in true]
        for x in displacers:
            ndisp+=1; disp_is_false+=1
            disp_sc.append(float(sc[x])); disp_cooc.append(cond(x,M))
    f=lambda a: float(np.mean(a)) if a else float('nan')
    print(f"\n  [{lab}]  n={len(gs)}")
    print(f"    (A) decoder rank of TRUE minor (1=top): mean {f(ranks):.1f}  median {np.median(ranks):.0f}")
    print(f"    (C) decoder score: true minor {f(minor_sc):.3f}   displacer {f(disp_sc):.3f}")
    print(f"    (B) train cooc w/ majors: true minor {f(minor_cooc):.3f}   displacer {f(disp_cooc):.3f}   (displacer>>minor => combo-memorization)")

analyze(cells[(True,False)],"pure DECODER-miss")
analyze(cells[(True,True)],"both KEEP (control)")
print("\n  under-read  => true-minor score near floor & buried deep; displacer cooc ~= minor cooc (arbitrary filler).")
print("  combo-mem   => displacer score high & displacer cooc >> true-minor cooc (memorized partner wins the slot).")

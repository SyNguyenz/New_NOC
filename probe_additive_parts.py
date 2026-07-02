"""
Probe the additive-decoder checkpoints PART BY PART vs per_donor, to see if the additive readout closes the
under-read (model decode vs its own encoder soft-vote ceiling) and handles the decisive-allele / decoy problem.
For each run: per-NOC  model-oracle (rank the model's logits_cls, top-true-NOC, set-EM)  vs  soft-vote ceiling
(linear probe on the SAME H)  vs gap.  Smaller gap => the decoder reads its encoder better.
Plus: on N5 set-misses, does the model recover the IDENTIFIABLE dropped donor (mean 2.9 decisive alleles)?
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,"."); from models.set_transformer import SetTransformerMixture
DATA=Path("data_insilico_w"); DEV="cuda" if torch.cuda.is_available() else "cpu"
RUNS=[Path(p) for p in sys.argv[1:]] or [Path("results/inc3_repC_additive_seed42")]
def build(run):
    cfg=json.load(open(run/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats") or 8
    m=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
        dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
        n_token_feats=n_tok,encoder=cfg.get("encoder","isab") or "isab",num_embed=cfg.get("num_embed","raw") or "raw",
        n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
        aux_heads=cfg.get("aux_heads",False),sparse_attn=bool(cfg.get("sparse_attn") or False),
        nc_attn=cfg.get("nc_attn") or "none",nc_learnable_bias=bool(cfg.get("nc_learnable_bias") or False)).to(DEV)
    sd=torch.load(run/"best_model.pt",map_location=DEV,weights_only=True); miss,unexp=m.load_state_dict(sd,strict=False)
    bad=[k for k in list(miss)+list(unexp) if "encoder" in k or "decoder" in k or "cls" in k]
    if bad: print(f"  !! {run.name}: {len(bad)} key mismatches e.g. {bad[:3]}")
    m.eval(); return m,n_tok,cfg
def load(s,n_tok): return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
                           np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
def fit(Hs,Ds):
    X=torch.from_numpy(np.concatenate(Hs).astype(np.float32)).to(DEV); y=torch.from_numpy(np.concatenate(Ds).astype(int)).long().to(DEV)
    clf=nn.Linear(X.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
    for ep in range(50):
        perm=torch.randperm(len(y),device=DEV)
        for s in range(0,len(y),8192):
            b=perm[s:s+8192]; opt.zero_grad(); lf(clf(X[b]),y[b]).backward(); opt.step()
    return clf.weight.detach().cpu().numpy(), clf.bias.detach().cpu().numpy()
for run in RUNS:
    model,n_tok,cfg=build(run)
    @torch.no_grad()
    def enc(tk,mk,idx,bs=128):
        out={}
        for s in range(0,len(idx),bs):
            sel=idx[s:s+bs]; _,H,_=model._encode_set(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
            for j,gi in enumerate(sel): out[int(gi)]=H[j].cpu().numpy()
        return out
    @torch.no_grad()
    def mlogit(tk,mk,idx,bs=256):
        out=np.zeros((len(idx),45),np.float32)
        for s in range(0,len(idx),bs):
            sel=idx[s:s+bs]; o=model(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
            out[s:s+len(sel)]=o["logits_cls"].cpu().numpy()
        return out
    tk_tr,mk_tr,at_tr=load("train",n_tok); rng=np.random.default_rng(0)
    Hm=enc(tk_tr,mk_tr,rng.choice(len(at_tr),size=5000,replace=False)); HH=[];DD=[]
    for gi,H in Hm.items():
        a=at_tr[gi]; v=np.where(a>=0)[0]; HH.append(H[v]); DD.append(a[v])
    W,B=fit(HH,DD)
    tk,mk,at=load("test",n_tok); ids=np.array([g for g in range(len(at)) if (at[g]>=0).any()])
    Hte=enc(tk,mk,ids); ML=mlogit(tk,mk,ids); MLm={g:ML[i] for i,g in enumerate(ids)}
    mo=defaultdict(lambda:[0,0]); ho=defaultdict(lambda:[0,0])
    for g in ids:
        a=at[g]; v=np.where(a>=0)[0]; true=set(int(x) for x in np.unique(a[v])); noc=len(true)
        mt=set(np.argsort(MLm[g])[::-1][:noc].tolist()); mo[noc][0]+=(mt==true); mo[noc][1]+=1
        z=Hte[g][v]@W.T+B; z-=z.max(1,keepdims=True); P=np.exp(z);P/=P.sum(1,keepdims=True)
        ht=set(np.argsort(P.sum(0))[::-1][:noc].tolist()); ho[noc][0]+=(ht==true); ho[noc][1]+=1
    print(f"\n=== {run.name}  (cls_decoder={cfg.get('cls_decoder')}, sparse={bool(cfg.get('sparse_attn') or False)}, nc_attn={cfg.get('nc_attn') or 'none'}) ===")
    print("  NOC   model-oracle   softvote-ceiling   gap(ceil-model)")
    for noc in range(1,6):
        if mo[noc][1]:
            mm=mo[noc][0]/mo[noc][1]; hh=ho[noc][0]/ho[noc][1]
            print(f"   {noc}      {mm:.3f}          {hh:.3f}            {hh-mm:+.3f}")

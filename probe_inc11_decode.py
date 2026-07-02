"""
Inc11 part 2 — separate ENCODER-info from MODEL-READOUT, and locate mab0's residual wall.

[C] per-NOC set-oracle EM two ways on REAL TEST:
      model-decode  = rank model's 45 cls probs, take top-(true NOC), set-match   (= encoder + trained decoder)
      H-linear-vote = pooled soft-vote of the SAME linear identity probe over all peaks (= encoder info only)
    If H-vote >> model-decode for an arm, the loss is in the DECODER/readout, not the encoder.

[D] faintest-minor recall vs #PRIVATE alleles present (model-decode), base vs mab0.
    F34/F35 said recall is gated by #private alleles (conjunctive under-determination). Shows whether mab0's
    gain comes from the isolation component (helps absorbed peaks) or touches the few-private-allele residual.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w")
RUNS=[Path(p) for p in sys.argv[1:]] or [
    Path("results/inc2_2d_sparse_seed42"), Path("results/inc11_nc_mab0_seed42"), Path("results/inc11_nc_both_seed42")]
DEV="cuda" if torch.cuda.is_available() else "cpu"; geno=load_raw_genotypes()

def build(run):
    cfg=json.load(open(run/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
    m=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
        dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
        n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
        n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
        aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),vib=cfg.get("vib",False),
        mass_pool=cfg.get("mass_pool",False),nc_attn=cfg.get("nc_attn","none"),nc_learnable_bias=cfg.get("nc_learnable_bias",False)).to(DEV)
    m.load_state_dict(torch.load(run/"best_model.pt",map_location=DEV,weights_only=True),strict=False); m.eval()
    return m,n_tok
def load(s,n_tok):
    return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
            np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")

@torch.no_grad()
def encode(model,tk,mk,idxs,bs=128):
    out={}
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; _,H,_=model._encode_set(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        for j,gi in enumerate(sel): out[int(gi)]=H[j].cpu().numpy()
    return out
@torch.no_grad()
def mscores(model,tk,mk,idxs,bs=256):
    out=np.zeros((len(idxs),45),np.float32)
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; o=model(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        out[s:s+len(sel)]=torch.sigmoid(o["logits_cls"]).cpu().numpy()
    return out
def private_of(g,d,info,tk):
    others=[KNOWN[o] for o in info if o!=d]; gX=geno.get(KNOWN[d],{}); priv=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: priv.add((L,a))
    return priv

print(f"device={DEV}")
results={}
for run in RUNS:
    model,n_tok=build(run); tk_tr,mk_tr,at_tr=load("train",n_tok); tk,mk,at=load("test",n_tok)
    rng=np.random.default_rng(0)
    Hmap=encode(model,tk_tr,mk_tr,rng.choice(len(at_tr),size=5000,replace=False))
    HH=[];DD=[]
    for gi,H in Hmap.items():
        a=at_tr[gi]; v=np.where(a>=0)[0]; HH.append(H[v]); DD.append(a[v])
    Htr=np.concatenate(HH).astype(np.float32); dtr=np.concatenate(DD).astype(int)
    clf=nn.Linear(Htr.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
    Xt=torch.from_numpy(Htr).to(DEV); yt=torch.from_numpy(dtr).long().to(DEV); lf=nn.CrossEntropyLoss()
    for ep in range(60):
        perm=torch.randperm(len(yt),device=DEV)
        for s in range(0,len(yt),8192):
            b=perm[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
    W=clf.weight.detach().cpu().numpy(); B=clf.bias.detach().cpu().numpy()

    # test info
    samp={}
    for gi in range(len(at)):
        a=at[gi]; v=np.where(a>=0)[0]
        if len(v)==0: continue
        lh=tk[gi][:,2]; info={}
        for d in np.unique(a[v]): info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
        order=sorted(info,key=lambda d:-info[d]["h"])
        for r,d in enumerate(order): info[d]["rank"]=r
        samp[gi]={"info":info,"noc":len(order),"true":set(int(x) for x in np.unique(a[v]))}
    allids=np.array(list(samp.keys()))
    Hm=encode(model,tk,mk,allids); MS=mscores(model,tk,mk,allids); MSmap={g:MS[i] for i,g in enumerate(allids)}
    def softmax_rows(X):
        Z=X@W.T+B; Z-=Z.max(1,keepdims=True); E=np.exp(Z); return E/E.sum(1,keepdims=True)

    # [C] per-NOC model-decode vs H-vote oracle
    mo=defaultdict(list); ho=defaultdict(list)
    for g in allids:
        noc=samp[g]["noc"]; true=samp[g]["true"]
        mtop=set(np.argsort(MSmap[g])[::-1][:noc].tolist()); mo[noc].append(mtop==true)
        a=at[g]; v=np.where(a>=0)[0]; vote=softmax_rows(Hm[g][v]).sum(0)
        htop=set(np.argsort(vote)[::-1][:noc].tolist()); ho[noc].append(htop==true)

    # [D] faintest-minor recall by #private alleles present (model-decode), N5 only
    bucket=defaultdict(lambda:[0,0])  # npriv-bucket -> [hit,total]
    for g in [x for x in allids if samp[x]["noc"]==5]:
        d=[dd for dd in samp[g]["info"] if samp[g]["info"][dd]["rank"]==4][0]
        priv=private_of(g,d,samp[g]["info"],tk); a=at[g]; v=np.where(a>=0)[0]
        npriv=sum(1 for j in v if int(a[j])==d and (int(tk[g][j,0]),akey(tk[g][j,1])) in priv)
        b="1" if npriv<=1 else ("2" if npriv==2 else ("3-4" if npriv<=4 else "5+"))
        hit=d in set(np.argsort(MSmap[g])[::-1][:5].tolist())
        bucket[b][0]+=hit; bucket[b][1]+=1
    results[run.name]={"mo":mo,"ho":ho,"bucket":bucket}
    nca=json.load(open(run/'metrics.json'))['config'].get('nc_attn','none')
    print(f"\n=== {run.name} (nc_attn={nca}) ===")
    print("  [C] set-oracle EM per NOC  (model-decode / H-linear-vote)")
    for n in range(1,6):
        if n in mo: print(f"      N{n}:  model {np.mean(mo[n]):.3f}   H-vote {np.mean(ho[n]):.3f}   gap(Hvote-model) {np.mean(ho[n])-np.mean(mo[n]):+.3f}")
    print("  [D] faintest-minor (N5) recall by #private alleles present (model-decode):")
    for b in ["1","2","3-4","5+"]:
        if b in bucket and bucket[b][1]: print(f"      npriv={b:>3}:  recall {bucket[b][0]/bucket[b][1]:.2f}  (n={bucket[b][1]})")

# side-by-side D
print("\n=== [D] side-by-side faint-minor recall by #private (model-decode) ===")
names=list(results.keys())
print("  npriv   " + "  ".join(f"{n.split('_seed')[0].replace('inc2_2d_','').replace('inc11_',''):>14}" for n in names))
for b in ["1","2","3-4","5+"]:
    row=[]
    for n in names:
        bk=results[n]["bucket"].get(b)
        row.append(f"{bk[0]/bk[1]:.2f}(n={bk[1]})" if bk and bk[1] else "   -   ")
    print(f"  {b:>5}   " + "  ".join(f"{x:>14}" for x in row))

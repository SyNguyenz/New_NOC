"""
Inc11 mechanism probe — EVIDENCE for: (1) why `both` fails, (2) what `mab0` actually fixes,
(3) whether it is thorough, on the SAME checkpoints, with the SAME no-train encoder probe.

For each run we report, restricted to REAL TEST:
  [E1] encoder per-peak identity READABILITY   = linear-probe (fit on train H) accuracy on N5 private peaks
  [E2] EXPLAINING-AWAY / isolation failure      = of the faint minor's (rank-4) private peaks, where does
                                                  the encoder-H point among the 5 contributors?
                                                    ->minor  (isolated, good) / ->major (absorbed, the wall) / ->other-minor
  [N]  encoder output token L2 norm by NOC      = does removing softmax row-normalization (mab1 sigmoid in `both`)
                                                  blow up / destabilize the residual stream as #peaks grows?
The linear probe is REFIT per checkpoint on that checkpoint's own H (so it measures that encoder, not a shared basis).
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA = Path("data_insilico_w")
RUNS = [Path(p) for p in sys.argv[1:]] or [
    Path("results/inc2_2d_sparse_seed42"),
    Path("results/inc11_nc_mab0_seed42"),
    Path("results/inc11_nc_both_seed42"),
]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
geno = load_raw_genotypes()

def build(run):
    cfg = json.load(open(run/"metrics.json"))["config"]; n_tok = cfg["n_token_feats"]
    m = SetTransformerMixture(
        n_loci=cfg.get("n_loci",24), d_locus=cfg.get("d_locus",16), d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4), n_isab=cfg.get("n_isab",2), m_inducing=cfg.get("m_inducing",32),
        n_classes=45, n_noc=6, dropout=cfg.get("dropout",0.1),
        cls_decoder=cfg.get("cls_decoder","pooled"), decoder_source=cfg.get("decoder_source","encoded"),
        n_token_feats=n_tok, encoder=cfg.get("encoder","isab"), dec_layers=cfg.get("dec_layers",2),
        num_embed=cfg.get("num_embed","raw"), n_freq=cfg.get("n_freq",8), d_num_emb=cfg.get("d_num_emb",8),
        periodic_sigma=cfg.get("periodic_sigma",1.0), aux_heads=cfg.get("aux_heads",False),
        sparse_attn=cfg.get("sparse_attn",False), vib=cfg.get("vib",False),
        mass_pool=cfg.get("mass_pool",False), nc_attn=cfg.get("nc_attn","none"),
        nc_learnable_bias=cfg.get("nc_learnable_bias",False)).to(DEV)
    missing, unexpected = m.load_state_dict(
        torch.load(run/"best_model.pt", map_location=DEV, weights_only=True), strict=False)
    # guard: nc_attn arch mismatch would show up as many missing/unexpected keys
    bad = [k for k in list(missing)+list(unexpected) if "encoder" in k]
    if bad: print(f"  !! {run.name}: {len(bad)} encoder-key mismatches (arch wrong!) e.g. {bad[:3]}")
    m.eval(); return m, n_tok

def load(s, n_tok):
    return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
            np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")

@torch.no_grad()
def encode(model, tk, mk, idxs, bs=128):
    out={}; norms={}
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        _,H,_=model._encode_set(t,m)
        Hc=H.cpu().numpy()
        for j,gi in enumerate(sel):
            out[int(gi)]=Hc[j]
    return out

def private_of(g, d, info, at, tk):
    others=[KNOWN[o] for o in info if o!=d]; gX=geno.get(KNOWN[d],{}); priv=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: priv.add((L,a))
    return priv

print(f"device={DEV}")
for run in RUNS:
    model,n_tok = build(run)
    tk_tr,mk_tr,at_tr = load("train",n_tok)
    tk,mk,at = load("test",n_tok)
    rng=np.random.default_rng(0)

    # ---- fit linear identity probe on this checkpoint's TRAIN H ----
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
    with torch.no_grad():
        fit_acc=(clf(Xt).argmax(1)==yt).float().mean().item()
    W=clf.weight.detach().cpu().numpy(); B=clf.bias.detach().cpu().numpy()
    def probe_pred(Hrows, contribs):
        Z=Hrows@W.T+B; sc={c:Z[:,c] for c in contribs}
        return np.stack([sc[c] for c in contribs],1).argmax(1)  # index into contribs

    # ---- test sample info ----
    samp={}
    for gi in range(len(at)):
        a=at[gi]; v=np.where(a>=0)[0]
        if len(v)==0: continue
        lh=tk[gi][:,2]; info={}
        for d in np.unique(a[v]): info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
        order=sorted(info,key=lambda d:-info[d]["h"])
        for r,d in enumerate(order): info[d]["rank"]=r
        samp[gi]={"info":info,"noc":len(order)}

    # ---- [N] encoder token norm by NOC ----
    normbynoc=defaultdict(list)
    for noc in range(1,6):
        ids=np.array([g for g in samp if samp[g]["noc"]==noc][:300])
        if len(ids)==0: continue
        Hm=encode(model,tk,mk,ids)
        for g in ids:
            a=at[g]; v=np.where(a>=0)[0]
            normbynoc[noc].append(np.linalg.norm(Hm[g][v],axis=1).mean())

    # ---- [E1][E2] N5 faint-minor private-peak isolation ----
    keep=np.array([g for g in samp if samp[g]["noc"]==5])
    Hm=encode(model,tk,mk,keep)
    tom=tomaj=toother=npk=0
    for g in keep:
        d=[dd for dd in samp[g]["info"] if samp[g]["info"][dd]["rank"]==4][0]  # faintest minor
        contribs=list(samp[g]["info"].keys()); priv=private_of(g,d,samp[g]["info"],at,tk)
        a=at[g]; v=np.where(a>=0)[0]
        pk=[j for j in v if int(a[j])==d and (int(tk[g][j,0]),akey(tk[g][j,1])) in priv]
        if not pk: continue
        preds=probe_pred(Hm[g][pk], contribs)
        for pi in preds:
            pred=contribs[pi]; npk+=1
            if pred==d: tom+=1
            elif samp[g]["info"][pred]["rank"]<4: tomaj+=1
            else: toother+=1

    print(f"\n=== {run.name}  (nc_attn={json.load(open(run/'metrics.json'))['config'].get('nc_attn','none')}) ===")
    print(f"  [E1] encoder per-peak identity probe fit-acc (train)   : {fit_acc:.3f}")
    print(f"  [E2] faintest-minor PRIVATE peak destination (n={npk}) : ->minor {tom/npk:.2f}  ->MAJOR {tomaj/npk:.2f}  ->other-minor {toother/npk:.2f}")
    print(f"  [N]  mean ||H|| by NOC : " + "  ".join(f"N{n}={np.mean(normbynoc[n]):.2f}" for n in range(1,6) if n in normbynoc))
print("\nlegend: ->MAJOR high = explaining-away / isolation failure (the encoder wall). lower is better.")

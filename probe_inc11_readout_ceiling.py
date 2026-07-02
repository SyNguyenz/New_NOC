"""
Is the residual encoder-ceiling miss (.839 soft-vote < 1.0) an AGGREGATION/READOUT limit or a genuine
ENCODER-INFO limit? Decisive tests on the SAME mab0 H (probe refit on train, combo-disjoint, no GT privilege):

[CEIL] three readouts on the identical per-peak H:
         SOFT = sum_j softmax(H_j)          (the .839 number; SUMS over all peaks -> shared mass dilutes a faint minor)
         MAX  = max_j  softmax(H_j)          (anti-aggregation: ONE strong isolated peak suffices)
         MLP  = sum_j softmax(MLP(H_j))      (nonlinear readout; can the info be read better?)
       If MAX or MLP >> SOFT -> the info IS in H, the limit is the READOUT (how peaks are aggregated).
       If all ~.84 -> encoder-info-limited (no readout recovers it).

[DIAG] for faint minors MISSED by SOFT: is their private-peak isolation p_iso high (encoder DID isolate ->
       readout/aggregation drowned it) or low (encoder did NOT isolate -> genuine encoder failure)?
       And: does MAX RECOVER the missed-but-isolated ones? (MAX recovering them = aggregation, causally.)
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1]) if len(sys.argv)>1 else Path("results/inc11_nc_mab0_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; geno=load_raw_genotypes()
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),num_embed=cfg.get("num_embed","raw"),n_freq=cfg.get("n_freq",8),
    d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),aux_heads=cfg.get("aux_heads",False),
    sparse_attn=cfg.get("sparse_attn",False),nc_attn=cfg.get("nc_attn","none"),nc_learnable_bias=cfg.get("nc_learnable_bias",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()
def load(s): return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
                     np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")
@torch.no_grad()
def encode(tk,mk,idxs,bs=128):
    out={}
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; _,H,_=model._encode_set(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        for j,gi in enumerate(sel): out[int(gi)]=H[j].cpu().numpy()
    return out
def private_of(g,d,info,at,tk):
    others=[KNOWN[o] for o in info if o!=d]; gX=geno.get(KNOWN[d],{}); priv=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: priv.add((L,a))
    return priv

# fit linear + MLP probe on train H
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0)
Hmap=encode(tk_tr,mk_tr,rng.choice(len(at_tr),size=5000,replace=False)); HH=[];DD=[]
for gi,H in Hmap.items():
    a=at_tr[gi]; v=np.where(a>=0)[0]; HH.append(H[v]); DD.append(a[v])
Htr=np.concatenate(HH).astype(np.float32); dtr=np.concatenate(DD).astype(int)
Xt=torch.from_numpy(Htr).to(DEV); yt=torch.from_numpy(dtr).long().to(DEV)
lin=nn.Linear(Htr.shape[1],45).to(DEV)
mlp=nn.Sequential(nn.Linear(Htr.shape[1],256),nn.ReLU(),nn.Linear(256,45)).to(DEV)
for net in (lin,mlp):
    opt=torch.optim.Adam(net.parameters(),lr=1e-2,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
    for ep in range(60):
        perm=torch.randperm(len(yt),device=DEV)
        for s in range(0,len(yt),8192):
            b=perm[s:s+8192]; opt.zero_grad(); lf(net(Xt[b]),yt[b]).backward(); opt.step()
Wl=lin.weight.detach().cpu().numpy(); Bl=lin.bias.detach().cpu().numpy()
@torch.no_grad()
def mlp_prob(H):
    z=mlp(torch.from_numpy(H).to(DEV)); return torch.softmax(z,1).cpu().numpy()
def lin_prob(H):
    Z=H@Wl.T+Bl; Z-=Z.max(1,keepdims=True); E=np.exp(Z); return E/E.sum(1,keepdims=True)

tk,mk,at=load("test"); samp={}
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: continue
    lh=tk[gi][:,2]; info={}
    for d in np.unique(a[v]): info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
    order=sorted(info,key=lambda d:-info[d]["h"])
    for r,d in enumerate(order): info[d]["rank"]=r
    if len(order)==5: samp[gi]={"info":info,"true":set(int(x) for x in np.unique(a[v]))}
ids=np.array(list(samp.keys())); Hm=encode(tk,mk,ids)

soft_em=max_em=mlp_em=0; n=len(ids)
miss_iso=[]; miss_recov_max=0; miss_iso_n=0; miss_total=0
for g in ids:
    a=at[g]; v=np.where(a>=0)[0]; true=samp[g]["true"]; info=samp[g]["info"]
    sm=lin_prob(Hm[g][v]); mp=mlp_prob(Hm[g][v])
    soft=sm.sum(0); mx=sm.max(0); mlpv=mp.sum(0)
    st=set(np.argsort(soft)[::-1][:5]); mt=set(np.argsort(mx)[::-1][:5]); lt=set(np.argsort(mlpv)[::-1][:5])
    soft_em+=(st==true); max_em+=(mt==true); mlp_em+=(lt==true)
    d=[dd for dd in info if info[dd]["rank"]==4][0]
    if d not in st:  # SOFT missed the faint minor
        miss_total+=1
        priv=private_of(g,d,info,at,tk); vpos={int(p):r for r,p in enumerate(v)}
        pk=[int(p) for p in v if int(a[p])==d and (int(tk[g][p,0]),akey(tk[g][p,1])) in priv]
        if pk:
            contribs=list(info.keys())
            piso=np.mean([contribs[int(np.argmax([sm[vpos[p]][c] for c in contribs]))]==d for p in pk])
            miss_iso.append(piso)
            if piso>=0.5:
                miss_iso_n+=1; miss_recov_max+=(d in mt)
print(f"=== {RUN.name}: readout ceiling on the SAME mab0 H (N5 n={n}) ===")
print(f"  [CEIL] set-EM   SOFT(sum) {soft_em/n:.3f}   MAX {max_em/n:.3f}   MLP {mlp_em/n:.3f}")
print(f"         MAX-SOFT = {max_em/n-soft_em/n:+.3f}   MLP-SOFT = {mlp_em/n-soft_em/n:+.3f}")
print(f"  [DIAG] faint minors MISSED by SOFT: {miss_total}")
print(f"         mean private-peak isolation p_iso of missed = {np.mean(miss_iso):.2f}  ( >0.5 => encoder DID isolate them )")
print(f"         of missed-but-ISOLATED (p_iso>=.5, n={miss_iso_n}): recovered by MAX readout = {miss_recov_max}/{miss_iso_n} = {miss_recov_max/max(miss_iso_n,1):.0%}")
print("\n  MAX/MLP >> SOFT  AND  missed are isolated  AND  MAX recovers them  => AGGREGATION/READOUT limit (claim supported).")
print("  all readouts ~equal AND missed have LOW p_iso                      => genuine ENCODER-INFO limit (claim refuted).")

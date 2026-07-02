"""
WHY does the base encoder miss ~.21 at N5? Decompose the isolation failure (extends
probe_encoder_isolate.py: same clean train-fit per-peak probe, same private-allele-peak read,
restricted to the sample's 5 contributors) along three decision-relevant axes:

  (A) by CONTRIBUTOR RANK (0=major .. 4=faintest): isolation rate (->self) of each rank's PRIVATE peaks.
      -> is absorption only the faintest, or graded up the ranks?
  (B) #CONTRIBUTORS not-isolated per N5 sample (a contributor is "lost" if <50% of its private peaks
      point to self): distribution 0/1/2/3+.  -> is it "fix the 1 faintest" or "fix many"?
  (C) absorbed (->major) private peaks: is the locus CROWDED (a STRONGER contributor has a peak at the
      same locus in this sample = physical masking, near-floor) or CLEAN (no stronger peak there =
      pure context-wash the encoder has no excuse for = clearly fixable)?

Run on BASE by default (the thing we want to fix). eval-only, frozen ckpt.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc2_2d_sparse_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
geno=load_raw_genotypes()
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
    vib=cfg.get("vib",False),mass_pool=cfg.get("mass_pool",False),
    attn_sink=int(cfg.get("attn_sink",0) or 0),donor_recon=cfg.get("donor_recon",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()
print(f"loaded {RUN.name}")

def load(s):
    return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
            np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")

@torch.no_grad()
def encode_full(tk,mk,idxs,bs=128):
    out={}
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        _,H,_=model._encode_set(t,m); H=H.cpu().numpy()
        for j,gi in enumerate(sel): out[int(gi)]=H[j]
    return out

# fit clean per-peak probe on train H (identical to F33)
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0)
HH=[];DD=[]
for gi,H in encode_full(tk_tr,mk_tr,rng.choice(len(at_tr),size=6000,replace=False)).items():
    a=at_tr[gi]; v=np.where(a>=0)[0]; HH.append(H[v]); DD.append(a[v])
Htr=np.concatenate(HH).astype(np.float32); dtr=np.concatenate(DD).astype(int)
clf=nn.Linear(Htr.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
Xt=torch.from_numpy(Htr).to(DEV); yt=torch.from_numpy(dtr).long().to(DEV); lf=nn.CrossEntropyLoss()
for ep in range(60):
    perm=torch.randperm(len(yt),device=DEV)
    for s in range(0,len(yt),8192):
        b=perm[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
W=clf.weight.detach().cpu().numpy(); B=clf.bias.detach().cpu().numpy()
def softmax_rows(X):
    Z=X@W.T+B; Z-=Z.max(1,keepdims=True); E=np.exp(Z); return E/E.sum(1,keepdims=True)
print("probe fit")

# test N5 setup
tk,mk,at=load("test")
samp={}
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: continue
    lh=tk[gi][:,2]; info={}
    for d in np.unique(a[v]):
        info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
    order=sorted(info,key=lambda d:-info[d]["h"])
    for r,d in enumerate(order): info[d]["rank"]=r
    samp[gi]={"info":info,"noc":len(order)}
keep=[g for g in samp if samp[g]["noc"]==5]
Hmap=encode_full(tk,mk,np.array(keep))

def private_of(g,d):
    others=[KNOWN[o] for o in samp[g]["info"] if o!=d]; gX=geno.get(KNOWN[d],{}); priv=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: priv.add((L,a))
    return priv

# ---- collect per-(rank) isolation + crowded/clean split + per-sample lost count ----
rank_self=np.zeros(5); rank_maj=np.zeros(5); rank_oth=np.zeros(5); rank_n=np.zeros(5)
crowd=defaultdict(int)   # ("crowded"/"clean") among ABSORBED ->major faint peaks
crowd_self=defaultdict(int)  # CONTROL: same, among SELF-isolated faint peaks (base rate)
absorber_here=[0,0]      # [absorber-present-at-locus, absorber-absent] among absorbed faint peaks
lost_hist=defaultdict(int)
hbin_self=defaultdict(lambda:[0,0])   # height-ratio bin -> [self, total]

for g in keep:
    a=at[g]; v=np.where(a>=0)[0]
    info=samp[g]["info"]; contribs=list(info.keys()); tot_h=sum(info[c]["h"] for c in contribs)
    # locus -> ranks of contributors with a peak there (from attr)
    loc_ranks=defaultdict(set)
    for j in v: loc_ranks[int(tk[g][j,0])].add(info[int(a[j])]["rank"])
    lost=0
    for d in contribs:
        r=info[d]["rank"]; priv=private_of(g,d)
        pk=[j for j in v if int(a[j])==d and (int(tk[g][j,0]),akey(tk[g][j,1])) in priv]
        if not pk: continue
        sm=softmax_rows(Hmap[g][pk])
        nself=0
        for row,j in zip(sm,pk):
            sc={c:row[c] for c in contribs}; pred=max(sc,key=sc.get)
            rank_n[r]+=1
            L=int(tk[g][j,0]); stronger_here=any(rr<r for rr in loc_ranks[L])
            if pred==d:
                rank_self[r]+=1; nself+=1
                if r>=3: crowd_self[("crowded" if stronger_here else "clean")]+=1   # base-rate control
            elif info[pred]["rank"]<r:
                rank_maj[r]+=1
                if r>=3:
                    crowd[("crowded" if stronger_here else "clean")]+=1
                    absorber_here[0 if info[pred]["rank"] in loc_ranks[L] else 1]+=1  # absorber at THIS locus?
            else:
                rank_oth[r]+=1
        hr=info[d]["h"]/tot_h; bk=min(int(hr/0.05),5)   # 0-5% .. >25%
        hbin_self[bk][0]+=nself; hbin_self[bk][1]+=len(pk)
        if nself/len(pk) < 0.5: lost+=1
    lost_hist[lost]+=1

print(f"\nN5 samples analysed: {len(keep)}")
print("\n(A) PRIVATE-peak isolation by contributor RANK (0=strongest major .. 4=faintest minor)")
print("   rank   n_privpeaks   ->self   ->major   ->other-minor")
for r in range(5):
    n=rank_n[r]
    if n: print(f"    {r}      {int(n):8d}     {rank_self[r]/n:.2f}     {rank_maj[r]/n:.2f}      {rank_oth[r]/n:.2f}")

print("\n(A') isolation vs the contributor's HEIGHT-RATIO bin")
print("   ratio-bin     ->self    n")
labs={0:"0-5%",1:"5-10%",2:"10-15%",3:"15-20%",4:"20-25%",5:">25%"}
for bk in sorted(hbin_self):
    s,t=hbin_self[bk]
    if t: print(f"    {labs[bk]:8s}     {s/t:.2f}    {t}")

print("\n(B) # contributors LOST (<50% private peaks self-isolated) per N5 sample")
tot=sum(lost_hist.values())
for k in sorted(lost_hist):
    print(f"    {k} lost : {lost_hist[k]:4d} samples ({lost_hist[k]/tot:.1%})")

print("\n(C) faint (rank>=3) private peaks: locus-CROWDED rate, ABSORBED vs SELF-isolated (control)")
cc=crowd["crowded"]; cl=crowd["clean"]; tt=cc+cl
sc_c=crowd_self["crowded"]; sc_l=crowd_self["clean"]; st=sc_c+sc_l
if tt:  print(f"    ABSORBED->major : crowded {cc/tt:.1%}  (n={tt})")
if st:  print(f"    SELF-isolated   : crowded {sc_c/st:.1%}  (n={st})   <- base rate")
print("    => if ABSORBED crowded-rate ~ base rate, crowding is NOT the discriminator (it's near-universal at N5).")
ah=absorber_here[0]+absorber_here[1]
if ah: print(f"    of absorbed faint peaks, the ABSORBING major has a peak at THE SAME locus: {absorber_here[0]/ah:.1%} (n={ah})")

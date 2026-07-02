"""
Does a HEIGHT/likelihood signal COMPLEMENT (not replace) the rarity ensemble on the floor donors?
Floor = faint donors whose alleles are ALL panel-common (no decisive allele) -> rarity-match (binary
presence) can't disambiguate; the only remaining signal is QUANTITATIVE height (which donors' genotypes,
weighted, reconstruct the observed peak HEIGHTS = NNLS mixture deconvolution, EuroForMix-spirit).

Standalone NNLS is known worse than neural (F33 .55<.73) -- but so was rarity-match standalone (.79<.94),
yet it was complementary (+.11). Test the SAME for height: ensemble neural+rarity+NNLS, and recovery
specifically on the panel-common FLOOR donors.

  +NNLS recovers floor donors / lifts set-EM above .83 => height is complementary => .84 floor BREAKS.
  no recovery on floor                                  => .84 is a real information floor.
"""
import json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from scipy.optimize import nnls
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w"); RUN="results/inc2_2d_sparse_seed42"; DEV="cuda" if torch.cuda.is_available() else "cpu"
geno=load_raw_genotypes()
cfg=json.load(open(Path(RUN)/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
m=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","per_donor"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
    attn_sink=int(cfg.get("attn_sink",0) or 0),donor_recon=cfg.get("donor_recon",False)).to(DEV)
m.load_state_dict(torch.load(Path(RUN)/"best_model.pt",map_location=DEV,weights_only=True),strict=False); m.eval()

tk=np.load(DATA/"tokens8_test.npy")[:,:,:n_tok].astype(np.float32); mk=np.load(DATA/"mask_test.npy").astype(bool)
at=np.load(DATA/"attr_test.npy"); y=np.load(DATA/"y_test_set.npy"); noc=np.clip(np.load(DATA/"noc_test.npy").astype(int),1,5)
# standardization stats to invert log_h -> RFU. tokens log_h is standardized? check range: log_h was raw (0.69..10). Use exp.
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")
panel=defaultdict(int)
for d,loc in geno.items():
    for L,al in loc.items():
        for a in al: panel[(L,a)]+=1
ref={c:set((L,a) for L,al in geno.get(KNOWN[c],{}).items() for a in al) for c in range(45)}

idxN5=[gi for gi in range(len(at)) if noc[gi]==5 and len(np.unique(at[gi][at[gi]>=0]))==5]
@torch.no_grad()
def neural(idx,bs=128):
    P=np.zeros((len(idx),45),np.float32)
    for s in range(0,len(idx),bs):
        sel=idx[s:s+bs]; o=m(torch.from_numpy(tk[sel]).to(DEV),torch.from_numpy(mk[sel]).to(DEV))
        P[s:s+len(sel)]=torch.sigmoid(o["logits_cls"]).cpu().numpy()
    return P
NS=neural(np.array(idxN5))

def scores(gi):
    v=np.where(at[gi]>=0)[0]
    keys=[(int(tk[gi][j,0]),akey(tk[gi][j,1])) for j in v]
    h=np.array([float(np.exp(tk[gi][j,2])) for j in v])              # RFU per peak
    mc=np.zeros(45); mr=np.zeros(45)
    G=np.zeros((len(v),45))
    for c in range(45):
        rc=ref[c]
        for r,k in enumerate(keys):
            if k in rc:
                G[r,c]=1.0; mc[c]+=1; mr[c]+=1.0/panel[k]
    w,_=nnls(G,h)                                                    # height deconvolution -> per-donor contribution
    return mc,mr,w,keys
def z(s): s=s.astype(float); sd=s.std(); return (s-s.mean())/(sd+1e-8)
def setem(score,gi): return set(np.argsort(score)[::-1][:5].tolist())==set(np.where(y[gi]==1)[0].tolist())

MR=[];WT=[];MC=[];KEYS=[]
for gi in idxN5:
    mc,mr,w,keys=scores(gi); MC.append(mc); MR.append(mr); WT.append(w); KEYS.append(set(keys))

def em_of(combine):
    return np.mean([setem(combine(i,gi),gi) for i,gi in enumerate(idxN5)])
print(f"N5 n={len(idxN5)}\n")
print("standalone set-EM:")
print(f"  neural   {em_of(lambda i,gi:NS[i]):.3f}")
print(f"  rarity   {em_of(lambda i,gi:MR[i]):.3f}")
print(f"  NNLS-ht  {em_of(lambda i,gi:WT[i]):.3f}   (height deconvolution alone)")
print("\nensemble set-EM:")
print(f"  neural+rarity        {em_of(lambda i,gi:z(NS[i])+2*z(MR[i])):.3f}   (= current .83 baseline)")
print(f"  neural+rarity+NNLS   {em_of(lambda i,gi:z(NS[i])+2*z(MR[i])+2*z(WT[i])):.3f}")
best=0;bl=None
for a in [1,2,3]:
    for b in [0.5,1,2,3]:
        e=em_of(lambda i,gi:z(NS[i])+a*z(MR[i])+b*z(WT[i]))
        if e>best: best=e;bl=(a,b)
print(f"  best neural+a*rarity+b*NNLS = {best:.3f} at (a,b)={bl}")

# floor-specific recovery: faint donor all-panel-common, missed by neural+rarity -> does +NNLS recover?
print("\nFLOOR donors (faint all-panel-common, missed by neural+rarity): does +NNLS recover them?")
tot=rec=0
for i,gi in enumerate(idxN5):
    v=np.where(at[gi]>=0)[0]; a=at[gi]
    lh=tk[gi][:,2]; hsum={int(d):float(np.exp(lh[a==d]).sum()) for d in np.unique(a[v])}
    faint=min(hsum,key=hsum.get)
    base=z(NS[i])+2*z(MR[i])
    if faint in set(np.argsort(base)[::-1][:5].tolist()): continue        # not a miss
    # panel-common floor? faint has no panel-rare(<=2) allele present
    present=[k for k in KEYS[i] if k in ref[faint]]
    if any(panel[k]<=2 for k in present): continue                        # has decisive allele = not floor
    tot+=1
    full=z(NS[i])+2*z(MR[i])+2*z(WT[i])
    rec+=(faint in set(np.argsort(full)[::-1][:5].tolist()))
print(f"  floor faint-donors: {tot} | recovered by +NNLS-height: {rec} ({rec/max(tot,1):.2f})")
print("\n  recovery>0 & set-EM up => height COMPLEMENTS => .84 floor breaks. else => real floor.")

"""
Two questions before building the decisive-allele-weighting lever:

Q1 (does it solve it THOROUGHLY?): sweep the neural+rarity ensemble lambda for N5 set-EM, and compute an
   ORACLE-UNION upper bound (per sample, correct if neural OR M_count OR M_rarity is right) = the ceiling of
   the whole reference-matching direction. Then characterize the RESIDUAL misses: do the still-missed true
   donors have a PANEL-RARE allele present (recoverable headroom for a better scorer) or are all their
   alleles panel-common (a genuine ambiguity FLOOR)?

Q2 (why fix the DECODER if the problem is the ENCODER?): compare the gain to the ENCODER's own readout
   ceiling (soft-vote svmax on H, mixture-only). If ensemble > encoder-soft-vote-ceiling, the gain comes
   from EXTERNAL reference-panel info the encoder never had -> it is NOT an encoder-isolation failure but a
   reference-matching gap, which lives at the donor-scoring (decoder) stage.
"""
import json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
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
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")
panel=defaultdict(int)
for d,loc in geno.items():
    for L,al in loc.items():
        for a in al: panel[(L,a)]+=1
ref={c:[(L,a) for L,al in geno.get(KNOWN[c],{}).items() for a in al] for c in range(45)}

idxN5=[gi for gi in range(len(at)) if noc[gi]==5 and len(np.unique(at[gi][at[gi]>=0]))==5]
@torch.no_grad()
def run(idx,bs=128):
    NS=np.zeros((len(idx),45),np.float32); SV=np.zeros((len(idx),45),np.float32)
    for s in range(0,len(idx),bs):
        sel=idx[s:s+bs]; T=torch.from_numpy(tk[sel]).to(DEV); M=torch.from_numpy(mk[sel]).to(DEV); o=m(T,M)
        NS[s:s+len(sel)]=torch.sigmoid(o["logits_cls"]).cpu().numpy()
        a=F.softmax(o["logits_attr"],dim=-1)[:,:,:45]*M.unsqueeze(-1); SV[s:s+len(sel)]=a.max(1).values.cpu().numpy()
    return NS,SV
NS,SV=run(np.array(idxN5))

def mscore(gi):
    obs=set((int(tk[gi][j,0]),akey(tk[gi][j,1])) for j in np.where(at[gi]>=0)[0])
    mc=np.zeros(45); mr=np.zeros(45)
    for c in range(45):
        for k in ref[c]:
            if k in obs: mc[c]+=1; mr[c]+=1.0/panel[k]
    return mc,mr,obs
def z(s): s=s.astype(float); return (s-s.mean())/(s.std()+1e-8)
def setem(score,gi): return set(np.argsort(score)[::-1][:5].tolist())==set(np.where(y[gi]==1)[0].tolist())

MC=[]; MR=[]; OBS=[]
for i,gi in enumerate(idxN5):
    mc,mr,obs=mscore(gi); MC.append(mc); MR.append(mr); OBS.append(obs)

print(f"N5 n={len(idxN5)}\n")
print("Q2: gain source vs the ENCODER's own readout ceiling")
print(f"  neural set-EM           : {np.mean([setem(NS[i],gi) for i,gi in enumerate(idxN5)]):.3f}")
print(f"  encoder soft-vote (svmax): {np.mean([setem(SV[i],gi) for i,gi in enumerate(idxN5)]):.3f}   (mixture-only ceiling)")
print(f"  neural + rarity-match    : {np.mean([setem(z(NS[i])+2*z(MR[i]),gi) for i,gi in enumerate(idxN5)]):.3f}   (adds reference panel)")
print("  -> ensemble ABOVE encoder soft-vote ceiling => gain = EXTERNAL reference info, not encoder isolation.\n")

print("Q1: ceiling of the reference-matching direction")
for lam in [0,1,2,3,5,10]:
    em=np.mean([setem(z(NS[i])+lam*z(MR[i]),gi) for i,gi in enumerate(idxN5)])
    print(f"  ens+rarity lambda={lam:<2}: set-EM {em:.3f}")
# oracle-union upper bound: correct if ANY of neural / M_count / M_rarity is right
oru=np.mean([max(setem(NS[i],gi),setem(MC[i],gi),setem(MR[i],gi)) for i,gi in enumerate(idxN5)])
print(f"  ORACLE-UNION(neural,M_count,M_rarity): {oru:.3f}   (upper bound of this direction)")

# residual: of sets missed by oracle-union, do the missing true donors have a PANEL-RARE allele present?
print("\nResidual (sets MISSED even by oracle-union): is there recoverable decisive evidence, or a FLOOR?")
floor=head=0
for i,gi in enumerate(idxN5):
    if max(setem(NS[i],gi),setem(MC[i],gi),setem(MR[i],gi)): continue
    obs=OBS[i]; true=set(np.where(y[gi]==1)[0].tolist())
    top=set(np.argsort(z(NS[i])+2*z(MR[i]))[::-1][:5].tolist()); missing=true-top
    for d in missing:
        rare_present=sum(1 for k in ref[d] if k in obs and panel[k]<=2)
        if rare_present>=1: head+=1
        else: floor+=1
tot=head+floor
print(f"  missing true donors: {tot} | have a panel-rare(<=2) allele present (HEADROOM): {head} ({head/max(tot,1):.2f}) "
      f"| all alleles panel-common (FLOOR): {floor} ({floor/max(tot,1):.2f})")

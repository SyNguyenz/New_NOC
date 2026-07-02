"""
VERIFY (causal, decoder-level) the probe4 claim "few PRIVATE alleles -> faint donor missed", controlling
for the soft-vote-readout tautology and the #peaks/height confound.

Uses the ACTUAL DECODER (sigmoid logits top-5), and a WITHIN-DONOR dose-response: take faint donors that
ARE recalled at baseline and have >=5 private-allele peaks, then mask peaks and re-decode:
  KEEP-r-private : mask all but r of the donor's PRIVATE peaks (keep ALL shared)  -> recall vs r
  MASK-m-shared  : keep ALL private, mask m of the donor's SHARED peaks            -> control

If recall collapses as private peaks are removed (KEEP-r curve steep) but is robust to removing the SAME
number of SHARED peaks => PRIVATE alleles CAUSALLY drive recall = mechanism CONFIRMED at the model/decoder
level (not a soft-vote artifact, not just '#peaks'). If KEEP-r stays high or MASK-shared also drops => the
private-allele story is NOT the cause.
"""
import json, numpy as np, torch
from pathlib import Path
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

import sys
DATA=Path("data_insilico_w"); RUN=sys.argv[1] if len(sys.argv)>1 else "results/inc2_2d_sparse_seed42"; DEV="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0); geno=load_raw_genotypes()
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
@torch.no_grad()
def decode(mkrow):
    o=m(torch.from_numpy(tk[GI][None]).to(DEV),torch.from_numpy(mkrow[None]).to(DEV))
    return torch.sigmoid(o["logits_cls"])[0].cpu().numpy()
def recalled(mkrow,d,k=5):
    return d in set(np.argsort(decode(mkrow))[::-1][:k])

def private_of(gi,d,contribs):
    others=[KNOWN[o] for o in contribs if o!=d]; gX=geno.get(KNOWN[d],{}); pr=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: pr.add((L,a))
    return pr

# collect recalled faint donors with >=5 private peaks
cases=[]
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if noc[gi]!=5 or len(np.unique(a[v]))!=5: continue
    lh=tk[gi][:,2]; hsum={int(d):float(np.exp(lh[a==d]).sum()) for d in np.unique(a[v])}
    faint=min(hsum,key=hsum.get); contribs=list(hsum)
    pr=private_of(gi,faint,contribs)
    pk=[j for j in v if int(a[j])==faint]
    ppk=[j for j in pk if (int(tk[gi][j,0]),akey(tk[gi][j,1])) in pr]
    spk=[j for j in pk if j not in ppk]
    cases.append((gi,faint,ppk,spk))

print("collecting baseline-recalled faint donors with >=5 private peaks ...")
rng=np.random.default_rng(0)
sel=[]
for gi,faint,ppk,spk in cases:
    GI=gi
    if len(ppk)>=5 and recalled(mk[gi],faint): sel.append((gi,faint,ppk,spk))
print(f"  n usable = {len(sel)} (baseline recall = 1.0 by construction)\n")

def sweep_keep_private(r):
    hit=tot=0
    for gi,faint,ppk,spk in sel:
        global GI; GI=gi
        drop=ppk if r==0 else list(rng.choice(ppk,len(ppk)-r,replace=False)) if len(ppk)>r else []
        mkr=mk[gi].copy(); mkr[drop]=False
        hit+=recalled(mkr,faint); tot+=1
    return hit/tot
def mask_shared(mm):
    hit=tot=0; used=0
    for gi,faint,ppk,spk in sel:
        if len(spk)<mm: continue
        global GI; GI=gi
        drop=list(rng.choice(spk,mm,replace=False)); mkr=mk[gi].copy(); mkr[drop]=False
        hit+=recalled(mkr,faint); tot+=1
    return (hit/tot if tot else float('nan')), tot

print("CAUSAL dose-response (within-donor; baseline all-recalled=1.00):")
print("  keep r PRIVATE peaks (mask the rest; keep ALL shared):")
for r in [0,1,2,3,4]:
    print(f"    r={r}: recall {sweep_keep_private(r):.3f}")
print("  CONTROL mask m SHARED peaks (keep ALL private):")
for mm in [2,4,6]:
    rec,tn=mask_shared(mm); print(f"    m={mm}: recall {rec:.3f}  (n={tn})")
print("\n  steep drop as private removed + shared-mask robust => PRIVATE alleles CAUSALLY drive recall.")
print("  flat private-drop OR shared-mask also drops => private-allele mechanism NOT confirmed.")

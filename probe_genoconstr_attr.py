"""
Headroom of lever A (genotype-constrained attribution), NO TRAIN.
Reload inc13_B, get full attr LOGITS, then MASK each peak's owner softmax to donors that actually
CARRY that allele (+background). Re-argmax. Measure: does the faint minor's PRIVATE peak now go to T?
(unmasked baseline: private->T 30%, shared->T 1%). Also: total peaks credited to missed-T.
"""
import os, json
from pathlib import Path
import numpy as np, torch
from models.set_transformer import SetTransformerMixture
DA=Path("data_insilico_w"); RUN=Path("results/inc13_B_distill_seed42"); G="data/donor_geno.npy"
DEVc=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def ab(a): return int(round(float(a)*10))
def kk(l,a): return (int(round(float(l))),ab(a))
g=np.load(G); gm=np.load(G.replace(".npy","_mask.npy")).astype(bool); C=g.shape[0]
dset=[set(kk(g[c,j,0],g[c,j,1]) for j in range(g.shape[1]) if gm[c,j]) for c in range(C)]
carriers={}
for c in range(C):
    for it in dset[c]: carriers.setdefault(it,[]).append(c)

cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats",8)
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
    dropout=0.1,cls_decoder="per_donor",decoder_source="encoded",n_token_feats=n_tok,encoder="isab++",dec_layers=2,
    num_embed="periodic",n_freq=8,d_num_emb=8,periodic_sigma=0.3,aux_heads=True,sparse_attn=True).to(DEVc)
sd=torch.load(RUN/"best_model.pt",map_location=DEVc); sd=sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd
m.load_state_dict(sd,strict=False); m.eval()
tk=np.load(DA/f"tokens{n_tok}_test.npy").astype(np.float32); mk=np.load(DA/"mask_test.npy")
y_p=np.load(RUN/"y_test_pred.npy").astype(bool); y_t=np.load(DA/"y_test_set.npy").astype(bool); noc=np.load(DA/"noc_test.npy")

@torch.no_grad()
def attr_logits(t,k):
    out=[]
    for i in range(0,len(t),128):
        o=m(torch.from_numpy(t[i:i+128]).to(DEVc),torch.from_numpy(k[i:i+128].astype(bool)).to(DEVc))
        out.append(o["logits_attr"].cpu().numpy())
    return np.concatenate(out)
sel=np.where(noc==5)[0]
AL=attr_logits(tk[sel],mk[sel])      # (n,N,C+1)

unm={"sh_T":0,"sh_n":0,"pr_T":0,"pr_n":0}; msk={"sh_T":0,"pr_T":0}
credT_unm=[]; credT_msk=[]
for s_i,i in enumerate(sel):
    pidx={}
    for k in np.where(mk[i])[0]:
        it=kk(tk[i,k,0],tk[i,k,1])
        if it not in pidx or tk[i,k,2]>tk[i,pidx[it],2]: pidx[it]=k
    true=np.where(y_t[i])[0]; lg=AL[s_i]
    # masked argmax for ALL valid peaks
    masked_owner={}
    for it,k in pidx.items():
        valid=carriers.get(it,[])+[C]
        v=np.full(C+1,-1e9); v[valid]=lg[k,valid]; masked_owner[k]=int(np.argmax(v))
    for T in [c for c in true if not y_p[i,c]]:
        cu=cm=0
        for it in dset[T]&set(pidx):
            k=pidx[it]; sharers=[c for c in true if it in dset[c]]
            ow_u=int(np.argmax(lg[k])); ow_m=masked_owner[k]
            if ow_u==T: cu+=1
            if ow_m==T: cm+=1
            if len(sharers)>=2:
                unm["sh_n"]+=1; unm["sh_T"]+=int(ow_u==T); msk["sh_T"]+=int(ow_m==T)
            else:
                unm["pr_n"]+=1; unm["pr_T"]+=int(ow_u==T); msk["pr_T"]+=int(ow_m==T)
        credT_unm.append(cu); credT_msk.append(cm)
print("=== lever A headroom: genotype-masked attribution on N5 missed-minor peaks ===")
print(f"  PRIVATE peaks -> T :  unmasked {unm['pr_T']}/{unm['pr_n']}={unm['pr_T']/max(1,unm['pr_n']):.2f}"
      f"   MASKED {msk['pr_T']}/{unm['pr_n']}={msk['pr_T']/max(1,unm['pr_n']):.2f}")
print(f"  SHARED  peaks -> T :  unmasked {unm['sh_T']}/{unm['sh_n']}={unm['sh_T']/max(1,unm['sh_n']):.2f}"
      f"   MASKED {msk['sh_T']}/{unm['sh_n']}={msk['sh_T']/max(1,unm['sh_n']):.2f}")
print(f"  avg #peaks credited to a missed-minor:  unmasked {np.mean(credT_unm):.2f}   MASKED {np.mean(credT_msk):.2f}")
print(f"\n  (mask = a peak can only be owned by a donor CARRYING that allele, from the panel, +background)")

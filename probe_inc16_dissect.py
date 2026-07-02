"""
Dissect WHY add_recon (inc16) regressed N5. Hypothesis: the additive height-recon loss
(Σ_donor sigmoid(logit)·φ·owner ≈ observed heights) rewards donors that carry MUCH height
(majors) and starves the FAINT minor (tiny height contribution) -> suppresses exactly the
faint-minor logit that N5 needs.  Compare per-donor logits base(inc6_maskp) vs inc16 on N5,
grouped by faintness rank (phi) and decoy.
"""
import os, json
from pathlib import Path
import numpy as np, torch
from models.set_transformer import SetTransformerMixture

DATA=Path(os.environ.get("STR_DATA_DIR","data_insilico_w")); DEVc=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load(sp): return (np.load(DATA/f"tokens8_{sp}.npy").astype(np.float32),np.load(DATA/f"mask_{sp}.npy"),
                      np.load(DATA/f"y_{sp}_set.npy").astype(bool),np.load(DATA/f"noc_{sp}.npy").astype(int))
def build(run):
    cfg=json.load(open(Path(run)/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats",8)
    m=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
        dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","per_donor"),decoder_source=cfg.get("decoder_source","encoded"),
        n_token_feats=n_tok,encoder=cfg.get("encoder","isab++"),dec_layers=cfg.get("dec_layers",2),
        num_embed=cfg.get("num_embed","periodic"),n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),
        periodic_sigma=cfg.get("periodic_sigma",0.3),aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False)).to(DEVc)
    sd=torch.load(Path(run)/"best_model.pt",map_location=DEVc); sd=sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd
    m.load_state_dict(sd,strict=False); m.eval(); return m
@torch.no_grad()
def logits(m,tok,msk):
    L=[]
    for i in range(0,len(tok),256):
        o=m(torch.from_numpy(tok[i:i+256]).to(DEVc),torch.from_numpy(msk[i:i+256].astype(bool)).to(DEVc))
        L.append(o["logits_cls"].cpu().numpy())
    return np.concatenate(L)

tk,mk,y,noc=load("test"); phi=np.load(DATA/"phi_test.npy")
B=build("results/inc6_maskp_seed42"); I=build("results/inc16_addrecon_seed42")
Lb=logits(B,tk,mk); Li=logits(I,tk,mk)

def oracle(L,k):
    sel=np.where(noc==k)[0]; h=0
    for i in sel:
        top=np.argsort(L[i])[::-1][:k]; pr=np.zeros(45,int); pr[top]=1; h+=(pr==y[i]).all()
    return h/len(sel)
print("per-NOC oracle:  NOC " + "  ".join(f"{k}[base{oracle(Lb,k):.3f}/inc16{oracle(Li,k):.3f}]" for k in [3,4,5]))

# ── N5: logit by faintness rank (phi) + decoy, for BOTH models ──
sel5=np.where(noc==5)[0]
rk_b=[[] for _ in range(5)]; rk_i=[[] for _ in range(5)]
dec_b=[]; dec_i=[]; abs_b=[]; abs_i=[]
miss_b=miss_i=0; missrank_b=[0]*5; missrank_i=[0]*5
for s in sel5:
    pres=np.where(y[s])[0]; order=pres[np.argsort(phi[s,pres])]   # faintest..strongest
    for r,c in enumerate(order):
        rk_b[r].append(Lb[s,c]); rk_i[r].append(Li[s,c])
    absent=np.where(~y[s])[0]
    dec_b.append(Lb[s,absent].max()); dec_i.append(Li[s,absent].max())
    abs_b.append(Lb[s,absent].mean()); abs_i.append(Li[s,absent].mean())
    tb=set(np.argsort(Lb[s])[::-1][:5]); ti=set(np.argsort(Li[s])[::-1][:5])
    for r,c in enumerate(order):
        if c not in tb: miss_b+=1; missrank_b[r]+=1
        if c not in ti: miss_i+=1; missrank_i[r]+=1
print(f"\nN5 mean logit by faintness rank (r0=FAINTEST .. r4=strongest):")
print(f"  {'rank':>6} {'base':>8} {'inc16':>8} {'Δ(inc16-base)':>14}")
for r in range(5):
    print(f"  {('r'+str(r)):>6} {np.mean(rk_b[r]):>8.2f} {np.mean(rk_i[r]):>8.2f} {np.mean(rk_i[r])-np.mean(rk_b[r]):>14.2f}")
print(f"  {'decoy':>6} {np.mean(dec_b):>8.2f} {np.mean(dec_i):>8.2f} {np.mean(dec_i)-np.mean(dec_b):>14.2f}   (top absent / sample)")
print(f"  {'absent':>6} {np.mean(abs_b):>8.2f} {np.mean(abs_i):>8.2f} {np.mean(abs_i)-np.mean(abs_b):>14.2f}   (mean absent)")
print(f"\nN5 misses (true contributor outside top-5):  base={miss_b}  inc16={miss_i}")
print(f"  miss count by faintness rank r0..r4:  base={missrank_b}  inc16={missrank_i}")
# margin: faintest-true minus top-decoy (how far the faint minor is from beating the decoy)
mb=np.mean([rk_b[0][j]-dec_b[j] for j in range(len(dec_b))]); mi=np.mean([rk_i[0][j]-dec_i[j] for j in range(len(dec_i))])
print(f"\n  margin (faintest-true logit − top-decoy logit):  base={mb:+.2f}   inc16={mi:+.2f}   (more negative = miss)")

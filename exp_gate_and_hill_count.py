"""
exp_gate_and_hill_count.py — two count measurements on the LIVE checkpoint (inc22_fixed), no retraining:
  (A) GATE as count source: extract AdaSlot existence gate (B,45) + slot_mass via a forward hook.
      Does Sum(gate) track true NOC?  Where does it saturate?  Confirm the N5 joint_card collapse.
  (B) HILL q=0/q1/q2 count from the independent EM phi vs the RF-on-prob-profile count, per-NOC.
      Fair test of which INPUT signal carries N5 count info (all use the same RF mapping fit on val).
"""
import json, numpy as np, torch
from models.set_transformer import SetTransformerMixture
from sklearn.ensemble import RandomForestClassifier
import phi_rerank as pr

DEVICE="cpu"; RUN="results/inc22_fixed_aslot_seed42"
cfg=json.load(open(RUN+"/metrics.json"))["config"]
def LD(f): return np.load("data_insilico_w/%s.npy"%f)
Xt,Mt,yt,nt = LD("tokens8_test"),LD("mask_test").astype(bool),LD("y_test_set"),LD("noc_test").clip(1,5)
Xv,Mv,yv,nv = LD("tokens8_val"), LD("mask_val").astype(bool), LD("y_val_set"), LD("noc_val").clip(1,5)
g=np.load("data/donor_geno.npy").astype(np.float32); gmask=np.load("data/donor_geno_mask.npy")

ALLELE_OFF,n_cls,LUT_W=30,int(cfg.get("n_classes",45)),1024
owner_lut=torch.zeros(24,LUT_W,n_cls); gm=torch.from_numpy(gmask).bool()
for c in range(min(n_cls,g.shape[0])):
    for j in range(g.shape[1]):
        if gm[c,j]:
            li=int(g[c,j,0]); ab=int(round(float(g[c,j,1])*10))+ALLELE_OFF
            if 0<=li<24 and 0<=ab<LUT_W: owner_lut[li,ab,c]=1.0
model=SetTransformerMixture(
    n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),
    n_classes=n_cls,n_noc=cfg.get("n_noc",6),dropout=cfg.get("dropout",0.1),
    cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=int(cfg.get("n_token_feats",8)),encoder=cfg.get("encoder","isab"),
    num_embed=cfg.get("num_embed","raw"),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False),
    donor_geno=torch.from_numpy(g),donor_geno_mask=torch.from_numpy(gmask),
    nc_attn=cfg.get("nc_attn","none"),soft_geno_attr=cfg.get("soft_geno_attr",False),
    feas_filter=cfg.get("feas_filter",False),set_of_set=cfg.get("set_of_set",False),owner_lut=owner_lut,
    n_slot_iters=int(cfg.get("n_slot_iters",3)),ot_eps=float(cfg.get("ot_eps",0.05)),
    ot_iters=int(cfg.get("ot_iters",5)),noc_head_v2=bool(cfg.get("noc_head_v2",False)),
).to(DEVICE)
model.load_state_dict(torch.load(RUN+"/best_model.pt",weights_only=True,map_location=DEVICE),strict=False)
model.eval()

_cap={}
def hook(m,i,o): _cap["gate"]=o["gate"].detach().cpu().numpy(); _cap["slot_mass"]=o["slot_mass"].detach().cpu().numpy()
model.cls_decoder_module.register_forward_hook(hook)

@torch.no_grad()
def infer(X,M):
    P=[];card=[];gate=[];smass=[]
    for s in range(0,len(X),256):
        out=model(torch.tensor(X[s:s+256]),torch.tensor(M[s:s+256]))
        P.append(torch.sigmoid(out["logits_cls"]).numpy())
        card.append(out["logits_card"].numpy())
        gate.append(_cap["gate"]); smass.append(_cap["slot_mass"])
    return (np.concatenate(P),np.concatenate(card),np.concatenate(gate),np.concatenate(smass))
P_te,C_te,G_te,SM_te=infer(Xt,Mt); P_va,C_va,G_va,SM_va=infer(Xv,Mv)
print(f"inferred. gate {G_te.shape}\n")

def per_noc_acc(pred,noc):
    return {j:round(float((pred[noc==j]==j).mean()),3) for j in range(1,6)}, round(float((pred==noc).mean()),4)

# ===== (A) GATE as count source =====
print("=== (A) GATE existence as NOC signal (test) ===")
sumg=G_te.sum(1); hardg=(G_te>0.5).sum(1)
for j in range(1,6):
    m=nt==j
    print(f"  NOC={j}: mean Sum(gate)={sumg[m].mean():5.2f}  mean #(gate>0.5)={hardg[m].mean():5.2f}  (n={m.sum()})")
print(f"  corr( Sum(gate), trueNOC ) = {np.corrcoef(sumg,nt)[0,1]:.3f}")
jc_pred=C_te.argmax(1)+1   # aslot gate-based joint_card
acc,ov=per_noc_acc(jc_pred,nt); print(f"  aslot joint_card (argmax logits_card) per-NOC {acc} overall {ov}")
smc=(SM_te>0.5).sum(1)
print(f"  slot_mass: mean #(slot_mass>0.5) by NOC: "+
      " ".join(f"N{j}={smc[nt==j].mean():.2f}" for j in range(1,6)))

# ===== (B) HILL count vs RF =====
def hill_feats(PH):
    p=PH/np.clip(PH.sum(1,keepdims=True),1e-9,None)
    rich=lambda t:(p>t).sum(1)
    with np.errstate(divide="ignore",invalid="ignore"):
        sh=-np.nansum(np.where(p>0,p*np.log(p),0.0),1); q1=np.exp(sh)
    q2=1.0/np.clip((p**2).sum(1),1e-9,None)
    return np.stack([rich(0.02),rich(0.05),rich(0.10),q1,q2],1), p
PH_te=pr.deconv_phi(Xt,Mt,g,gmask,n_iters=12); PH_va=pr.deconv_phi(Xv,Mv,g,gmask,n_iters=12)
Hte,pte=hill_feats(PH_te); Hva,pva=hill_feats(PH_va)

def card_feats(P):
    s=np.sort(P,1)[:,::-1][:,:8]
    return np.concatenate([s,P.sum(1,keepdims=True),(P>=0.5).sum(1,keepdims=True)],1)
def gate_feats(G):
    s=np.sort(G,1)[:,::-1][:,:10]
    return np.concatenate([s,G.sum(1,keepdims=True),(G>0.5).sum(1,keepdims=True)],1)

def rf(Ftr,ytr,Fte):
    return RandomForestClassifier(n_estimators=300,max_depth=6,random_state=42).fit(Ftr,ytr).predict(Fte)

print("\n=== (B) count accuracy by INPUT signal (RF fit on val->true NOC) ===")
sets={
 "prob-profile (current RF)": (card_feats(P_va),card_feats(P_te)),
 "phi-Hill (q0/q1/q2)":       (Hva,Hte),
 "gate profile":              (gate_feats(G_va),gate_feats(G_te)),
 "Hill + gate + prob":        (np.concatenate([Hva,gate_feats(G_va),card_feats(P_va)],1),
                               np.concatenate([Hte,gate_feats(G_te),card_feats(P_te)],1)),
}
for name,(Ftr,Fte) in sets.items():
    acc,ov=per_noc_acc(rf(Ftr,nv,Fte),nt); print(f"  {name:26s} N5={acc[5]:.3f}  N4={acc[4]:.3f}  N3={acc[3]:.3f}  overall={ov:.4f}")
# naive (uncalibrated) Hill rounding, N5 sensitivity
for col,nm in [(3,"round(q1=expShannon)"),(4,"round(q2=invSimpson)")]:
    pr_=np.clip(np.rint(Hte[:,col]),1,5).astype(int); acc,ov=per_noc_acc(pr_,nt)
    print(f"  naive {nm:22s} N5={acc[5]:.3f}  N4={acc[4]:.3f}  overall={ov:.4f}")
print("\n  mean phi-Hill q1(expShannon) by true NOC: "+" ".join(f"N{j}={Hte[nt==j,3].mean():.2f}" for j in range(1,6)))
print(  "  mean phi-Hill q0(rich@0.05) by true NOC: "+" ".join(f"N{j}={Hte[nt==j,1].mean():.2f}" for j in range(1,6)))

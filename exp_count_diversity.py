"""
C1 — count-estimator DIVERSITY + no-trade ceiling (MEASUREMENT, not a foregone conclusion).
Are the count signals {prob-profile, gate, MAC} DIVERSE (make different N4/N5 errors)? If yes, a stack
could reach BOTH N4 and N5 (no trade). If they err the SAME way, no combiner escapes the N4<->N5 trade.
All estimators = RandomForest fit on TRUE noc_val (C6-clean: val labels only), eval on test.
"""
import json, numpy as np, torch
from models.set_transformer import SetTransformerMixture
from sklearn.ensemble import RandomForestClassifier
RUN="results/inc22_fixed_aslot_seed42"
def LD(f): return np.load("data_insilico_w/%s.npy"%f)
Xt,Mt,nt=LD("tokens8_test"),LD("mask_test").astype(bool),LD("noc_test").clip(1,5)
Xv,Mv,nv=LD("tokens8_val"), LD("mask_val").astype(bool), LD("noc_val").clip(1,5)
g=np.load("data/donor_geno.npy").astype(np.float32); gmask=np.load("data/donor_geno_mask.npy")
owner_lut=torch.zeros(24,1024,45); gm=torch.from_numpy(gmask).bool()
for c in range(45):
    for j in range(g.shape[1]):
        if gm[c,j]:
            li=int(g[c,j,0]); ab=int(round(float(g[c,j,1])*10))+30
            if 0<=li<24 and 0<=ab<1024: owner_lut[li,ab,c]=1.0
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,
    n_noc=6,dropout=0.1,cls_decoder="aslot",n_token_feats=8,encoder="isab++",num_embed="periodic",
    periodic_sigma=0.3,aux_heads=True,sparse_attn=True,donor_geno=torch.from_numpy(g),
    donor_geno_mask=torch.from_numpy(gmask),nc_attn="mab0",soft_geno_attr=True,feas_filter=True,
    set_of_set=True,owner_lut=owner_lut,n_slot_iters=3,ot_eps=0.05,ot_iters=5,noc_head_v2=True)
m.load_state_dict(torch.load(RUN+"/best_model.pt",weights_only=True,map_location="cpu"),strict=False); m.eval()
_cap={}
m.cls_decoder_module.register_forward_hook(lambda mod,i,o:_cap.update(gate=o["gate"].detach().numpy()))
@torch.no_grad()
def infer(X,M):
    P,G,MAC=[],[],[]
    for s in range(0,len(X),256):
        t=torch.tensor(X[s:s+256]); msk=torch.tensor(M[s:s+256])
        out=m(t,msk); P.append(torch.sigmoid(out["logits_cls"]).numpy()); G.append(_cap["gate"])
        MAC.append(m._mac_feats_torch(t,msk).numpy())
    return np.concatenate(P),np.concatenate(G),np.concatenate(MAC)
Pv,Gv,Cv=infer(Xv,Mv); Pt,Gt,Ct=infer(Xt,Mt)

def card_feats(P): s=np.sort(P,1)[:,::-1][:,:8]; return np.concatenate([s,P.sum(1,keepdims=True),(P>=0.5).sum(1,keepdims=True)],1)
def gate_feats(G): s=np.sort(G,1)[:,::-1][:,:10]; return np.concatenate([s,G.sum(1,keepdims=True),(G>0.5).sum(1,keepdims=True)],1)
def rf(Ftr,Fte): return RandomForestClassifier(300,max_depth=6,random_state=42).fit(Ftr,nv).predict(Fte)

est={"prob":rf(card_feats(Pv),card_feats(Pt)),
     "gate":rf(gate_feats(Gv),gate_feats(Gt)),
     "mac": rf(Cv,Ct)}
stack=rf(np.concatenate([card_feats(Pv),gate_feats(Gv),Cv],1),
         np.concatenate([card_feats(Pt),gate_feats(Gt),Ct],1))

def pnc(p): return " ".join(f"N{j}={np.mean(p[nt==j]==j):.3f}" for j in range(2,6))
print("=== individual count estimators (per NOC, test) ===")
for k,p in est.items(): print(f"  {k:5s}: {pnc(p)}  overall={np.mean(p==nt):.4f}")
print(f"  STACK: {pnc(stack)}  overall={np.mean(stack==nt):.4f}")

print("\n=== DIVERSITY: do they err differently? ===")
ks=list(est);
for a in range(len(ks)):
    for b in range(a+1,len(ks)):
        ea=(est[ks[a]]!=nt); eb=(est[ks[b]]!=nt)
        both=np.mean(ea&eb); dis=np.mean(ea!=eb)
        # complementarity on N5: when A wrong, is B right?
        n5=nt==5; aw=n5&ea; comp5=np.mean(~eb[aw]) if aw.sum() else float('nan')
        print(f"  {ks[a]} vs {ks[b]}: both-wrong={both:.3f} disagree={dis:.3f} | N5: P(B right | A wrong)={comp5:.2f}")

print("\n=== NO-TRADE CEILING: '>=1 of {prob,gate,mac} correct' per NOC (best any combiner could do) ===")
correct=np.stack([est[k]==nt for k in est],0)   # (3,N)
anyc=correct.any(0)
for j in range(2,6):
    sel=nt==j; print(f"  N{j}: >=1 correct = {np.mean(anyc[sel]):.3f}   (vs best single = {max(np.mean((est[k]==nt)[sel]) for k in est):.3f})")
print(f"  overall >=1 correct = {np.mean(anyc):.4f}")

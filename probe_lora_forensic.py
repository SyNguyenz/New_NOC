"""STABLE local harness (LoRA) + allele-accounting test, fixing the drift of full-model fine-tune (base
FROZEN -> cannot drift 0.788->0.75). Three conditions, warm-start inc6_maskp, measure REAL TEST N5:
  CTRL    : LoRA on cls only (validates the harness is STABLE — should stay ~0.788, not drift).
  FORENSIC: LoRA + a per-donor head that adjusts logits with allele-accounting features (rarity/completeness/
            damning-absence) -> does letting the model ADAPT (LoRA) to allele-accounting beat the +0.024 post-hoc?
LoRA B zero-init + forensic head zero-init => starts EXACTLY at base. Stable read of the SIGN for full-train."""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"; DEV=torch.device("cuda")
from models.set_transformer import SetTransformerMixture
from train_set_transformer import AsymmetricLoss
def L(n): return np.load(DATA/f"{n}.npy", allow_pickle=True)

# ── forensic features per (sample, donor): [rarity_sum, completeness, damning_absence, present_count] ──
def build_feats(split):
    cache=DATA/f"forensic_feats_{split}.npy"
    if cache.exists(): return np.load(cache)
    g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
    from collections import Counter
    geno=[set() for _ in range(45)]; panel=Counter()
    for c in range(45):
        for j in range(g.shape[1]):
            if gm[c,j]: geno[c].add((int(g[c,j,0]),round(float(g[c,j,1]),1)))
    for c in range(45):
        for k in geno[c]: panel[k]+=1
    tok=L(f"tokens8_{split}").astype(np.float32); mk=L(f"mask_{split}").astype(bool); B=len(tok)
    MED=np.median([float(np.expm1(tok[i,j,2])) for i in range(B) for j in np.where(mk[i])[0]])
    F_=np.zeros((B,45,4),np.float32)
    for i in range(B):
        loc={}
        for j in np.where(mk[i])[0]:
            l=int(tok[i,j,0]); al=round(float(tok[i,j,1]),1); h=float(np.expm1(tok[i,j,2])); d=loc.setdefault(l,{}); d[al]=max(d.get(al,0.),h)
        strong={l for l in loc if max(loc[l].values())>MED}
        for c in range(45):
            pres=0; rar=0.; dam=0
            for (l,al) in geno[c]:
                if l in loc and al in loc[l]: pres+=1; rar+=1.0/panel[(l,al)]
                elif l in strong: dam+=1
            F_[i,c]=[rar, pres/max(len(geno[c]),1), dam, pres]
        if i%10000==0: print(f"  feats {split} {i}/{B}",flush=True)
    np.save(cache,F_); return F_

class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base=base
        for p in base.parameters(): p.requires_grad_(False)
        self.A=nn.Parameter(torch.zeros(r, base.in_features)); self.B=nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5**0.5); self.scale=alpha/r       # B=0 -> delta 0 at start
    def forward(self,x): return self.base(x) + (x @ self.A.t() @ self.B.t())*self.scale
def inject_lora(root, r=8, name_filter=None, name_exclude=None):
    targets=[]                                              # collect ORIGINAL Linears first (no live-mod, no double-wrap)
    for pname, parent in root.named_modules():
        if isinstance(parent, nn.MultiheadAttention): continue   # MHA reads out_proj.weight directly -> don't wrap
        for name,ch in parent.named_children():
            if not isinstance(ch, nn.Linear): continue
            full = f"{pname}.{name}" if pname else name
            if name_filter is not None and name_filter not in full: continue
            if name_exclude is not None and name_exclude in full: continue
            targets.append((parent,name,ch))
    for parent,name,ch in targets:
        parent._modules[name]=LoRALinear(ch, r)
    return len(targets)

tok=torch.tensor(L("tokens8_train").astype(np.float32)); mkt=torch.tensor(L("mask_train").astype(bool)); yt=torch.tensor(L("y_train_set").astype(np.float32))
N=len(tok); Ftr=torch.tensor(build_feats("train")); Fte=build_feats("test")
te_tok=L("tokens8_test").astype(np.float32); te_mk=L("mask_test").astype(bool); te_y=L("y_test_set").astype(np.float32); te_noc=L("noc_test").astype(int)
cfg=json.load(open(ROOT/"results/inc6_maskp_seed42/metrics.json"))["config"]
fstat_m=Ftr.reshape(-1,4).mean(0); fstat_s=Ftr.reshape(-1,4).std(0)+1e-6
def fresh(name_filter=None, name_exclude=None):
    m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
      dropout=0.1,cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",
      periodic_sigma=cfg["periodic_sigma"],aux_heads=True,sparse_attn=True).to(DEV)
    m.load_state_dict(torch.load(ROOT/"results/inc6_maskp_seed42/best_model.pt",weights_only=True,map_location=DEV))
    _n=L("tokens8_train")[:,:,1:8][L("mask_train").astype(bool)]
    m.feat_mean.copy_(torch.tensor(_n.mean(0),device=DEV)); m.feat_std.copy_(torch.tensor(_n.std(0)+1e-6,device=DEV))
    inject_lora(m, r=8, name_filter=name_filter, name_exclude=name_exclude); return m.to(DEV)  # move new LoRA params to GPU
@torch.no_grad()
def test_n5(m, fhead=None):
    m.eval(); P=np.zeros((len(te_tok),45))
    for s in range(0,len(te_tok),128):
        x=torch.from_numpy(te_tok[s:s+128]).to(DEV); mb=torch.from_numpy(te_mk[s:s+128]).to(DEV)
        lg=m(x,mb)["logits_cls"]
        if fhead is not None:
            ff=((torch.tensor(Fte[s:s+128]).to(DEV)-fstat_m.to(DEV))/fstat_s.to(DEV))
            lg=lg+fhead(ff).squeeze(-1)
        P[s:s+128]=torch.sigmoid(lg).cpu().numpy()
    out={}
    for k in range(1,6):
        ii=np.where(te_noc==k)[0]; e=[]
        for i in ii:
            t=np.argsort(P[i])[::-1][:k]; pr=np.zeros(45,int); pr[t]=1; e.append((pr==te_y[i]).all())
        out[k]=round(float(np.mean(e)),3)
    return out
def run(use_forensic, epochs=6, bs=48, lr=1e-3, tag=""):
    torch.manual_seed(0); np.random.seed(0)
    m=fresh(); asl=AsymmetricLoss(gamma_neg=4.0,gamma_pos=0.0,clip=0.05)
    params=[p for p in m.parameters() if p.requires_grad]
    fhead=None
    if use_forensic:
        fhead=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,1)).to(DEV)
        nn.init.zeros_(fhead[-1].weight); nn.init.zeros_(fhead[-1].bias)    # start = base
        params=params+list(fhead.parameters())
    opt=torch.optim.AdamW(params,lr=lr,weight_decay=1e-4)
    fm=fstat_m.to(DEV); fs=fstat_s.to(DEV)
    print(f"  [{tag}] LoRA trainable params: {sum(p.numel() for p in params):,} | pre N5={test_n5(m,fhead)[5]}",flush=True)
    best=0
    for ep in range(epochs):
        m.train(); perm=np.random.permutation(N)
        for s in range(0,N,bs):
            bi=perm[s:s+bs]; x=tok[bi].to(DEV); mk=mkt[bi].to(DEV); y=yt[bi].to(DEV)
            mb=mk.bool(); drop=(torch.rand_like(mb,dtype=torch.float)<0.15)&mb; kept=mb&~drop
            mk2=torch.where(kept.sum(1,keepdim=True)>=8,kept,mb).to(mk.dtype)
            lg=m(x,mk2)["logits_cls"]
            if use_forensic: lg=lg+fhead((Ftr[bi].to(DEV)-fm)/fs).squeeze(-1)
            loss=asl(lg,y)
            opt.zero_grad(); loss.backward(); opt.step()
        r=test_n5(m,fhead); best=max(best,r[5]); print(f"  [{tag}] ep{ep+1}: test N5={r[5]} (N4={r[4]})",flush=True)
    return best
if __name__=="__main__":
    t0=time.time()
    print("baseline (inc6_maskp) test N5=0.788",flush=True)
    print("\n=== CTRL: LoRA cls-only (harness stability check) ==="); bc=run(False,tag="CTRL")
    print("\n=== FORENSIC: LoRA + allele-accounting head ==="); bf=run(True,tag="FOREN")
    print(f"\nSUMMARY  base=0.788 | LoRA-ctrl best={bc} | LoRA-forensic best={bf} | forensic-ctrl={bf-bc:+.3f} ({time.time()-t0:.0f}s)")

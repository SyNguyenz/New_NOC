"""
Can STKIM (de-emphasize the dominant majors) solve the encoder's N5 isolation problem THOROUGHLY?

STKIM's mechanism = stop the few dominant instances monopolizing attention. No-train CEILING of that
mechanism, end-to-end (full model decode, no probe-readout confound): progressively SUPPRESS the major
contributors' (rank<4) height features toward the set-min by (1-alpha), KEEP the faintest (rank4), and
read the deployed decoder. Per alpha on real N5:
   faint(rank4) recall  -> how high can the minor be lifted (the isolation CEILING of de-emphasis)
   major(rank0-3) recall -> what de-emphasis COSTS the majors
   set-EM               -> net (cost included)

Read:
  faint recall climbs toward the major level (~.97) at strong suppression  => the minor CAN be isolated
     once the majors stop dominating => STKIM (which KEEPS majors but learns balanced attention) has a HIGH
     ceiling -> it can plausibly solve it (its true ceiling >= the best-alpha set-EM, since it avoids the
     major-recall cost this crude probe pays).
  faint recall plateaus well below ~.95 even at full suppression => a residual the mechanism CANNOT fix
     (hard floor) => STKIM helps but does NOT solve it thoroughly.
"""
import json
import numpy as np, torch
from pathlib import Path
from models.set_transformer import SetTransformerMixture

DATA=Path("data_insilico_w"); RUN="results/inc2_2d_sparse_seed42"; DEV="cuda" if torch.cuda.is_available() else "cpu"
HCOLS=[2,5,7]
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

samp=[]
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0 or noc[gi]!=5: continue
    lh=tk[gi][:,2]; info={int(d):float(np.exp(lh[a==d]).sum()) for d in np.unique(a[v])}
    if len(info)!=5: continue
    order=sorted(info,key=lambda d:-info[d]); rank={d:r for r,d in enumerate(order)}
    samp.append((gi,v,rank))
print(f"N5 samples: {len(samp)}\n")

@torch.no_grad()
def decode(tkrow,mkrow):
    o=m(torch.from_numpy(tkrow[None]).to(DEV),torch.from_numpy(mkrow[None]).to(DEV))
    return torch.sigmoid(o["logits_cls"])[0].cpu().numpy()

for alpha in [1.0,0.7,0.4,0.0]:
    em=fr=0; mr=[0,0]
    for gi,v,rank in samp:
        tki=tk[gi].copy()
        strong=[j for j in v if rank[int(at[gi][j])]<4]
        for c in HCOLS:
            mn=tk[gi][v,c].min(); tki[strong,c]=mn+alpha*(tk[gi][strong,c]-mn)
        P=decode(tki,mk[gi]); top=set(np.argsort(P)[::-1][:5]); ts=set(np.where(y[gi]==1)[0])
        faint=[d for d in rank if rank[d]==4][0]
        em+=(top==ts); fr+=(faint in top)
        for d in rank:
            if rank[d]<4: mr[1]+=1; mr[0]+=(d in top)
    n=len(samp)
    print(f"alpha={alpha:>4} (1=native,0=majors flattened): faint(r4) recall {fr/n:.3f}  "
          f"major(r0-3) recall {mr[0]/mr[1]:.3f}  N5 set-EM {em/n:.3f}")
print("\n(faint recall is the de-emphasis ISOLATION ceiling; set-EM pays a major-recall cost STKIM would NOT.)")

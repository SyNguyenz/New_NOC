import json
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from train_set_transformer import DEVICE
from models.set_transformer import SetTransformerMixture
from features.enrich import enrich_tokens

D = Path("data_insilico_w")

def build(cfg, state):
    kw = dict(n_loci=cfg.get("n_loci",24), d_locus=cfg.get("d_locus",16), d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4), n_isab=cfg.get("n_isab",2), m_inducing=cfg.get("m_inducing",32),
        n_classes=cfg.get("n_classes",45), n_noc=cfg.get("n_noc",6), dropout=cfg.get("dropout",0.1),
        cls_decoder=cfg.get("cls_decoder","pooled"), decoder_source=cfg.get("decoder_source","encoded"),
        n_token_feats=cfg.get("n_token_feats",8), encoder=cfg.get("encoder","isab"), dec_layers=cfg.get("dec_layers",2),
        num_embed=cfg.get("num_embed","raw"), n_freq=cfg.get("n_freq",8), d_num_emb=cfg.get("d_num_emb",8),
        periodic_sigma=cfg.get("periodic_sigma",1.0), aux_heads=cfg.get("aux_heads",False),
        noc_contrast=cfg.get("noc_contrast",False), noc_detach=(cfg.get("noc_contrast_mode","shared")=="detach"),
        d_proj=cfg.get("d_proj",64), sparse_attn=cfg.get("sparse_attn",False),
        geno_query=bool(cfg.get("geno_query")), donor_contrast=cfg.get("donor_contrast",False),
        noc_ord_head=cfg.get("noc_ord_head",False), noc_ord_detach=cfg.get("noc_ord_detach",False),
        noc_ord_replace=cfg.get("noc_ord_replace",False))
    if cfg.get("geno_query"):
        kw["donor_geno"]=torch.zeros_like(state["donor_geno"]).float()
        kw["donor_geno_mask"]=torch.zeros_like(state["donor_geno_mask"]).bool()
    return SetTransformerMixture(**kw).to(DEVICE)

def load(arm):
    ck=Path("results")/f"{arm}_seed42"; cfg=json.load(open(ck/"metrics.json"))["config"]
    state=torch.load(ck/"best_model.pt",map_location=DEVICE,weights_only=True)
    if "p5_noc_intrinsic" in arm:
        from train_p5_noc_intrinsic import P5Model; m=P5Model().to(DEVICE)
    else:
        m=build(cfg,state)
    m.load_state_dict(state); m.eval(); return m, cfg.get("n_token_feats",8)

tok3=np.load(D/"tokens_test.npy").astype(np.float32); mk=np.load(D/"mask_test.npy").astype(bool)
noc=np.clip(np.load(D/"noc_test.npy").astype(int),1,5); enC=enrich_tokens(tok3,mk)
tokO=np.load(D/"tokens8_open.npy").astype(np.float32); mkO=np.load(D/"mask_open.npy").astype(bool)

def rej(model,n_tok,en,mask):
    R=[]
    with torch.no_grad():
        for i in range(0,len(en),256):
            o=model(torch.from_numpy(en[i:i+256,:,:n_tok]).to(DEVICE), torch.from_numpy(mask[i:i+256]).to(DEVICE))
            if "logit_reject" not in o: return None
            R.append(torch.sigmoid(o["logit_reject"]).cpu().numpy().ravel())
    return np.concatenate(R)

arms=["inc4_p1_stack","inc4_p2_local","inc4_p3_irm","inc4_p4_decorr","inc4_p5_noc_intrinsic",
      "inc3_repA_genoq","inc3_nocV2_ordrnc_pcgrad"]
print(f"open n={len(mkO)}  closed full-test n={len(mk)} (N4/5={int((noc>=4).sum())})")
print(f"{'arm':<28}{'AUROC_all':>10}{'AUROC_N45':>11}  (train-time reject from metrics.json)")
for a in arms:
    model,n_tok=load(a)
    Rc=rej(model,n_tok,enC,mk); Ro=rej(model,n_tok,tokO[:,:,:n_tok],mkO)
    if Rc is None: print(f"{a:<28}  no reject head"); continue
    au=roc_auc_score(np.r_[np.zeros(len(Rc)),np.ones(len(Ro))], np.r_[Rc,Ro])
    hi=noc>=4
    au2=roc_auc_score(np.r_[np.zeros(hi.sum()),np.ones(len(Ro))], np.r_[Rc[hi],Ro])
    old=json.load(open(Path("results")/f"{a}_seed42"/"metrics.json")).get("reject_auroc","?")
    print(f"{a:<28}{au:>10.4f}{au2:>11.4f}  ({old})")

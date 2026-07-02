"""
train_hybrid.py — Two-stream HYBRID Set Transformer.

  logits_cls = flat_head(Xflat)  +  set_scale * tanh(set_adjust(z_mix))

Aligned Xflat 590-bin base (LR/XGB-like, preserves minority donors that the
permutation-invariant token-set drowns) + bounded ISAB set-attention adjustment
(interaction/joint structure, the thing that made ST win on leaky data).
NOC + reject heads still read z_mix (ST's genuine strengths).

Motivation: probe showed donor 48 (combo 46,47,48) gets prob 0.75 from LR (Xflat)
but 0.00-0.13 from every token-set ST variant. Hybrid keeps the flat base as a
non-suppressible floor while letting the set stream add context.

Data: tokens/mask/Xflat/y/noc per split + open set for reject head.

Usage:
  python train_hybrid.py
  python train_hybrid.py --set_scale_fixed 0.0   # ablation: flat-only (=MLP on Xflat)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from sklearn.metrics import f1_score, roc_auc_score

import sys
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data")))
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
import torch.nn.functional as F
from train_set_transformer import (compute_pos_weight, full_report, cardinality_target,
                                    topk_decode, posthoc_cardinality, per_noc_em,
                                    card_features, two_stage_cardinality,
                                    build_pgnoc_refs, pgnoc_cost_features)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HybridDataset(Dataset):
    """Closed-set: tokens, mask, Xflat, y, noc."""
    def __init__(self, split: str, tok_prefix: str = "tokens"):
        self.tokens = torch.from_numpy(np.load(DATA_DIR / f"{tok_prefix}_{split}.npy").astype(np.float32))
        self.mask   = torch.from_numpy(np.load(DATA_DIR / f"mask_{split}.npy"))
        self.xflat  = torch.from_numpy(np.load(DATA_DIR / f"Xflat_{split}.npy").astype(np.float32))
        self.y      = torch.from_numpy(np.load(DATA_DIR / f"y_{split}_set.npy"))
        self.noc    = torch.from_numpy(np.load(DATA_DIR / f"noc_{split}.npy").astype(np.int64))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.mask[i], self.xflat[i], self.y[i], self.noc[i]


class OpenSetDataset(Dataset):
    """Open-set tokens + mask + Xflat (for reject head; reject_label=1)."""
    def __init__(self, tok_prefix: str = "tokens"):
        self.tokens = torch.from_numpy(np.load(DATA_DIR / f"{tok_prefix}_open.npy").astype(np.float32))
        self.mask   = torch.from_numpy(np.load(DATA_DIR / "mask_open.npy"))
        self.xflat  = torch.from_numpy(np.load(DATA_DIR / "Xflat_open.npy").astype(np.float32))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.mask[i], self.xflat[i]


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_true, all_pred, all_noc = [], [], []
    for tokens, mask, xflat, y, noc in loader:
        out = model(tokens.to(DEVICE), mask.to(DEVICE), xflat.to(DEVICE))
        probs = torch.sigmoid(out["logits_cls"]).cpu().numpy()
        all_pred.append((probs >= threshold).astype(np.float32))
        all_true.append(y.numpy())
        all_noc.append(noc.numpy())
    y_true = np.concatenate(all_true); y_pred = np.concatenate(all_pred)
    noc_all = np.concatenate(all_noc)
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return mf1, y_true, y_pred, noc_all


@torch.no_grad()
def evaluate_oracle_em(model, loader):
    """Returns (overall oracle EM, MACRO-over-NOC oracle Recall@k) on val.
    SELECT on macro-recall: (a) graded ranking metric (R@k family) — smoother/less noisy than
    all-or-nothing subset-EM for model selection (multi-label lit: subset accuracy too strict);
    (b) macro-averaged over NOC strata 1..5 — includes NOC1 (catches regression) but at 1/5 weight
    so it does NOT saturate (overall EM/recall is ~constant via the flat base on 82%-NOC1 val);
    discriminative signal lives in N4/N5. recall@k with k=#true donors == precision@k here."""
    model.eval()
    all_probs, all_true, all_noc = [], [], []
    for tokens, mask, xflat, y, noc in loader:
        out = model(tokens.to(DEVICE), mask.to(DEVICE), xflat.to(DEVICE))
        all_probs.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
        all_true.append(y.numpy()); all_noc.append(noc.numpy())
    probs = np.concatenate(all_probs); y_true = np.concatenate(all_true); noc = np.concatenate(all_noc)
    yp = np.zeros_like(probs, dtype=int); rec = np.zeros(len(probs))
    for i in range(len(probs)):
        k = int(max(1, min(5, noc[i])))
        top = np.argsort(probs[i])[::-1][:k]; yp[i, top] = 1
        rec[i] = y_true[i][top].sum() / k                       # oracle recall@k (= precision@k, k=#true)
    em = (y_true == yp).all(1)
    nocc = np.clip(noc, 1, 5)
    strata = [rec[nocc == j].mean() for j in range(1, 6) if (nocc == j).any()]
    return float(em.mean()), float(np.mean(strata))


def train(cfg):
    subdir = cfg.get("out_subdir", "set_transformer_hybrid")
    results_dir = ROOT / "results" / subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    n_tok = cfg.get("n_token_feats", 3)
    tok_prefix = f"tokens{n_tok}" if n_tok > 3 else "tokens"
    train_ds = HybridDataset("train", tok_prefix); val_ds = HybridDataset("val", tok_prefix); test_ds = HybridDataset("test", tok_prefix)
    open_ds = OpenSetDataset(tok_prefix)
    n_flat = train_ds.xflat.shape[1]

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    # SELECTION set = combo-disjoint balanced in-silico DEV if present (make_dev_split.py);
    # else fall back to real val. Real val is still used for the decoder stage1 prior.
    if (DATA_DIR / f"{tok_prefix}_dev.npy").exists():
        sel_loader = DataLoader(HybridDataset("dev", tok_prefix), batch_size=256, shuffle=False)
        print(f"selection set = in-silico DEV ({len(sel_loader.dataset)} samples)")
    else:
        sel_loader = val_loader; print("selection set = real val (no dev split found)")
    open_iter = iter(DataLoader(
        open_ds, sampler=RandomSampler(open_ds, replacement=True, num_samples=len(train_ds)*cfg["epochs"]),
        batch_size=max(1, int(cfg["batch_size"]*cfg.get("open_ratio", 0.25)))))

    model = SetTransformerMixture(
        n_loci=cfg.get("n_loci",24), d_locus=cfg.get("d_locus",16), d_model=cfg.get("d_model",128),
        n_heads=cfg.get("n_heads",4), n_isab=cfg.get("n_isab",2), m_inducing=cfg.get("m_inducing",32),
        n_classes=45, n_noc=6, dropout=cfg.get("dropout",0.1),
        cls_decoder="hybrid", n_flat=n_flat,
        decouple_reject=cfg.get("decouple_reject", False), n_token_feats=n_tok,
        encoder=cfg.get("encoder", "isab"),
        num_embed=cfg.get("num_embed", "raw"), n_freq=cfg.get("n_freq", 8),
        d_num_emb=cfg.get("d_num_emb", 8), periodic_sigma=cfg.get("periodic_sigma", 1.0)).to(DEVICE)

    # Enriched tokens (n_token_feats=8): set per-feature standardization from train valid peaks.
    if n_tok > 3:
        tk = train_ds.tokens.numpy(); mk = train_ds.mask.numpy().astype(bool)
        num = tk[:, :, 1:n_tok][mk]
        model.feat_mean.copy_(torch.tensor(num.mean(0), dtype=torch.float32, device=DEVICE))
        model.feat_std.copy_(torch.tensor(num.std(0) + 1e-6, dtype=torch.float32, device=DEVICE))
        print(f"enriched tokens: n_token_feats={n_tok}, num_embed={cfg.get('num_embed','raw')}, feat_std={np.round(num.std(0),3)}")

    set_scale_fixed = cfg.get("set_scale_fixed", None)
    if set_scale_fixed is not None:
        model.set_scale.data = torch.tensor(float(set_scale_fixed), device=DEVICE)
        model.set_scale.requires_grad_(False)
        print(f"set_scale FIXED at {set_scale_fixed} (not learned)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device {DEVICE} | Params {n_params:,} | n_flat={n_flat}")
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)} | Open {len(open_ds)}")

    y_train_np = np.load(DATA_DIR / "y_train_set.npy")
    if cfg.get("loss", "bce") == "asl":
        from train_set_transformer import AsymmetricLoss
        bce_cls = AsymmetricLoss(
            gamma_neg=cfg.get("asl_gamma_neg", 4.0),
            gamma_pos=cfg.get("asl_gamma_pos", 0.0),
            clip=cfg.get("asl_clip", 0.05),
        )
        print(f"cls loss: ASL(gamma_neg={cfg.get('asl_gamma_neg',4.0)}, "
              f"gamma_pos={cfg.get('asl_gamma_pos',0.0)}, clip={cfg.get('asl_clip',0.05)})")
    else:
        pos_weight = compute_pos_weight(y_train_np).clamp(max=cfg.get("pos_weight_cap",10.0)).to(DEVICE)
        bce_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"cls loss: BCE(pos_weight cap={cfg.get('pos_weight_cap',10.0)})")
    bce_rej = nn.BCEWithLogitsLoss()
    alpha = cfg.get("alpha_reject",0.5); beta = cfg.get("beta_card",0.3)
    card_lam = cfg.get("card_lambda", 0.02)
    # Cardinality class weights (train is ~85% NOC=1 -> plain CE collapses to k=1)
    _nc = np.bincount(np.clip(np.load(DATA_DIR/"noc_train.npy"),1,5)-1, minlength=5).astype(float)
    _w = 1.0/np.clip(_nc,1,None); _w = _w/_w.mean()
    card_w = torch.tensor(np.clip(_w, 0.5, 2.0), dtype=torch.float32).to(DEVICE)   # bounded both ways
    print(f"card class weights: {[round(x,2) for x in card_w.tolist()]}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr",3e-4), weight_decay=cfg.get("weight_decay",1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)

    best_f1, best_epoch, patience_count = 0.0, 0, 0
    patience = cfg.get("patience",15); epochs = cfg.get("epochs",100); history=[]
    print(f"\nTraining up to {epochs} epochs (patience={patience}) ...")
    t0=time.time()

    for epoch in range(1, epochs+1):
        model.train(); epoch_loss=epoch_cls=epoch_rej=epoch_noc_l=0.0
        for tokens, mask, xflat, y, noc in train_loader:
            tokens, mask, xflat = tokens.to(DEVICE), mask.to(DEVICE), xflat.to(DEVICE)
            y, noc = y.to(DEVICE), noc.to(DEVICE)
            out = model(tokens, mask, xflat)
            loss_cls = bce_cls(out["logits_cls"], y)
            card_tgt = cardinality_target(torch.sigmoid(out["logits_cls"]).detach(), y, card_lam)
            loss_noc = F.cross_entropy(out["logits_card"], card_tgt, weight=card_w)
            rej_closed = out["logit_reject"]; rej_label = torch.zeros(len(tokens),1,device=DEVICE)
            try:
                ob = next(open_iter)
            except StopIteration:
                open_iter = iter(DataLoader(open_ds, batch_size=max(1,int(cfg["batch_size"]*cfg.get("open_ratio",0.25))),
                                            sampler=RandomSampler(open_ds, replacement=True, num_samples=len(open_ds)*10)))
                ob = next(open_iter)
            o_tok,o_mask,o_xf = ob[0].to(DEVICE),ob[1].to(DEVICE),ob[2].to(DEVICE)
            rej_open = model(o_tok,o_mask,o_xf)["logit_reject"]
            all_rej = torch.cat([rej_closed, rej_open],0)
            all_lbl = torch.cat([rej_label, torch.ones(len(o_tok),1,device=DEVICE)],0)
            loss_rej = bce_rej(all_rej, all_lbl)
            loss = loss_cls + alpha*loss_rej + beta*loss_noc
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            bs=len(tokens); epoch_loss+=loss.item()*bs; epoch_cls+=loss_cls.item()*bs
            epoch_rej+=loss_rej.item()*bs; epoch_noc_l+=loss_noc.item()*bs
        n=len(train_ds); epoch_loss/=n
        val_f1,_,_,_ = evaluate(model, val_loader)
        val_em, val_macrec = evaluate_oracle_em(model, sel_loader)   # on DEV (balanced) if present
        sel = val_macrec                                 # SELECTION = macro-over-NOC oracle Recall@k
        scheduler.step(sel)
        history.append({"epoch":epoch,"loss":round(epoch_loss,4),"val_macro_f1":round(val_f1,4),
                        "val_oracle_em":round(val_em,4),"val_macro_recall":round(val_macrec,4)})
        if epoch%10==0 or epoch==1:
            print(f"  Ep {epoch:3d} | loss={epoch_loss:.4f} (cls={epoch_cls/n:.3f} rej={epoch_rej/n:.3f} noc={epoch_noc_l/n:.3f}) "
                  f"| val_macrec={val_macrec:.4f} val_em={val_em:.4f} val_f1={val_f1:.4f} | set_scale={float(model.set_scale):.3f} | lr={optimizer.param_groups[0]['lr']:.1e}")
        if sel>best_f1:
            best_f1,best_epoch,patience_count = sel,epoch,0
            torch.save(model.state_dict(), results_dir/"best_model.pt")
        else:
            patience_count+=1
            if patience_count>=patience:
                print(f"  Early stop ep {epoch} (best {best_epoch}, val macro-recall={best_f1:.4f})"); break
    print(f"Training done in {time.time()-t0:.1f}s")

    model.load_state_dict(torch.load(results_dir/"best_model.pt", weights_only=True))
    # ── Test: cardinality-head k* decode (deployable) + oracle ──────────────
    model.eval(); probs_list,card_list,yt_list,noc_list=[],[],[],[]
    with torch.no_grad():
        for tokens,mask,xflat,y,noc in test_loader:
            out=model(tokens.to(DEVICE),mask.to(DEVICE),xflat.to(DEVICE))
            probs_list.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
            card_list.append(out["logits_card"].cpu().numpy())
            yt_list.append(y.numpy()); noc_list.append(noc.numpy())
    P_te=np.concatenate(probs_list); card_te=np.concatenate(card_list)
    y_te=np.concatenate(yt_list); noc_te=np.concatenate(noc_list)
    # Val + train probs for post-hoc / two-stage cardinality
    vp,vy=[],[]
    with torch.no_grad():
        for tokens,mask,xflat,y,_ in val_loader:
            vp.append(torch.sigmoid(model(tokens.to(DEVICE),mask.to(DEVICE),xflat.to(DEVICE))["logits_cls"]).cpu().numpy()); vy.append(y.numpy())
    P_va=np.concatenate(vp); y_va=np.concatenate(vy)
    tp=[]
    with torch.no_grad():
        for tokens,mask,xflat,y,noc in train_loader:
            tp.append(torch.sigmoid(model(tokens.to(DEVICE),mask.to(DEVICE),xflat.to(DEVICE))["logits_cls"]).cpu().numpy())
    P_tr=np.concatenate(tp)

    # combo-invariant MAC/height + prob-profile features for two-stage decoder.
    # train = combo-diverse in-silico (multi-rich) -> stage2;  val = real prior -> stage1.
    tok_tr=train_ds.tokens.numpy(); msk_tr=train_ds.mask.numpy(); noc_tr=train_ds.noc.numpy()
    tok_va=val_ds.tokens.numpy();   msk_va=val_ds.mask.numpy();   noc_va=val_ds.noc.numpy()
    tok_te=test_ds.tokens.numpy();  msk_te=test_ds.mask.numpy()
    F_tr=card_features(P_tr,tok_tr,msk_tr); F_va=card_features(P_va,tok_va,msk_va); F_te=card_features(P_te,tok_te,msk_te)

    # pgNOC continuous-model (deconvolution) cost-curve features — global ref, the
    # deployable +pgNOC config (real-test 0.954 vs 0.950; lifts NOC4 .689->.733).
    print("  building pgNOC reference + cost-curve features (deconvolution) ...")
    G_ref=build_pgnoc_refs(train_ds.xflat.numpy(), train_ds.y.numpy(), noc_tr)
    Fp_tr=np.hstack([F_tr, pgnoc_cost_features(train_ds.xflat.numpy(), P_tr, G_ref)])
    Fp_va=np.hstack([F_va, pgnoc_cost_features(val_ds.xflat.numpy(),   P_va, G_ref)])
    Fp_te=np.hstack([F_te, pgnoc_cost_features(test_ds.xflat.numpy(),  P_te, G_ref)])
    pg_reg={"max_depth":4,"min_child_weight":10}

    k_card=card_te.argmax(1)+1
    k_post=posthoc_cardinality(P_va,y_va,P_te)
    k_two=two_stage_cardinality(F_tr,noc_tr,F_va,noc_va,F_te)
    k_pg=two_stage_cardinality(Fp_tr,noc_tr,Fp_va,noc_va,Fp_te,stage2_reg=pg_reg)
    em_card=per_noc_em(y_te,topk_decode(P_te,k_card),noc_te)
    em_post=per_noc_em(y_te,topk_decode(P_te,k_post),noc_te)
    em_two=per_noc_em(y_te,topk_decode(P_te,k_two),noc_te)
    em_pg=per_noc_em(y_te,topk_decode(P_te,k_pg),noc_te)
    oracle=per_noc_em(y_te,topk_decode(P_te,noc_te),noc_te)
    # two-stage + pgNOC features = deployable headline (selected on in-silico dev).
    decode_name="two_stage_pgnoc"; k_use=k_pg
    y_pred=topk_decode(P_te,k_use)
    te=full_report(y_te,y_pred,noc_te,"HYBRID — TEST (two-stage + pgNOC decode)")
    print(f"  {'decode':<16}{'overall':>8}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
    for nm,r in [("oracle",oracle),("joint-card",em_card),("post-hoc",em_post),
                 ("two-stage",em_two),("two-stage+pgNOC",em_pg)]:
        print(f"  {nm:<16}"+"".join(f"{x:>7.3f}" for x in r))
    oracle_em=oracle[0]; card_noc_acc=float((np.clip(k_use,1,5)==noc_te).mean())

    # reject AUROC
    model.eval(); scores,labels=[],[]
    with torch.no_grad():
        for tokens,mask,xflat,_,_ in test_loader:
            scores.append(torch.sigmoid(model(tokens.to(DEVICE),mask.to(DEVICE),xflat.to(DEVICE))["logit_reject"]).cpu().numpy()); labels.append(np.zeros(len(tokens)))
        for o_tok,o_mask,o_xf in DataLoader(open_ds,batch_size=256):
            scores.append(torch.sigmoid(model(o_tok.to(DEVICE),o_mask.to(DEVICE),o_xf.to(DEVICE))["logit_reject"]).cpu().numpy()); labels.append(np.ones(len(o_tok)))
    auroc=float(roc_auc_score(np.concatenate(labels), np.concatenate(scores).ravel()))
    print(f"\n  Reject AUROC: {auroc:.4f}  | final set_scale={float(model.set_scale):.4f}")

    np.save(results_dir/"y_test_pred.npy",y_pred); np.save(results_dir/"y_test_true.npy",y_te)
    json.dump({"model":"hybrid","config":cfg,"best_val_macro_recall":round(best_f1,4),"best_epoch":best_epoch,
               "decode":decode_name,"em_joint_card":round(float(em_card[0]),4),
               "em_post_hoc":round(float(em_post[0]),4),"em_two_stage":round(float(em_two[0]),4),
               "em_two_stage_pgnoc":round(float(em_pg[0]),4),
               "oracle_em":round(float(oracle_em),4),"card_noc_acc":round(card_noc_acc,4),
               "per_noc_oracle":{str(j):round(float(oracle[j]),4) for j in range(1,6)},
               "per_noc_two_stage":{str(j):round(float(em_two[j]),4) for j in range(1,6)},
               "per_noc_two_stage_pgnoc":{str(j):round(float(em_pg[j]),4) for j in range(1,6)},
               "reject_auroc":auroc,"set_scale":float(model.set_scale),
               "per_noc":te.get("per_noc",{}),"history":history,
               "test":{k:v for k,v in te.items() if k!="per_noc"}},
              open(results_dir/"metrics.json","w"), indent=2)
    print(f"\nSaved -> {results_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT/"configs"/"set_transformer.json"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out_subdir", type=str, default=None)
    ap.add_argument("--set_scale_fixed", type=float, default=None,
                    help="Fix set_scale (0.0 = flat-only ablation = MLP on Xflat)")
    ap.add_argument("--loss", type=str, default=None, choices=["bce", "asl"])
    ap.add_argument("--decouple_reject", action="store_true",
                    help="Separate PMA pool for reject head (decouple from donor-ID)")
    ap.add_argument("--n_token_feats", type=int, default=None,
                    help="3=baseline token; 8=enriched relational token (Increment 1, needs tokens8_*.npy)")
    ap.add_argument("--encoder", type=str, default=None, choices=["isab", "isab++"],
                    help="isab (LayerNorm) | isab++ (SetNorm + clean-path)")
    ap.add_argument("--num_embed", type=str, default=None, choices=["raw", "periodic"],
                    help="raw=scalar→shared Linear | periodic=per-feature PLR embedding (Gorishniy 2022)")
    ap.add_argument("--n_freq", type=int, default=None, help="periodic embedding frequencies per feature")
    ap.add_argument("--d_num_emb", type=int, default=None, help="periodic embedding dim per feature")
    ap.add_argument("--periodic_sigma", type=float, default=None, help="periodic frequency init std")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if args.epochs is not None: cfg["epochs"]=args.epochs
    if args.out_subdir is not None: cfg["out_subdir"]=args.out_subdir
    if args.set_scale_fixed is not None: cfg["set_scale_fixed"]=args.set_scale_fixed
    if args.loss is not None: cfg["loss"]=args.loss
    if args.decouple_reject: cfg["decouple_reject"]=True
    if args.n_token_feats is not None: cfg["n_token_feats"]=args.n_token_feats
    if args.encoder is not None: cfg["encoder"]=args.encoder
    if args.num_embed is not None: cfg["num_embed"]=args.num_embed
    if args.n_freq is not None: cfg["n_freq"]=args.n_freq
    if args.d_num_emb is not None: cfg["d_num_emb"]=args.d_num_emb
    if args.periodic_sigma is not None: cfg["periodic_sigma"]=args.periodic_sigma
    train(cfg)

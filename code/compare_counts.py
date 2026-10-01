"""
compare_counts.py — does the NoC LoRA hold up when the donor combinations under test are unseen?

One archive fold per run, on that fold's SAVED Phase 1 backbone (no Phase 1), every count scored on the
same real test set with the same combination groups:

  rf              post-hoc RandomForest on the sorted 45-donor probability profile, fit
                  leave-one-combination-out (the count before the LoRA; 0.8817 on fold 1, old code)
  clone           the pre-LoRA noc_branch: a full COPY of the backbone fine-tuned on the same 5 folds with
                  the published recipe (10 epochs noc_head @3e-4, then 15 epochs all @1e-5). 0.8507 on the
                  fold 1 test with the old code. The copy trains; `model` itself is never touched.
  clone_long      the same copy trained for --clone_steps (3000) steps, i.e. the LoRA budget. Fills the
                  missing cell: clone was only ever measured at ~255 steps and the LoRA at 3000, so
                  "clone beats LoRA" was confounded with how hard each was fine-tuned.
  lora_real       the current method: LoRA r8 fit 5-fold INSIDE the real test, combination-disjoint
                  (fold 1: 0.5115, fold 2: 0.609 on 2026-09-18 — reproduced here as a check)
  lora_sim        LoRA r8 fit on the fold's IN-SILICO train only (true NoC, ~33k combinations). No real
                  mixture is seen, so the whole test is combination-disjoint by construction
  lora_sim_real   lora_sim, then the lora_real 5-fold fine-tune on top (same folds, same recipe)
  ens_rf_sim      0.5 * rf + 0.5 * lora_sim        (weights fixed here, never searched)
  ens_rf_simreal  0.5 * rf + 0.5 * lora_sim_real

RULE, written 2026-09-18 before any of these ran: a candidate replaces rf as the count of the 10-fold
report only if its macro F1 beats rf on fold 1 AND is not below rf on fold 2; otherwise rf ships.
Steps, learning rates and weights are fixed below and are not tuned on any test number.
ID is untouched: every count reads the frozen backbone, and [ID GUARD] must print the main run's
fingerprints (fold 1: fp(logits_cls) 056326a272, fold 2: e102bc28c6).

EARLY STOPS (none of them can change the verdict):
  [STOP] fp(logits_cls) differs from the fold's known value -> wrong backbone/data, stop before any LoRA
  --skip_lora_real   lora_real is already measured (0.5115 / 0.609); skip its 20-min re-run
  --gate <cmp_fold1.json>  exit 10 when no candidate beats rf on fold 1: the rule is then already
                     decided and fold 2 need not run

usage:
  python compare_counts.py --fold 1 --data <fold dataset dir> --search /kaggle/input   # finds the stamped backbone
  python compare_counts.py --fold 1 --data <dir> --backbone <best_model.pt>
  python compare_counts.py --summarize cmp_fold1.json cmp_fold2.json
"""
import argparse
import copy
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SOURCES = ["rf", "clone", "clone_long", "lora_real", "lora_sim", "lora_sim_real",
           "ens_rf_clone", "ens_rf_clonelong", "ens_rf_sim", "ens_rf_simreal"]
EXPECT_FP = {1: "056326a272", 2: "e102bc28c6"}      # fp(logits_cls) of the 2026-09-18 main runs
RULE = ("candidate replaces rf only if macro F1 > rf on fold 1 AND >= rf on fold 2; otherwise rf ships "
        "(fixed 2026-09-18, before the run)")


# ── post-hoc RandomForest, verbatim from the pre-LoRA trainer (archive/2026-09-15_truoc_clean) ──────
def card_feats(P):
    s = np.sort(P, 1)[:, ::-1][:, :8]
    return np.concatenate([s, P.sum(1, keepdims=True), (P >= 0.5).sum(1, keepdims=True)], 1)


def rf_fit_predict(P_fit, y_fit, P_test, lam=0.02):
    """RF on the sorted probability profile, target = EM-optimal k. Returns (N, 5) class posteriors."""
    from sklearn.ensemble import RandomForestClassifier
    C = P_fit.shape[1]; tk = np.ones(len(P_fit), int)
    for i in range(len(P_fit)):
        K = max(int(y_fit[i].sum()), 1); best, bc = 1, 9e9
        for k in range(1, 6):
            yp = np.zeros(C); yp[np.argsort(P_fit[i])[::-1][:k]] = 1
            c = (y_fit[i] * (1 - yp)).sum() / K + ((1 - y_fit[i]) * yp).sum() / k + lam * k
            if c < bc:
                bc, best = c, k
        tk[i] = best
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42).fit(card_feats(P_fit), tk)
    out = np.zeros((len(P_test), 5))
    out[:, rf.classes_.astype(int) - 1] = rf.predict_proba(card_feats(P_test))
    return out


def rf_loco(P_te, y_te, cid, P_va, y_va):
    """Each combination is counted by an RF fit on the OTHER combinations + real val (the k=1 rows)."""
    mix = cid >= 0
    groups = [np.where(cid == c)[0] for c in np.unique(cid[mix])]
    if (~mix).any():
        groups.append(np.where(~mix)[0])
    P = np.zeros((len(P_te), 5))
    for g in groups:
        held = np.zeros(len(P_te), bool); held[g] = True
        fit = mix & ~held
        P[g] = rf_fit_predict(np.concatenate([P_te[fit], P_va]), np.concatenate([y_te[fit], y_va]), P_te[g])
    return P


# ── LoRA on the in-silico train (stage A) ────────────────────────────────────────────────────────────
def fit_lora_sim(model, tok, msk, y, steps, warm, rank, seed, dev):
    """Same recipe as the real fits (head warm-up at 3e-4, then LoRA 5e-4 / head 5e-5, cosine, clip 1.0,
    class-weighted CE, no selection), counted in steps: `warm` head-only steps, then `steps` steps."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, RandomSampler, TensorDataset
    from noc_lora import _NocLoraView, _train_pool_report
    m = copy.deepcopy(model).to(dev).enable_noc_lora(rank)
    net = _NocLoraView(m)
    w = 1.0 / np.sqrt(np.clip(np.bincount(y, minlength=6)[1:], 1, None))
    w = torch.tensor(np.clip(w / w.mean(), 0.6, 1.6), dtype=torch.float32, device=dev)
    ds = TensorDataset(torch.from_numpy(tok), torch.from_numpy(msk), torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)
    batches = lambda n: DataLoader(ds, batch_size=64, sampler=RandomSampler(ds, True, n * 64, generator=g))
    mon = np.random.RandomState(seed).choice(len(tok), min(2000, len(tok)), replace=False)   # TRAIN rows only

    def step(opt, tb, mb, nb, clip):
        loss = F.cross_entropy(net(tb.to(dev), mb.to(dev)), (nb - 1).to(dev), weight=w)
        opt.zero_grad(); loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
        opt.step()
        return loss.item()

    opt = torch.optim.AdamW(net.noc_head.parameters(), lr=3e-4, weight_decay=1e-4)
    net.train()
    for tb, mb, nb in batches(warm):
        step(opt, tb, mb, nb, False)
    opt = torch.optim.AdamW([{"params": list(net.noc_head.parameters()), "lr": 5e-5},
                             {"params": net.lora_params(), "lr": 5e-4}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
    for i, (tb, mb, nb) in enumerate(batches(steps), 1):
        net.train(); ls = step(opt, tb, mb, nb, True); sch.step()
        if i in (1, steps // 2, steps):
            net.eval()
            lr_, rec, mf1 = _train_pool_report(net, tok[mon], msk[mon], y[mon], w, dev)
            print(f"       [TRAIN lora_sim] step {i:>5}/{steps} batch loss {ls:.3f} | in-silico pool: loss {lr_:.3f} "
                  f"macroF1 {mf1:.3f} recall NOC1..5 " + " ".join(f"{r:.2f}" for r in rec), flush=True)
    net.eval()
    return m, net


def clone_kfold(model, tok, msk, noc, groups, K, seed, dev, warm_ep=10, full_ep=15, full_lr=1e-5,
                steps=None, open_tok=None, open_msk=None, tag="clone"):
    """noc_branch, verbatim recipe: per fold a full copy of the backbone is fine-tuned on the other folds'
    rows through logits_card (warm-up on noc_head, then everything), and scores its held-out rows. The
    copy is what trains; the ID model is untouched. `steps` overrides full_ep to a step budget."""
    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import StratifiedGroupKFold
    from torch.utils.data import DataLoader, TensorDataset
    y = np.clip(np.asarray(noc), 1, 5).astype(np.int64)
    folds = list(StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed).split(tok, y - 1, groups=groups))
    P = np.zeros((len(tok), 5)); Po = np.zeros((len(open_tok), 5)) if open_tok is not None else None

    @torch.no_grad()
    def predict(m, T_, M_, rows):
        out = np.zeros((len(rows), 5))
        for i in range(0, len(rows), 256):
            sl = rows[i:i + 256]
            out[i:i + 256] = m(torch.from_numpy(T_[sl]).to(dev), torch.from_numpy(M_[sl]).to(dev))["logits_card"].softmax(1).cpu().numpy()
        return out

    for fi, (tr, te) in enumerate(folds, 1):
        t0 = time.time()
        assert not (set(groups[tr].tolist()) & set(groups[te].tolist())), "combination leak"
        w = 1.0 / np.sqrt(np.clip(np.bincount(y[tr], minlength=6)[1:], 1, None))
        w = torch.tensor(np.clip(w / w.mean(), 0.6, 1.6), dtype=torch.float32, device=dev)
        g = torch.Generator().manual_seed(int(seed) * 100 + fi)
        dl = DataLoader(TensorDataset(torch.from_numpy(tok[tr]), torch.from_numpy(msk[tr]), torch.from_numpy(y[tr])),
                        batch_size=64, shuffle=True, generator=g)
        ep = full_ep if steps is None else max(1, int(round(steps / max(1, len(dl)))))
        m = copy.deepcopy(model).to(dev)

        def fit(params, lr, n_ep):
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_ep, 1), eta_min=1e-6)
            last = 0.0
            for _ in range(n_ep):
                m.train(); tot = 0.0
                for tb, mb, nb in dl:
                    ls = F.cross_entropy(m(tb.to(dev), mb.to(dev))["logits_card"], (nb - 1).to(dev), weight=w)
                    opt.zero_grad(); ls.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                    tot += ls.item()
                sch.step(); last = tot / max(len(dl), 1)
            return last
        for q in m.parameters():
            q.requires_grad = False
        head = list(m.cls_decoder_module.noc_head.parameters())
        for q in head:
            q.requires_grad = True
        fit(head, 3e-4, warm_ep)
        for q in m.parameters():
            q.requires_grad = True
        loss = fit(list(m.parameters()), full_lr, ep)
        m.eval()
        P[te] = predict(m, tok, msk, te)
        if Po is not None:
            Po += predict(m, open_tok, open_msk, np.arange(len(open_tok)))
        kt_ = P[te].argmax(1) + 1
        print(f"  [{tag}] fold {fi}/{K}: train {len(tr)} rows / {len(np.unique(groups[tr]))} combos, "
              f"{warm_ep} warm + {ep} epochs x {len(dl)} batches (~{ep * len(dl)} steps) | train loss {loss:.3f} | "
              f"held-out acc {float((kt_ == y[te]).mean()):.4f} | {time.time() - t0:.0f}s", flush=True)
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return P, (Po / K if Po is not None else None)


def score(kt, P, rank_te, y_true, T):
    from sklearn.metrics import confusion_matrix, f1_score
    kp = P.argmax(1) + 1
    f = f1_score(kt, kp, average=None, labels=[1, 2, 3, 4, 5], zero_division=0)
    rec = [float((kp[kt == v] == v).mean()) if (kt == v).any() else None for v in range(1, 6)]
    em = T.per_noc_em(y_true, T.topk_decode(rank_te, kp), kt)
    return {"macro_f1": round(float(f.mean()), 4), "accuracy": round(float((kp == kt).mean()), 4),
            "f1": [round(float(x), 4) for x in f], "recall": [None if r is None else round(r, 4) for r in rec],
            "em_at_k": round(float(em[0]), 4), "confusion": confusion_matrix(kt, kp, labels=[1, 2, 3, 4, 5]).tolist()}


def find_backbone(stamp, explicit, roots):
    if explicit:
        p = Path(explicit); p = p / "best_model.pt" if p.is_dir() else p
        st = p.with_suffix(".fold.json")
        if st.exists() and json.load(open(st)).get("combo_md5") != stamp["combo_md5"]:
            raise SystemExit(f"--backbone {p} was trained on another fold (stamp mismatch)")
        print(f"backbone {p} | stamp {'matches this fold' if st.exists() else 'MISSING (cannot be verified)'}")
        return p
    hits = sorted({Path(s).parent / "best_model.pt" for r in roots
                   for s in glob.glob(f"{r}/**/best_model.fold.json", recursive=True)
                   if json.load(open(s)).get("combo_md5") == stamp["combo_md5"]})
    hits = [h for h in hits if h.exists()]
    if not hits:
        raise SystemExit(f"no best_model.pt with this fold's stamp (combo_md5 {stamp['combo_md5']}) under {roots}. "
                         "Add the earlier fold run's output as an input, or pass --backbone.")
    print(f"backbone {hits[0]} | stamp matches this fold" + (f" ({len(hits)} found, using the first)" if len(hits) > 1 else ""))
    return hits[0]


def run(args):
    import kaggle_run_increment1 as K
    from fold_report import fold_stamp
    src = K.find_fold_dir(args.data)
    fi = json.load(open(src / "fold_info.json"))
    if fi.get("fold") != args.fold:
        raise SystemExit(f"--fold {args.fold} but {src}/fold_info.json says fold {fi.get('fold')}")
    bb = find_backbone(fold_stamp(src), args.backbone, args.search)
    if (src / "tokens8_dev.npy").exists():         # already prepared (dev split + tokens8): use as is
        data_w = src
    else:
        if not args.skip_prep:
            K.stage_prep_data_w(src)
        data_w = K.DATA_W
    os.environ["STR_DATA_DIR"] = str(data_w)
    import torch
    import phi_rerank as pr
    import train_set_transformer as T
    from fold_report import arrfp, open_labels, params_fp
    from noc_lora import _lora_predict, noc_lora_kfold
    dev, cfg = T.DEVICE, T.CFG
    code = hashlib.md5((HERE / "compare_counts.py").read_bytes()).hexdigest()[:10]
    print(f"\ncompare_counts CMP {code} | fold {args.fold} | unknown donors {fi['unknown_donors']} | device {dev}")
    print(f"RULE: {RULE}")

    test_ds, val_ds, train_ds, open_ds = (T.ClosedSetDataset("test"), T.ClosedSetDataset("val"),
                                          T.ClosedSetDataset("train"), T.OpenSetDataset())
    gp = T.DATA_DIR / "donor_geno.npy"
    dgeno = torch.from_numpy(np.load(gp).astype(np.float32)); dmask = torch.from_numpy(np.load(gp.parent / "donor_geno_mask.npy"))
    owner, _ = T.build_luts(dgeno, dmask, cfg["n_classes"])
    model = T.SetTransformerMixture(
        n_loci=cfg["n_loci"], d_locus=cfg["d_locus"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_isab=cfg["n_isab"], m_inducing=cfg["m_inducing"], n_classes=cfg["n_classes"], dropout=cfg["dropout"],
        n_token_feats=cfg["n_token_feats"], n_freq=cfg["n_freq"], d_num_emb=cfg["d_num_emb"],
        periodic_sigma=cfg["periodic_sigma"], n_slot_iters=cfg["n_slot_iters"], ot_eps=cfg["ot_eps"],
        ot_iters=cfg["ot_iters"], gumbel_temp=cfg["gumbel_temp"], donor_geno=dgeno, donor_geno_mask=dmask,
        owner_lut=owner.to(dev), noc_head_v2=cfg["noc_head_v2"]).to(dev)
    model.load_state_dict(torch.load(bb, map_location=dev, weights_only=True), strict=False)
    model.eval()

    @torch.no_grad()
    def probs(tok, msk):
        out = []
        for i in range(0, len(tok), 256):
            out.append(model(torch.from_numpy(tok[i:i + 256]).to(dev), torch.from_numpy(msk[i:i + 256]).to(dev))["logits_cls"].cpu().numpy())
        return np.concatenate(out)

    tt, tm = np.ascontiguousarray(test_ds.tokens.numpy()), np.ascontiguousarray(test_ds.mask.numpy())
    L_te = probs(tt, tm); P_te = 1 / (1 + np.exp(-L_te))
    y_te = test_ds.y.numpy(); noc = test_ds.noc.numpy(); kt = np.clip(noc, 1, 5)
    P_va = 1 / (1 + np.exp(-probs(val_ds.tokens.numpy(), val_ds.mask.numpy()))); y_va = val_ds.y.numpy()
    cid = np.load(T.DATA_DIR / "combo_id_test.npy").astype(int)
    PHt = pr.deconv_phi(tt, tm, dgeno.numpy(), dmask.numpy())
    rank_te, _, _ = T.loco_decode(L_te, y_te, noc, PHt, cid)
    print(f"  [ID GUARD] fp(logits_cls)={arrfp(L_te)}  fp(rank_te)={arrfp(rank_te)}  (must equal the fold's main run)")
    want = {"auto": EXPECT_FP.get(args.fold), "none": None}.get(args.expect_fp, args.expect_fp)
    if want and arrfp(L_te) != want:
        raise SystemExit(f"[STOP] fp(logits_cls) {arrfp(L_te)} != expected {want} for fold {args.fold}: wrong backbone "
                         "or data. Nothing else is run.")
    print(f"  [ID GUARD] expected fingerprint: {want or 'not checked'} -> OK")

    fp_params = params_fp(model)                 # the ID model must not move: clone/LoRA train on copies
    lab, why = open_labels(T.DATA_DIR, len(open_ds), HERE / "open_labels")
    ot, om = np.ascontiguousarray(open_ds.tokens.numpy()), np.ascontiguousarray(open_ds.mask.numpy())
    o_true = np.clip(lab[0], 1, 5) if lab is not None else None
    graw = [tuple(np.flatnonzero(r).tolist()) for r in y_te]
    gid = {g: i for i, g in enumerate(sorted(set(graw)))}; groups = np.array([gid[g] for g in graw])
    P, Po, sec = {}, {}, {}

    t0 = time.time()
    P["rf"] = rf_loco(P_te, y_te, cid, P_va, y_va)
    mixr = noc >= 2                        # open: one RF fit on every closed mixture + val (open shares no combination with them)
    Po["rf"] = rf_fit_predict(np.concatenate([P_te[mixr], P_va]), np.concatenate([y_te[mixr], y_va]),
                              1 / (1 + np.exp(-probs(ot, om))))
    sec["rf"] = time.time() - t0

    if args.with_clone:
        t0 = time.time(); T.set_seed(args.seed)
        print(f"\n  === clone: full backbone copy, published recipe (10 ep noc_head @3e-4, {args.clone_full_ep} ep "
              f"all @{args.clone_full_lr:.0e}), same {args.lora_kfold} combination-disjoint folds ===")
        P["clone"], Po["clone"] = clone_kfold(model, tt, tm, noc, groups, args.lora_kfold, args.seed, dev,
                                              full_ep=args.clone_full_ep, full_lr=args.clone_full_lr,
                                              open_tok=ot, open_msk=om, tag="clone")
        sec["clone"] = time.time() - t0
    if args.with_clone_long:
        t0 = time.time(); T.set_seed(args.seed)
        print(f"\n  === clone_long: the same copy at the LoRA budget ({args.clone_steps} steps @{args.clone_full_lr:.0e}) ===")
        P["clone_long"], Po["clone_long"] = clone_kfold(model, tt, tm, noc, groups, args.lora_kfold, args.seed, dev,
                                                        full_lr=args.clone_full_lr, steps=args.clone_steps,
                                                        open_tok=ot, open_msk=om, tag="clone_long")
        sec["clone_long"] = time.time() - t0
    if args.skip_lora_real:
        print("\n  lora_real skipped (--skip_lora_real): measured 2026-09-18, fold 1 0.5115 / fold 2 0.609")
    else:
        t0 = time.time(); T.set_seed(args.seed)      # identical to the trainer -> reproduces the main run
        _, P["lora_real"], Po["lora_real"] = noc_lora_kfold(model, tt, tm, noc, groups, args.lora_kfold, args.lora_steps,
                                                            cfg["lora_rank"], args.seed, dev, ot, om)
        sec["lora_real"] = time.time() - t0

    if args.skip_lora_sim:
        print("\n  lora_sim / lora_sim_real skipped (--skip_lora_sim)")
    else:
        t0 = time.time(); T.set_seed(args.seed)
        ys = np.clip(train_ds.noc.numpy(), 1, 5).astype(np.int64)
        print(f"\n  === lora_sim: LoRA r{cfg['lora_rank']} on the in-silico train only ({len(ys)} rows, NOC "
              + str({v: int((ys == v).sum()) for v in range(1, 6)}) + f"), {args.sim_warm} warm-up + {args.sim_steps} steps ===")
        m_sim, net_sim = fit_lora_sim(model, np.ascontiguousarray(train_ds.tokens.numpy()),
                                      np.ascontiguousarray(train_ds.mask.numpy()), ys, args.sim_steps, args.sim_warm,
                                      cfg["lora_rank"], args.seed, dev)
        P["lora_sim"] = _lora_predict(net_sim, tt, tm, np.arange(len(tt)), dev)
        Po["lora_sim"] = _lora_predict(net_sim, ot, om, np.arange(len(ot)), dev)
        sec["lora_sim"] = time.time() - t0

        t0 = time.time(); T.set_seed(args.seed)
        print("\n  === lora_sim_real: lora_sim + the same 5-fold real fine-tune ===")
        _, P["lora_sim_real"], Po["lora_sim_real"] = noc_lora_kfold(m_sim, tt, tm, noc, groups, args.lora_kfold,
                                                                    args.lora_steps, cfg["lora_rank"], args.seed, dev, ot, om)
        sec["lora_sim_real"] = time.time() - t0
    for e, s_ in (("ens_rf_clone", "clone"), ("ens_rf_clonelong", "clone_long"),
                  ("ens_rf_sim", "lora_sim"), ("ens_rf_simreal", "lora_sim_real")):
        if s_ in P:                                   # 0.5/0.5, weight fixed in advance, never searched
            P[e] = 0.5 * P["rf"] + 0.5 * P[s_]; Po[e] = 0.5 * Po["rf"] + 0.5 * Po[s_]

    res = {"schema": "compare_counts/v1", "cmp_code": code, "fold": args.fold, "unknown_donors": fi["unknown_donors"],
           "seed": args.seed, "backbone": str(bb), "fp_logits_cls": arrfp(L_te), "rule": RULE,
           "recipe": {"lora_kfold": args.lora_kfold, "lora_steps": args.lora_steps, "sim_steps": args.sim_steps,
                      "sim_warm": args.sim_warm, "rank": cfg["lora_rank"], "clone_full_ep": args.clone_full_ep,
                      "clone_full_lr": args.clone_full_lr, "clone_steps": args.clone_steps},
           "per_noc_n": {str(v): int((kt == v).sum()) for v in range(1, 6)}, "sources": {}}
    print(f"\n  === fold {args.fold}: every count on the same {len(kt)} test profiles ===")
    print(f"  {'source':<16}{'macroF1':>8}{'acc':>7}   F1 NOC1..5{'':<22}{'EM@k':>6}{'open acc':>10}")
    for s in [x for x in SOURCES if x in P]:
        r = score(kt, P[s], rank_te, y_te, T)
        if o_true is not None:
            r["open_accuracy"] = round(float(((Po[s].argmax(1) + 1) == o_true).mean()), 4)
        r["seconds"] = round(sec.get(s, 0.0), 1)
        res["sources"][s] = r
        print(f"  {s:<16}{r['macro_f1']:>8.4f}{r['accuracy']:>7.3f}   " + " ".join(f"{x:.3f}" for x in r["f1"])
              + f"{r['em_at_k']:>9.3f}{r.get('open_accuracy', float('nan')):>10.3f}")
    out = (Path(args.out) if args.out else Path(f"/kaggle/working/cmp_fold{args.fold}.json")
           if Path("/kaggle/working").exists() else HERE / f"cmp_fold{args.fold}.json")
    json.dump(res, open(out, "w"), indent=1)
    same = params_fp(model) == fp_params
    print(f"  [ID GUARD] backbone params unchanged after every count: {same}")
    if not same:
        raise SystemExit("[STOP] the ID backbone moved during counting — results are not valid")
    print(f"\n  -> {out}")


def gate(path):
    """After fold 1: which candidates still beat rf. None left -> the rule is decided, exit 10."""
    r = json.load(open(path)); rf = r["sources"]["rf"]["macro_f1"]
    alive = [s for s in SOURCES[1:] if s in r["sources"] and r["sources"][s]["macro_f1"] > rf]
    print(f"[GATE] fold {r['fold']}: rf {rf:.4f} | beat rf: "
          + (", ".join(f"{s} {r['sources'][s]['macro_f1']:.4f}" for s in alive) if alive else "none"))
    if not alive:
        print("[GATE] no candidate beats rf on fold 1 -> rule decided (rf ships); later folds are not needed")
        sys.exit(10)


def summarize(paths):
    R = {r["fold"]: r for r in (json.load(open(p)) for p in paths)}
    print(f"RULE: {RULE}\n")
    print(f"  {'source':<16}" + "".join(f"{'fold ' + str(f):>10}" for f in sorted(R)))
    for s in SOURCES:
        print(f"  {s:<16}" + "".join(f"{R[f]['sources'][s]['macro_f1']:>10.4f}" if s in R[f]["sources"]
                                     else f"{'—':>10}" for f in sorted(R)))
    if not {1, 2} <= set(R):
        print("\n  verdict needs folds 1 and 2"); return
    rf1, rf2 = R[1]["sources"]["rf"]["macro_f1"], R[2]["sources"]["rf"]["macro_f1"]
    win = [s for s in SOURCES[1:] if s in R[1]["sources"] and s in R[2]["sources"]
           and R[1]["sources"][s]["macro_f1"] > rf1 and R[2]["sources"][s]["macro_f1"] >= rf2]
    print(f"\n  VERDICT: " + (f"{', '.join(win)} pass the rule -> may replace rf" if win else "no candidate passes -> rf ships"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", nargs="+", default=None, help="cmp_fold*.json files to pool")
    ap.add_argument("--gate", type=str, default=None, help="cmp_fold1.json: exit 10 if no candidate beats rf")
    ap.add_argument("--expect_fp", type=str, default="auto", help="auto (known per fold) | none | <10-hex>")
    ap.add_argument("--skip_lora_real", action="store_true", help="do not re-run the already measured lora_real")
    ap.add_argument("--skip_lora_sim", action="store_true", help="skip lora_sim / lora_sim_real")
    ap.add_argument("--with_clone", action="store_true", help="noc_branch: full backbone copy, published recipe")
    ap.add_argument("--with_clone_long", action="store_true", help="the same copy at --clone_steps (LoRA budget)")
    ap.add_argument("--clone_full_ep", type=int, default=15, help="published FULL-stage epochs (15 @1e-5)")
    ap.add_argument("--clone_full_lr", type=float, default=1e-5)
    ap.add_argument("--clone_steps", type=int, default=3000)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--data", type=str)
    ap.add_argument("--backbone", type=str, default=None, help="this fold's best_model.pt (default: search by stamp)")
    ap.add_argument("--search", nargs="+", default=["/kaggle/input"], help="where to look for a stamped backbone")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_kfold", type=int, default=5)
    ap.add_argument("--lora_steps", type=int, default=3000)
    ap.add_argument("--sim_steps", type=int, default=3000)
    ap.add_argument("--sim_warm", type=int, default=300)
    ap.add_argument("--skip_prep", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    if a.gate:
        gate(a.gate)
    elif a.summarize:
        summarize(a.summarize)
    else:
        run(a)

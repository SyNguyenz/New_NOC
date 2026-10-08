"""decode_layer.py - the test-time decode: who is in the sample and how many.

Three steps on top of the trained model, every parameter chosen on the validation split (real NOC1 val + combo-disjoint in-silico mixtures; no test sample):

1. PENALTY + BONUS on the gate: sample-level corrections over all donors, each with its own examination threshold.
   PENALTY  u_c = the largest share one peak holds of the positive contributions to c's gate (gradient x input on a
            per-peak weight multiplying the peak embedding). If u_c >= u_min that peak is the anchor: it fixes c's
            level, so how many of c's other alleles should stand next to the majors (the other donors with
            gate > .5) follows; p_c = how plausible what stands is (an anchor that is not c's allele: p_c = 0).
            logit_c -= a * u_c * (1 - p_c)
   BONUS    f2_c = c's mixture proportion in a joint fit with the majors, relative to the weakest major (the main
            filter: f2_c >= f2_min); f1_c = c's alleles standing beyond the coverage the sample gives any panel donor
            (a weight only); f3_c = c's gate relative to the best examined donor.
            logit_c += b * max(f1_c, .05) * f2_c * f3_c
   The count is k = round(sum of the corrected gates), clipped to 1..5.
2. ORDER: the set-conditional greedy. Each pick maximises its rerank score + 1.5 * (height it newly explains, over
   the profile's median peak) / its allele count; the rerank score is the phi-rerank of the model's logits.
3. DROP: the k-th pick is dropped when the height it newly explains is <= t.
4. OPEN SCORE: does the sample hold a person outside the panel? Nothing is trained on open samples; the score comes
   from how well the closed-trained model and the decoded set fit the panel: the attr head's background share of the
   height, the height on alleles no panel donor carries, the allele height the decoded set leaves unexplained, the gate
   total, the 6th largest gate and how weak the k-th gate is. Each is standardised on val and only its excess in
   the open direction counts (score = sum of max(0, z)); the sample is flagged above the 95th percentile of val.

Measured on dgw_ref (fold 0, parameters then chosen on the in-silico DEV split): DEV macro ID .873; real test macro count .950, macro ID .933 (NOC2-5 .961 / .974 /
.967 / .828).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import nnls

import gen_law as gl
import phi_rerank as pr

GREEDY_STEPS = 6
HEIGHT_W = 1.5                 # weight on newly explained height in the greedy
F1_FLOOR = 0.05                # smallest coverage weight a bonus candidate carries
GRID_U = (0.05, 0.075, 0.1, 0.15)
GRID_A = (8.0, 16.0, 32.0, 64.0)
GRID_F2 = (0.1, 0.3, 0.5, 0.9)
GRID_B = (16.0, 32.0, 64.0, 128.0)
GRID_ALPHA = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
OPEN_ACCEPT = 0.95             # share of val the open threshold accepts


def _key(locus, allele):
    return int(round(float(locus))), int(round(float(allele) * 10))


# ── model outputs ────────────────────────────────────────────────────────────
def model_outputs(model, tokens, mask, device, bs=48):
    """logits_cls, gate, per donor the gate-contribution imbalance u with its anchor token (-1 if none), and the attr
    head's background share of each sample's height.

    The contribution of peak t to donor c is d gate_logit_c / d w_t, with w a per-peak weight (= 1) multiplying the
    output of model.input_proj: one forward and one backward per donor, no peak is removed."""
    model.eval()
    holder = {"w": None}

    def scale(_m, _i, o):
        w = holder["w"]
        return o * w if (w is not None and o.dim() == 3 and o.shape[:2] == w.shape[:2]) else o

    hook = model.input_proj.register_forward_hook(scale)
    L, G, U, A, BG = [], [], [], [], []
    try:
        for s in range(0, len(tokens), bs):
            t = torch.as_tensor(tokens[s:s + bs], dtype=torch.float32, device=device)
            m = torch.as_tensor(mask[s:s + bs], device=device)
            w = torch.ones(t.shape[0], t.shape[1], 1, device=device, requires_grad=True)
            holder["w"] = w
            o = model(t, m); glog = o["gate_logit"]; n_cls = glog.shape[1]
            cc = torch.stack([torch.autograd.grad(glog[:, c].sum(), w, retain_graph=c < n_cls - 1)[0][..., 0]
                              for c in range(n_cls)], 1)                              # (B, C, tokens)
            pos = cc.clamp(min=0) * m[:, None, :].to(cc.dtype)
            tot = pos.sum(-1); mx, arg = pos.max(-1)
            U.append(torch.where(tot > 0, mx / tot.clamp(min=1e-30), torch.zeros_like(tot)).detach().cpu().numpy())
            A.append(torch.where(tot > 0, arg, torch.full_like(arg, -1)).cpu().numpy())
            L.append(o["logits_cls"].detach().cpu().numpy()); G.append(torch.sigmoid(glog).detach().cpu().numpy())
            h = torch.expm1(t[:, :, 2]).clamp(min=0) * m.to(t.dtype)
            bg = torch.softmax(o["logits_attr"].detach(), -1)[..., -1]
            BG.append(((bg * h).sum(1) / h.sum(1).clamp(min=1e-6)).cpu().numpy())
            holder["w"] = None
    finally:
        hook.remove()
    return np.concatenate(L), np.concatenate(G), np.concatenate(U), np.concatenate(A), np.concatenate(BG)


# ── sample-level coefficients ────────────────────────────────────────────────
class _Law:
    def __init__(self, law):
        self.law = law; self.own = law["own"]; self.dos = law["dos"]; self.bl = np.asarray(law["bin_locus"], int)
        self.ix = law["index"]; self.thr = gl.STAND_THR; self.pa, self.pb = law["pres"]

    def pres(self, e):
        return 1.0 / (1.0 + np.exp(-(self.pa + self.pb * np.log(np.maximum(e, 1.0)))))

    def residual(self, h, acc):
        """profile minus each accepted donor's alleles at its per-locus level and the stutter they cast"""
        own, dos, bl, thr = self.own, self.dos, self.bl, self.thr
        R = h.copy()
        for c in acc:
            stand = own[c] & (R > thr)
            lv = np.log(np.maximum(R[stand], 1.0) / np.maximum(dos[c][stand], 1.0))
            gmed = float(np.exp(np.median(lv))) if len(lv) else 1.0
            contrib = np.zeros(len(h))
            for L in np.unique(bl[own[c]]):
                bins = np.flatnonzero(own[c] & (bl == L)); sb = bins[stand[bins]]
                level = (float(np.exp(np.mean(np.log(np.maximum(R[sb], 1.0) / np.maximum(dos[c][sb], 1.0)))))
                         if len(sb) else gmed)
                contrib[bins] = np.minimum(R[bins], level * np.maximum(dos[c][bins], 1.0))
            R = np.maximum(R - contrib - gl.stutter_of(contrib, self.law), 0.0)
        return R

    def implausibility(self, h, others, c, anchor):
        """share of c's expected-present alleles (level read off the anchor, on the residual after `others`) that
        are missing"""
        R = self.residual(h, others) if others else h.copy()
        lvl = max(R[anchor], self.thr) / max(self.dos[c][anchor], 1.0)
        rest = np.flatnonzero(self.own[c]); rest = rest[rest != anchor]
        if not len(rest):
            return np.nan
        p = self.pres(lvl * np.maximum(self.dos[c][rest], 1.0)); miss = R[rest] <= self.thr
        return float((p * miss).sum() / max(p.sum(), 1e-9))

    def joint_fit(self, h_, members):
        """mixture proportions of `members` from the allele heights (stutter removed), per-locus shares, NNLS"""
        own, dos, bl, thr = self.own, self.dos, self.bl, self.thr
        allele = own[members].any(0)
        h = np.maximum(h_ - gl.stutter_of(h_ * allele, self.law), 0.0) * allele
        rows, rhs, wts = [], [], []
        for L in np.unique(bl[allele]):
            b = np.flatnonzero(allele & (bl == L)); tot = h[b].sum()
            if tot <= thr:
                continue
            for q in b:
                rows.append([dos[c][q] / 2.0 for c in members]); rhs.append(h[q] / tot); wts.append(np.sqrt(tot))
        if not rows:
            return np.zeros(len(members))
        w = np.array(wts); A = np.array(rows) * w[:, None]; y = np.array(rhs) * w
        lam = 10.0 * np.sqrt(np.mean(w ** 2))
        phi, _ = nnls(np.vstack([A, lam * np.ones(len(members))]), np.r_[y, lam])
        return phi


def coefficients(tokens, mask, G, U, anchor, law):
    """plausibility P (penalty), coverage F1 and joint-phi F2 (bonus), per sample and donor"""
    lw = _Law(law); own = lw.own
    H = gl.profile_heights(tokens, mask, law)
    N, C = G.shape; P = np.ones((N, C)); F1 = np.zeros((N, C)); F2 = np.zeros((N, C))
    nal = np.maximum(own.sum(1), 1).astype(float); u_floor = min(GRID_U)
    for i in range(N):
        majors = [int(c) for c in np.flatnonzero(G[i] > .5)]
        cov = (own & (H[i] > lw.thr)[None, :]).sum(1) / nal
        q_bg = float(np.median([cov[c] for c in range(C) if c not in majors]))
        for c in range(C):
            others = [m for m in majors if m != c]
            if U[i, c] >= u_floor and anchor[i, c] >= 0:
                b = lw.ix.get(_key(tokens[i, anchor[i, c], 0], tokens[i, anchor[i, c], 1]), -1)
                if b < 0 or not own[c][b]:
                    P[i, c] = 0.0                                # the gate leans on a peak that is not c's allele
                else:
                    imp = lw.implausibility(H[i], others, c, b)
                    P[i, c] = 1.0 - (imp if np.isfinite(imp) else 0.0)
            F1[i, c] = float(np.clip((cov[c] - q_bg) / max(1 - q_bg, 1e-6), 0, 1))
            if others:
                phi = lw.joint_fit(H[i], others + [c])
                F2[i, c] = float(np.clip(phi[-1] / max(float(phi[:-1].min()), 1e-6), 0, 1))
            else:
                F2[i, c] = 1.0
    return P, F1, F2


def corrected_gates(G, U, P, F1, F2, u_min, a, f2_min, b):
    L = np.log(np.clip(G, 1e-6, 1 - 1e-6) / np.clip(1 - G, 1e-6, 1))
    if a:
        L = L - a * np.where(U >= u_min, U * (1 - P), 0.0)
    if b:
        bm = F2 >= f2_min
        gx = np.where(bm, G, 0.0); f3 = gx / np.maximum(gx.max(1, keepdims=True), 1e-9)
        L = L + b * np.where(bm, np.maximum(F1, F1_FLOOR) * F2 * f3, 0.0)
    return 1.0 / (1.0 + np.exp(-L))


# ── order and count ──────────────────────────────────────────────────────────
def greedy(tokens, mask, R, dg, dgm, steps=GREEDY_STEPS):
    """set-conditional greedy: picks (N, steps) and the height each pick newly explains (N, steps)"""
    C = R.shape[1]
    key = [{_key(dg[c, j, 0], dg[c, j, 1]) for j in range(dg.shape[1]) if dgm[c, j]} for c in range(C)]
    ng = np.array([max(len(k_), 1) for k_ in key], float)
    N = len(tokens); picks = np.zeros((N, steps), int); gains = np.zeros((N, steps))
    for i in range(N):
        hm = {_key(tokens[i, p, 0], tokens[i, p, 1]): float(np.expm1(tokens[i, p, 2])) for p in np.flatnonzero(mask[i])}
        obs = sorted(hm); ix = {a: j for j, a in enumerate(obs)}
        w = np.array([hm[a] for a in obs], float)
        w = w / max(float(np.median(w)), 1e-9) if len(w) else w
        D = np.zeros((C, len(obs)), bool)
        for c in range(C):
            for a in key[c] & set(obs):
                D[c, ix[a]] = True
        picked, ex = [], np.zeros(len(obs), bool)
        for st in range(steps):
            gain = ((D & ~ex) * w).sum(1) / ng
            s = R[i] + (HEIGHT_W * gain if st else 0.0)
            s[picked] = -np.inf
            c = int(np.argmax(s)); picked.append(c); ex |= D[c]
            picks[i, st] = c; gains[i, st] = gain[c]
    return picks, gains


def drop_last(k, gains, t):
    """k -> k-1 when the k-th pick newly explains <= t (only for k >= 2)"""
    k = np.asarray(k).astype(int)
    return np.where((k >= 2) & (gains[np.arange(len(k)), np.clip(k, 1, gains.shape[1]) - 1] <= t), k - 1, k)


def to_sets(picks, k, n_cls):
    yp = np.zeros((len(k), n_cls), int)
    for i in range(len(k)):
        yp[i, picks[i, :int(np.clip(k[i], 1, 5))]] = 1
    return yp


def macro_id(yp, y, noc):
    """exact-set ID, macro over NOC2-5"""
    ok = (yp == (y > .5)).all(1)
    return float(np.mean([ok[noc == j].mean() for j in (2, 3, 4, 5) if (noc == j).any()]))


# ── the whole decode ─────────────────────────────────────────────────────────
class Split:
    """one split's inputs, model outputs and coefficients (y / noc may be None at deployment)"""

    def __init__(self, model, tokens, mask, y, noc, dg, dgm, law, device):
        self.tokens = np.asarray(tokens, np.float32); self.mask = np.asarray(mask).astype(bool)
        self.y = None if y is None else np.asarray(y)
        self.noc = None if noc is None else np.clip(np.asarray(noc).astype(int), 1, 5)
        self.L, self.G, self.U, self.anchor, self.bg = model_outputs(model, self.tokens, self.mask, device)
        self.law = law
        self.phi = pr.deconv_phi(self.tokens, self.mask, dg, dgm)
        self.P, self.F1, self.F2 = coefficients(self.tokens[:, :, :3], self.mask, self.G, self.U, self.anchor, law)
        self.dg, self.dgm = dg, dgm
        self.picks = self.gains = None

    def order(self, alpha):
        self.picks, self.gains = greedy(self.tokens, self.mask, pr.rerank_scores(self.L, self.phi, alpha),
                                        self.dg, self.dgm)

    def predict(self, params, t_drop):
        G2 = corrected_gates(self.G, self.U, self.P, self.F1, self.F2, *params)
        k = drop_last(np.clip(np.rint(G2.sum(1)), 1, 5), self.gains, t_drop)
        return to_sets(self.picks, k, self.G.shape[1]), k


def tune_alpha(dev):
    """phi-rerank alpha: exact ID of the top-true-k on val mixtures"""
    mix = dev.noc >= 2; best, bs = GRID_ALPHA[0], -1.0
    for a in GRID_ALPHA:
        r = pr.rerank_scores(dev.L[mix], dev.phi[mix], a); yy, nn = dev.y[mix], dev.noc[mix]
        ok = np.mean([set(np.argsort(-r[i])[:nn[i]]) == set(np.flatnonzero(yy[i] > .5)) for i in range(len(r))])
        if ok > bs:
            best, bs = a, ok
    return float(best)


def tune_drop(dev):
    """t for the drop rule, on the uncorrected gate count: maximise val ID over a grid of val quantiles"""
    kg = np.rint(dev.G.sum(1))
    grid = np.unique(np.quantile(dev.gains[:, 1:], np.linspace(0.01, 0.99, 300)))

    def id_at(t):
        k = np.clip(drop_last(kg, dev.gains, t), 1, 5)
        return np.mean([set(dev.picks[i, :k[i]]) == set(np.flatnonzero(dev.y[i] > .5)) for i in range(len(k))])
    return float(max(grid, key=id_at))


def tune_corrections(dev, t_drop):
    grid = [(u, a, f, b) for u in GRID_U for a in GRID_A for f in GRID_F2 for b in GRID_B]
    return max(grid, key=lambda p: macro_id(dev.predict(p, t_drop)[0], dev.y, dev.noc))


def fit(dev, log=print):
    """choose every decode parameter on val; returns the config apply() takes"""
    alpha = tune_alpha(dev); dev.order(alpha)
    t_drop = tune_drop(dev)
    params = tune_corrections(dev, t_drop)
    yp, _ = dev.predict(params, t_drop)
    cfg = {"alpha": alpha, "t_drop": t_drop, "u_min": params[0], "a": params[1], "f2_min": params[2], "b": params[3],
           "val_macro_id": round(macro_id(yp, dev.y, dev.noc), 4)}
    log(f"decode layer (chosen on val): {cfg}")
    return cfg


def apply(split, cfg):
    """(N, C) predicted sets and the counts"""
    split.order(cfg["alpha"])
    return split.predict((cfg["u_min"], cfg["a"], cfg["f2_min"], cfg["b"]), cfg["t_drop"])


# ── open score ───────────────────────────────────────────────────────────────
def open_features(split, k):
    """(N, 6) panel-fit features, each larger when a person outside the panel is present: attr background share,
    off-panel height share, unexplained after the decoded set, sum of gates, 6th largest gate, -log k-th gate"""
    law = split.law; lw = _Law(law); T, M, G = split.tokens, split.mask, split.G
    panel = {_key(split.dg[c, j, 0], split.dg[c, j, 1]) for c in range(split.dg.shape[0])
             for j in range(split.dg.shape[1]) if split.dgm[c, j]}
    H = gl.profile_heights(T[:, :, :3], M, law)
    N = len(T); off = np.zeros(N); unex = np.zeros(N)
    for i in range(N):
        p = np.flatnonzero(M[i]); h = np.expm1(T[i, p, 2].astype(np.float64))
        out = np.array([_key(T[i, q, 0], T[i, q, 1]) not in panel for q in p], bool)
        off[i] = h[out].sum() / max(h.sum(), 1e-6)
        S = [int(c) for c in split.picks[i, :int(np.clip(k[i], 1, 5))]]
        unex[i] = lw.residual(H[i], S)[law["carried"]].sum() / max(H[i].sum(), 1e-6)
    gs = np.sort(G, 1)[:, ::-1]; kg = np.clip(np.rint(G.sum(1)), 1, 5).astype(int)
    kth = -np.log(np.clip(gs[np.arange(N), kg - 1], 1e-6, 1.0))
    return np.c_[split.bg, off, unex, G.sum(1), gs[:, 5], kth]


def fit_open(dev, cfg):
    """standardisation and threshold of the open score, on val (call after fit)"""
    _, k = apply(dev, cfg)
    X = open_features(dev, k); mu, sd = X.mean(0), X.std(0) + 1e-9
    sc = np.maximum((X - mu) / sd, 0.0).sum(1)
    cfg.update(open_mu=mu.tolist(), open_sd=sd.tolist(), open_thr=float(np.quantile(sc, OPEN_ACCEPT)))
    return cfg


def open_score(split, cfg, k):
    """score and flag (score > threshold) per sample; k = the decoded counts from apply()"""
    X = open_features(split, k)
    sc = np.maximum((X - np.asarray(cfg["open_mu"])) / np.asarray(cfg["open_sd"]), 0.0).sum(1)
    return sc, sc > cfg["open_thr"]


# ── metrics ──────────────────────────────────────────────────────────────────
def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, (2 * p * r / max(p + r, 1e-12))


def multiclass_report(y, yp, classes):
    """per-class precision / recall / F1 / support, macro (unweighted mean over classes) and micro (pooled)"""
    per = {}
    for c in classes:
        tp = int(((yp == c) & (y == c)).sum()); fp = int(((yp == c) & (y != c)).sum()); fn = int(((yp != c) & (y == c)).sum())
        pr, rc, f = prf(tp, fp, fn); per[str(c)] = dict(precision=pr, recall=rc, f1=f, support=int((y == c).sum()))
    present = [per[str(c)] for c in classes if per[str(c)]["support"] > 0]
    macro = {m: float(np.mean([d[m] for d in present])) for m in ("precision", "recall", "f1")}
    tp = sum(int(((yp == c) & (y == c)).sum()) for c in classes)
    fp = sum(int(((yp == c) & (y != c)).sum()) for c in classes); fn = sum(int(((yp != c) & (y == c)).sum()) for c in classes)
    mp, mr, mf = prf(tp, fp, fn)
    return dict(per_class=per, macro=macro, micro=dict(precision=mp, recall=mr, f1=mf))


def multilabel_report(Y, Yp):
    """donor identification as multi-label: micro (pooled over sample x donor) and macro (mean over donors present)"""
    Y = Y > .5; Yp = Yp > .5
    tp = (Y & Yp).sum(0); fp = (~Y & Yp).sum(0); fn = (Y & ~Yp).sum(0)
    mp, mr, mf = prf(int(tp.sum()), int(fp.sum()), int(fn.sum()))
    per = [prf(int(a), int(b), int(c)) for a, b, c in zip(tp, fp, fn)]
    present = [d for d, n in zip(per, Y.sum(0)) if n > 0]
    macro = dict(precision=float(np.mean([d[0] for d in present])), recall=float(np.mean([d[1] for d in present])),
                 f1=float(np.mean([d[2] for d in present])))
    return dict(micro=dict(precision=mp, recall=mr, f1=mf), macro=macro)

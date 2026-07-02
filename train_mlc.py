"""Inc17 — Meta-Learning for Compositionality (MLC, Lake & Baroni Nature 2023) for donor-in-mixture ID.
Each EPISODE: a donor POOL; SUPPORT mixtures (labelled) -> per-donor episode PROTOTYPE (mean pooled-z over
support mixtures containing that donor); QUERY mixtures from COMBO-DISJOINT combos of the same pool ->
the per-donor decoder is CONDITIONED on the prototype to identify donors in the NOVEL combo. Meta-training
over many episodes teaches the SKILL of generalizing to novel combinations (not memorizing combos).
proto_proj zero-init => at start the prototype offset is 0 => reduces to the base decoder (warm-startable).
Inference: build the prototype from REAL VAL (deployable), evaluate REAL TEST.

Usage: python train_mlc.py [--warm_start results/inc6_maskp_seed42/best_model.pt] [--episodes 30000] [--seed 42] [--out_subdir inc17_mlc]
"""
import os, sys, json, shutil, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, argparse
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get("STR_DATA_DIR", str(ROOT/"data_insilico_w")))   # local default; overwritten on Kaggle
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
from train_set_transformer import AsymmetricLoss

def L(n): return np.load(DATA/f"{n}.npy", allow_pickle=True)

def prepare_data():
    """Standalone Kaggle data prep (mirrors kaggle_run_increment1.py steps 1-2b): copy the read-only input
    dataset to a WRITABLE dir, carve the combo-disjoint DEV split, build enriched tokens8_*. Runs only when
    INSILICO_W (the Kaggle input) is set AND tokens8 isn't already prepared. Sets the global DATA dir."""
    global DATA
    src = os.environ.get("INSILICO_W")
    if not src:                                              # local: data_insilico_w already has tokens8
        print(f"local data dir: {DATA}"); return
    SRC = Path(src); WORK = Path(os.environ.get("WORK_DIR", "/kaggle/working")); DW = WORK / "data_w"
    if (DW / "tokens8_train.npy").exists():
        DATA = DW; print(f"data already prepared at {DW}"); return
    DW.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob("*.npy"):
        if not (DW / f.name).exists(): shutil.copy(f, DW / f.name)
    for f in SRC.glob("*.json"): shutil.copy(f, DW / f.name)
    for nm in ("donor_geno.npy", "donor_geno_mask.npy"):     # shipped in the code bundle root
        if not (DW / nm).exists():
            for cand in (SRC / nm, WORK / nm, ROOT / nm, ROOT / "data" / nm):
                if cand.exists(): shutil.copy(cand, DW / nm); break
    print(f"copied data -> {DW}")
    subprocess.run([sys.executable, "make_dev_split.py", str(DW)], cwd=str(ROOT), check=True)
    subprocess.run([sys.executable, "features/enrich.py", str(DW)], cwd=str(ROOT), check=True)
    DATA = DW; print(f"data prepared at {DW}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--warm_start", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--pool_size", type=int, default=12)
    ap.add_argument("--per_combo", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--periodic_sigma", type=float, default=0.3)
    ap.add_argument("--out_subdir", type=str, default="inc17_mlc")
    a=ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng=np.random.RandomState(a.seed)
    prepare_data()                                           # Kaggle: copy input -> writable, dev-split, enrich
    outdir=ROOT/"results"/f"{a.out_subdir}_seed{a.seed}"; outdir.mkdir(parents=True, exist_ok=True)

    tok=L("tokens8_train").astype(np.float32); mk=L("mask_train").astype(bool); y=L("y_train_set").astype(np.float32)
    N=len(tok)
    # combo bitmask index for fast episode sampling
    combo_of=np.zeros(N, dtype=np.int64); combos=defaultdict(list)
    for i in range(N):
        d=np.where(y[i]>0.5)[0]; m=0
        for x in d: m|=(1<<int(x))
        combo_of[i]=m; combos[m].append(i)
    uniq=np.array(list(combos.keys()), dtype=np.int64)
    print(f"train {N} mixtures, {len(uniq)} unique combos")

    base=dict(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
              dropout=0.1,cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",
              periodic_sigma=a.periodic_sigma,aux_heads=True,sparse_attn=True)
    net=SetTransformerMixture(**base).to(DEV)
    if a.warm_start:
        sd=torch.load(ROOT/a.warm_start if not Path(a.warm_start).is_absolute() else a.warm_start,
                      weights_only=True, map_location=DEV)
        miss,unexp=net.load_state_dict(sd, strict=False); print(f"warm-start (missing {len(miss)})")
    proto_proj=nn.Linear(128,128).to(DEV); nn.init.zeros_(proto_proj.weight); nn.init.zeros_(proto_proj.bias)
    _num=tok[:,:,1:8][mk]; net.feat_mean.copy_(torch.tensor(_num.mean(0),device=DEV)); net.feat_std.copy_(torch.tensor(_num.std(0)+1e-6,device=DEV))
    opt=torch.optim.AdamW(list(net.parameters())+list(proto_proj.parameters()), lr=a.lr, weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.episodes, eta_min=a.lr*0.05)
    asl=AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)

    def enc_z(idx):
        x=torch.from_numpy(tok[idx]).to(DEV); m=torch.from_numpy(mk[idx]).to(DEV)
        _,H,pad=net._encode_set(x,m); z=net.pma(H,pad_mask=pad).squeeze(1); return z, x, m
    def build_proto(zs, ys):                                   # zs (S,d), ys (S,45) -> P (45,d)
        w=ys.t(); cnt=w.sum(1,keepdim=True).clamp(min=1.0); return (w @ zs)/cnt
    def query_logits(xq, mq, P):
        _,H,pad=net._encode_set(xq,mq); return net.cls_decoder_module(H, pad_mask=pad, geno_query=proto_proj(P))

    def sample_episode():
        for _ in range(200):
            pool=rng.choice(45, a.pool_size, replace=False); pm=0
            for d in pool: pm|=(1<<int(d))
            inside=uniq[(uniq & ~pm)==0]                       # combos fully inside the pool
            if len(inside)<6: continue
            rng.shuffle(inside); h=len(inside)//2
            sup_c=inside[:h]; sd=0
            for c in sup_c: sd|=int(c)
            qry_c=[int(c) for c in inside[h:] if (int(c) & ~sd)==0]   # query donors covered by support
            if not qry_c: continue
            sidx=[rng.choice(combos[int(c)]) for c in sup_c for _ in range(a.per_combo)]
            qidx=[rng.choice(combos[int(c)]) for c in qry_c for _ in range(a.per_combo)]
            return np.array(sidx), np.array(qidx)
        return None, None

    @torch.no_grad()
    def evaluate():
        net.eval()
        # prototype from REAL VAL (deployable)
        tv=L("tokens8_val").astype(np.float32); mkv=L("mask_val").astype(bool); yv=L("y_val_set").astype(np.float32)
        Z=[]; Y=[]
        for s in range(0,len(tv),256):
            x=torch.from_numpy(tv[s:s+256]).to(DEV); m=torch.from_numpy(mkv[s:s+256]).to(DEV)
            _,H,pad=net._encode_set(x,m); Z.append(net.pma(H,pad_mask=pad).squeeze(1)); Y.append(torch.from_numpy(yv[s:s+256]).to(DEV))
        P=build_proto(torch.cat(Z), torch.cat(Y))              # (45,d)
        tt=L("tokens8_test").astype(np.float32); mkt=L("mask_test").astype(bool); yt=L("y_test_set").astype(np.float32); noc=L("noc_test").astype(int)
        Pr=np.zeros((len(tt),45))
        for s in range(0,len(tt),256):
            x=torch.from_numpy(tt[s:s+256]).to(DEV); m=torch.from_numpy(mkt[s:s+256]).to(DEV)
            Pr[s:s+256]=torch.sigmoid(query_logits(x,m,P)).cpu().numpy()
        out={}
        for k in range(1,6):
            ii=np.where(noc==k)[0]; e=[]
            for i in ii:
                t=np.argsort(Pr[i])[::-1][:k]; pr=np.zeros(45,int); pr[t]=1; e.append((pr==yt[i]).all())
            out[k]=round(float(np.mean(e)),3)
        net.train(); return out

    print(f"baseline-ish eval (proto_proj=0 -> base):", evaluate(), flush=True)
    t0=time.time(); run=0.0
    for ep in range(a.episodes):
        sidx,qidx=sample_episode()
        if sidx is None: continue
        zs,_,_=enc_z(sidx); ys=torch.from_numpy(y[sidx]).to(DEV)
        P=build_proto(zs, ys)
        xq=torch.from_numpy(tok[qidx]).to(DEV); mq=torch.from_numpy(mk[qidx]).to(DEV); yq=torch.from_numpy(y[qidx]).to(DEV)
        logits=query_logits(xq, mq, P)
        loss=asl(logits, yq)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(proto_proj.parameters()),5.0); opt.step(); sched.step()
        run=0.98*run+0.02*loss.item()
        if (ep+1)%2000==0:
            r=evaluate(); print(f"ep{ep+1}/{a.episodes} loss~{run:.3f} test N5={r[5]} (N1-4 {r[1]}/{r[2]}/{r[3]}/{r[4]}) [{time.time()-t0:.0f}s]", flush=True)
            torch.save(net.state_dict(), outdir/"best_model.pt")
            json.dump({"episode":ep+1,"test_oracle":r,"config":{**base,"seed":a.seed,"mlc":True}}, open(outdir/"metrics.json","w"), indent=2)
    print("done", evaluate(), flush=True)

if __name__=="__main__":
    main()

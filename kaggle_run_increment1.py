"""
kaggle_run_increment1.py — Increment 1 (enriched relational token) head-to-head on Kaggle GPU.

Trains, on the SAME combo-diverse in-silico data (data_insilico_w), the contributor-ID/NOC model with
the BASELINE 3-field token vs the ENRICHED 8-field token (Hb/SR/rank/n/glob_rel), for hybrid and pure-ST.
Each run self-evaluates and saves results/<subdir>/metrics.json (per-NOC oracle + decoded EM, reject AUROC).

SETUP on Kaggle (GPU notebook):
  0. LOCAL prep (needs data_raw/ CSVs; produces the upload dataset). Run, in order:
        python extract_phi_condition.py      # real phi/condition/template/Q
        python synth/extract_genotypes.py    # consensus donor genotypes
        python build_real_attr.py            # real allele->donor labels (attr_{val,test})
        python extract_size.py               # real per-peak size(bp)
        STR_DATA_DIR=./data STR_OUT_DIR=./data_insilico_w \
          python make_insilico.py --build 50000 --noc_weights 1,1.5,2.5,2 --seed 42
     -> data_insilico_w now has tokens/mask/Xflat/y/noc + attr_train/phi_train/size_train
        + phi_/condition_/attr_/size_ for val/test (BYTE-IDENTICAL train to the old set; only
        extra label files added). For Inc1-only you can reuse the OLD data_insilico_w.
  1. Upload `data_insilico_w/` (all .npy + .json) as a Kaggle Dataset, e.g. "noc-insilico-w".
  2. Upload the code bundle (kaggle_bundle: *.py, models/, features/, configs/) as Dataset "noc-code".
     Then in a GPU (T4) notebook:
        !cp -r /kaggle/input/noc-code/* /kaggle/working/
        !INSILICO_W=/kaggle/input/noc-insilico-w RUNS=inc2_2b,inc2_2c,inc2_2b_pe05,inc2_2b_pe1,inc2_2b_pe3,inc2_2b_pe9,inc2_2b_pe11 \
            python /kaggle/working/kaggle_run_increment1.py
     (inc2_2b = base for the comparison; inc2_2c = +Rank-N-Contrast (§4b); inc2_2b_pe* = P4
      per-feature periodic embedding sigma-sweep + tok9/tok11 behind embedding.)
ENV:
  INSILICO_W : path to the data_insilico_w dataset (default /kaggle/input/noc-insilico-w)
  RUNS       : comma list of jobs (default = periodic-embedding: st_pd_pp8_pe,st_pd8_pe,hybrid8_pp_pe).
               INCREMENT 2 (2x2 + size, needs the regenerated dataset above):
                 inc2_base    (tok8 RAW isab++ = settled Inc1 baseline st_pd_pp8 .957, re-run)
                 inc2_2a      (tok9  = + forward-stutter)
                 inc2_2b      (tok8  + --aux_heads privileged supervision)
                 inc2_2ab     (tok9  + aux)
                 inc2_2a_full (tok11 = + size + degradation)   inc2_2ab_full (tok11 + aux)
               Inc1 jobs: hybrid3,hybrid8,st_pd3,st_pd8,st_pd_pp8,st_pd_pp3,st_pool8,hybrid8_pp.
  EPOCHS     : override epochs (default from config).
The runner: copies data -> make_dev_split -> features/enrich (tokens8/9/11) -> trains each RUN ->
  eval_phi_condition.py (A-D on real) for inc2 arms -> prints summary.
DOWNLOAD afterwards: /kaggle/working/results/  (metrics.json + best_model.pt + phi/attr preds per run).
"""
import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path(os.environ.get("WORK_DIR", "/kaggle/working"))
SRC = Path(os.environ.get("INSILICO_W", "/kaggle/input/noc-insilico-w"))
RUNS = os.environ.get("RUNS", "st_pd_pp8_pe,st_pd8_pe,hybrid8_pp_pe").split(",")
# SEEDS: comma list of training seeds per arm (default "42"). Multiple seeds → each arm runs
# once per seed into <subdir>_s<seed>, so aggregate_seeds.py can report per-NOC mean±CI and
# separate a real per-arm effect from run-to-run noise (NOC strata swing ~±10pp at n=44-48).
SEEDS = [s.strip() for s in os.environ.get("SEEDS", "42").split(",") if s.strip()]
EPOCHS = os.environ.get("EPOCHS")
PY = sys.executable
DATA_W = WORK / "data_w"

assert (SRC / "meta_set.json").exists(), f"data_insilico_w not found at {SRC} — set INSILICO_W"
subprocess.run([PY, "-c", "import torch;print('CUDA:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"])

# 1) Copy data to a WRITABLE dir (Kaggle input is read-only; enrich writes tokens8_* there)
DATA_W.mkdir(parents=True, exist_ok=True)
for f in SRC.glob("*.npy"):
    if not (DATA_W / f.name).exists():
        shutil.copy(f, DATA_W / f.name)
for f in SRC.glob("*.json"):
    shutil.copy(f, DATA_W / f.name)
# Inc3 A (--geno_query): reference-genotype query tokens are precomputed (build_donor_geno.py needs the
# raw xlsx, absent on Kaggle) and shipped in the code bundle; copy into the writable data dir.
for nm in ("donor_geno.npy", "donor_geno_mask.npy"):
    if not (DATA_W / nm).exists():
        for cand in (SRC / nm, WORK / nm):
            if cand.exists():
                shutil.copy(cand, DATA_W / nm); break
print(f"copied data -> {DATA_W}")

# 2-LUPI) If a LUPI arm is requested and the physical labels are absent, GENERATE the LUPI dataset
#   IN-PLACE (peak model + drop-in + privileged β/mu/var/dropin) from the real source already in DATA_W
#   (real single-source N1 + donor_geno/size + real val/test). This means you DON'T upload data_lupi —
#   point INSILICO_W at the existing in-silico dataset and the runner builds the LUPI labels here.
#   ⚠ It OVERWRITES *_train (LUPI train = real N1 + fresh synth); run LUPI arms in a SEPARATE invocation.
_LUPI_ARMS = {"lupi_v1_data", "lupi_v2_degr", "lupi_v3_mu", "lupi_v4_var", "lupi_v5_dropin"}
if any(r.strip() in _LUPI_ARMS for r in RUNS) and not (DATA_W / "mu_train.npy").exists():
    _genv = os.environ.copy(); _genv["STR_DATA_DIR"] = str(DATA_W); _genv["STR_OUT_DIR"] = str(DATA_W)
    print(">>> LUPI: generating physical-label dataset (drop-in + β/mu/var/dropin) in-place from source")
    subprocess.run([PY, "gen_lupi.py"], cwd=str(WORK), env=_genv, check=True)

# 2a) Carve a combo-disjoint balanced DEV split out of the in-silico train (for checkpoint
#     selection); shrinks *_train, writes *_dev. Real val kept for the decoder stage1 prior.
subprocess.run([PY, "make_dev_split.py", str(DATA_W)], cwd=str(WORK), check=True)

# 2b) Build enriched 8-field tokens (tokens8_{train,val,test,open,dev}.npy) in DATA_W
subprocess.run([PY, "features/enrich.py", str(DATA_W)], cwd=str(WORK), check=True)

# 2c) RICHER DATA: build R-replicate pooled peak arrays (_rep{R}) into DATA_W — ONLY when a replicate
#     arm is requested (the 50k×R generation is wasted otherwise). Runs after dev+enrich so y/noc/phi +
#     the 'dev' selection split exist; make_replicates reads/writes DATA_W and enriches its own tokens.
_REP_ARMS = {"inc25_replicates_aslot", "inc26_hybrid_aslot"}
if any(r.strip() in _REP_ARMS for r in RUNS):
    _renv = os.environ.copy(); _renv["STR_REPLICATES"] = os.environ.get("STR_REPLICATES", "3")
    _renv["STR_DATA_DIR"] = str(DATA_W)   # make_insilico (imported by make_replicates) reads size_train/meta/geno from HERE
    subprocess.run([PY, "make_replicates.py", str(DATA_W)], cwd=str(WORK), env=_renv, check=True)

# 3) Head-to-head trainings (each writes results/<subdir>/metrics.json)
env = os.environ.copy(); env["STR_DATA_DIR"] = str(DATA_W)


def run(cmd, env_over=None):
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=str(WORK), env=(env_over or env), check=True)


ep = (["--epochs", EPOCHS] if EPOCHS else [])
H = [PY, "train_hybrid.py", "--loss", "asl", "--decouple_reject"]
S = [PY, "train_set_transformer.py", "--loss", "asl"]
# Increment-9 shared base = pe_s3 + sparse (= inc2_2d_sparse config; DEV N5 oracle ~0.59).
BASE9 = ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads",
         "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn"]
# Increment-13 base = the inc6_maskp config (pe_s3 + sparse + mask_peaks 0.15) — the best checkpoint;
# the 3 generator-proportion arms train THIS exact config on differently-generated data.
GENMASKP = ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads",
            "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--mask_peaks", "0.15"]
# Inc-LUPI base = the canonical inc22_fixed_aslot config (trained on data_lupi via INSILICO_W=data_lupi).
LUPIB = ["--noc_head_v2", "--nc_attn", "mab0", "--feas_filter", "--soft_attr_label", "--n_token_feats", "8",
         "--cls_decoder", "aslot", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic",
         "--periodic_sigma", "0.3", "--mask_peaks", "0.15", "--set_of_set",
         "--n_slot_iters", "3", "--ot_eps", "0.05", "--ot_iters", "5", "--epochs", "150"]
# Head-to-head (token8 = enriched). ST tested with the FAIR readout (per_donor = Query2Label L=2,
# no query self-attn) under BOTH encoders (isab=LayerNorm vs isab++=SetNorm+clean-path), plus the
# pooled readout as the weak baseline (shows readout matters). hybrid = deployable reference.
jobs = {
    # enrichment effect (token3 vs token8), controlled, same dev-selection/readout/encoder:
    "hybrid3":    H + ["--n_token_feats", "3", "--out_subdir", "inc1_hybrid_tok3", *ep],
    "hybrid8":    H + ["--n_token_feats", "8", "--out_subdir", "inc1_hybrid_tok8", *ep],
    "st_pd3":     S + ["--n_token_feats", "3", "--cls_decoder", "per_donor", "--out_subdir", "inc1_st_perdonor_tok3", *ep],
    "st_pd8":     S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--out_subdir", "inc1_st_perdonor_tok8", *ep],
    # encoder effect (LayerNorm vs SetNorm) on fair-ST:
    "st_pd_pp8":  S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--out_subdir", "inc1_st_perdonor_tok8_pp", *ep],
    # readout baseline (pooled = the weak readout) + optional extras:
    "st_pool8":   S + ["--n_token_feats", "8", "--cls_decoder", "pooled",    "--out_subdir", "inc1_st_pooled_tok8", *ep],
    "hybrid8_pp": H + ["--n_token_feats", "8", "--encoder", "isab++", "--out_subdir", "inc1_hybrid_tok8_pp", *ep],
    "st_pd_pp3":  S + ["--n_token_feats", "3", "--cls_decoder", "per_donor", "--encoder", "isab++", "--out_subdir", "inc1_st_perdonor_tok3_pp", *ep],
    # PER-FEATURE EMBEDDING effect (§3): identical to st_pd_pp8 / hybrid8_pp but periodic PLR
    # numerical embedding instead of raw scalar→shared-Linear. Tests whether the rotation-invariance
    # bottleneck (Grinsztajn 2022) is what crippled the enriched tok8 token.
    "st_pd_pp8_pe": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--num_embed", "periodic", "--out_subdir", "inc1_st_perdonor_tok8_pp_pe", *ep],
    "st_pd8_pe":    S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--num_embed", "periodic", "--out_subdir", "inc1_st_perdonor_tok8_pe", *ep],
    "hybrid8_pp_pe": H + ["--n_token_feats", "8", "--encoder", "isab++", "--num_embed", "periodic", "--out_subdir", "inc1_hybrid_tok8_pp_pe", *ep],
    # ── INCREMENT 2 — 2x2 ablation (forward-stutter feature × privileged supervision), on the
    # settled Inc1 base (per_donor + isab++ + RAW num_embed = st_pd_pp8, oracle .957). RAW not
    # periodic: F8/P4 decided periodic σ=1 is untuned/unstable (a periodic inc2_base COLLAPSED to
    # oracle .837 N3/N4/N5≈0 vs raw .957) -> carry the stable RAW base; PE σ-sweep is deferred (P4).
    # REQUIRES data regenerated with the new
    # make_insilico.py (attr_train/phi_train present); the runner's dev_split + enrich carry them.
    # 2a = tok9 (adds forward-stutter); 2b = --aux_heads (allele→donor attr + phi, Kendall-weighted).
    #
    # REPRODUCIBILITY: regenerating with the EXACT original command is BYTE-IDENTICAL to the old
    # data_insilico_w (verified) — the new generator adds 0 RNG calls, only extra label files. So the
    # Inc1 baseline stays valid and inc2_base re-runs it exactly. Use:
    #   python extract_phi_condition.py            # real phi/condition/template/Q (run first)
    #   python synth/extract_genotypes.py          # consensus donor genotypes (for real attribution)
    #   python build_real_attr.py                  # real allele->donor labels (attr_{val,test})
    #   python extract_size.py                     # real per-peak size(bp) (needed for tok11 + size table)
    #   STR_DATA_DIR=.../data STR_OUT_DIR=.../data_insilico_w \
    #     python make_insilico.py --build 50000 --noc_weights 1,1.5,2.5,2 --seed 42
    # (changing build/noc_weights/seed WOULD change the train data -> not comparable to old Inc1.)
    # make_insilico copies phi_/condition_/attr_/size_ for val/test -> eval_phi_condition.py reports A-D on real.
    "inc2_base": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--out_subdir", "inc2_base", *ep],
    "inc2_2a":   S + ["--n_token_feats", "9", "--cls_decoder", "per_donor", "--encoder", "isab++", "--out_subdir", "inc2_2a_fwdstutter", *ep],
    "inc2_2b":   S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--out_subdir", "inc2_2b_privsup", *ep],
    "inc2_2ab":  S + ["--n_token_feats", "9", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--out_subdir", "inc2_2ab", *ep],
    # full 2a token = forward-stutter + SIZE + degradation residual (tok11; needs extract_size.py +
    # size table in make_insilico). inc2_2a vs inc2_2a_full isolates size's contribution.
    "inc2_2a_full":  S + ["--n_token_feats", "11", "--cls_decoder", "per_donor", "--encoder", "isab++", "--out_subdir", "inc2_2a_full", *ep],
    "inc2_2ab_full": S + ["--n_token_feats", "11", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--out_subdir", "inc2_2ab_full", *ep],
    # ── INCREMENT 2c — DECOUPLED ordinal NOC contrast (Rank-N-Contrast, Zha 2023) on the 2b base.
    # Adds a separate pool (pma_noc) + projection head (proj_noc, discarded at inference) carrying a
    # |NOCi-NOCj|-weighted ordinal contrast (§4b-A) so the representation is ordered along the 1->5
    # axis WITHOUT touching the ID readout (§4b-C valves 2/3). Kendall-weighted. Ablation = inc2_2b
    # (same flags) vs inc2_2c (+--noc_contrast). NO-REGRESSION GUARD: per-NOC ID EM on NOC1/2/3 must
    # not drop >1pp vs 2b (compare metrics.json per_noc) even if NOC4/5 improves (§4b-C valve 5).
    "inc2_2c":     S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--noc_contrast", "--out_subdir", "inc2_2c_contrast", *ep],
    # ── P4 — per-feature PERIODIC embedding sigma-sweep on the 2b base (resolves the F11/F8 OPEN
    # verdict: was raw-scalar delivery the bottleneck?). sigma in {0.05,0.1,0.3} (F8: sigma=1 unstable).
    "inc2_2b_pe05": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.05", "--out_subdir", "inc2_2b_pe_s05", *ep],
    "inc2_2b_pe1":  S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.1",  "--out_subdir", "inc2_2b_pe_s1",  *ep],
    "inc2_2b_pe3":  S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",  "--out_subdir", "inc2_2b_pe_s3",  *ep],
    # ── P4-features — do the DEFERRED 2a features (fwd-stutter tok9, size/degr tok11) help once
    # delivered behind per-feature embedding (F11 OPEN)? Same as 2b but periodic (sigma=0.1; retune to
    # the sweep winner). tok9 = +forward-stutter, tok11 = +size+degradation.
    "inc2_2b_pe9":  S + ["--n_token_feats", "9",  "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.1", "--out_subdir", "inc2_2b_pe_tok9",  *ep],
    "inc2_2b_pe11": S + ["--n_token_feats", "11", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.1", "--out_subdir", "inc2_2b_pe_tok11", *ep],
    # ── batch-3 CLEAN FEATURE ISOLATION (F14): tok8 vs tok9 vs tok11 at the SAME sigma=0.3 so the
    # FS / size effect is no longer confounded with sigma (pe_s3=tok8 is the matched baseline).
    "inc2_pe3_tok9":  S + ["--n_token_feats", "9",  "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--out_subdir", "inc2_2b_pe3_tok9",  *ep],
    "inc2_pe3_tok11": S + ["--n_token_feats", "11", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--out_subdir", "inc2_2b_pe3_tok11", *ep],
    # ── batch-3 RNC VALVE VARIANTS (F15, §4b-C) on the periodic sigma=0.3 base (pe_s3 = the no-RNC
    # baseline). Three separate arms to compare how the contrast gradient meets the shared encoder:
    #   shared = original 2c (grad reaches encoder) · detach = pma_noc pools H.detach (pure
    #   decoupling control) · pcgrad = valve 4 (project contrast grad off the main-task grad).
    "inc2_2c_shared": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--noc_contrast", "--noc_contrast_mode", "shared", "--out_subdir", "inc2_2c_pe3_shared", *ep],
    "inc2_2c_detach": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--noc_contrast", "--noc_contrast_mode", "detach", "--out_subdir", "inc2_2c_pe3_detach", *ep],
    "inc2_2c_pcgrad": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--noc_contrast", "--noc_contrast_mode", "pcgrad", "--out_subdir", "inc2_2c_pe3_pcgrad", *ep],
    # ── RNC FIX (F19): small tau (0.3) + fixed weight (bypass Kendall) so the contrast ACTUALLY
    # descends. _sh = shared (does it descend at all? → probe_noc_structure must pass first);
    # _pg = pcgrad (the ID-protected deployable version). Re-test ONLY after the probe confirms
    # z_noc_proj is ordinally structured (else the manipulation still failed).
    "inc2_2c_fix_sh": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--noc_contrast", "--noc_contrast_mode", "shared", "--rnc_tau", "0.3", "--rnc_fixed_weight", "1.0", "--out_subdir", "inc2_2c_fix_sh", *ep],
    "inc2_2c_fix_pg": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--noc_contrast", "--noc_contrast_mode", "pcgrad", "--rnc_tau", "0.3", "--rnc_fixed_weight", "1.0", "--out_subdir", "inc2_2c_fix_pg", *ep],
    # ── INCREMENT 2d (§7) architecture levers on the pe_s3 base — each ISOLATED for comparison.
    # sparse = sparsemax per-donor decoder (zero attention to spurious peaks);
    # irm = NOC-environment invariance penalty (suppress combo shortcut). Eval on REAL test + seeds.
    "inc2_2d_sparse": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--out_subdir", "inc2_2d_sparse", *ep],
    "inc2_2d_irm":    S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--irm", "--irm_lambda", "1.0", "--out_subdir", "inc2_2d_irm", *ep],
    "inc2_2d_irm10":  S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--irm", "--irm_lambda", "10.0", "--out_subdir", "inc2_2d_irm10", *ep],

    # ── Increment 3 (reports/design_increment3_levers.md): each arm = pe_s3 base + ONE published method.
    #    base = --n_token_feats 8 --cls_decoder per_donor --encoder isab++ --aux_heads --num_embed periodic --periodic_sigma 0.3
    # Group 1 — representation levers (lift faint-minor ID; info present per F6/measure_id_ambiguity.py):
    "inc3_repA_genoq":   S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--geno_query", "--out_subdir", "inc3_repA_genoq", *ep],
    "inc3_repB_donorcon":S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--donor_contrast", "--out_subdir", "inc3_repB_donorcon", *ep],
    "inc3_repC_additive":S + ["--n_token_feats", "8", "--cls_decoder", "additive",  "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--out_subdir", "inc3_repC_additive", *ep],
    # Group 2 — wire RNC INTO the count rep via a CORN ordinal head on its own encoder pool (fixes F21):
    "inc3_nocV1_ordrnc": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--noc_ord_head", "--noc_ord_detach", "--noc_contrast", "--rnc_tau", "0.3", "--rnc_fixed_weight", "1.0", "--out_subdir", "inc3_nocV1_ordrnc", *ep],
    "inc3_nocV2_ordrnc_pcgrad": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--noc_ord_head", "--noc_contrast", "--noc_contrast_mode", "pcgrad", "--rnc_tau", "0.3", "--rnc_fixed_weight", "1.0", "--out_subdir", "inc3_nocV2_ordrnc_pcgrad", *ep],
    "inc3_nocV3_ordreplace": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--noc_ord_head", "--noc_ord_replace", "--out_subdir", "inc3_nocV3_ordreplace", *ep],

    # ── INCREMENT 4 (combo-generalization, F29): the binding wall is ID-oracle combo-overfit
    #    (train N5 oracle ~1.0 -> held-out-combo dev ~.57; real≈dev). Judge EVERY arm on the
    #    DEV per-NOC oracle (measure_insilico_oracle.py), NOT aggregate EM (F23 trap). Base for
    #    P1-P4 = pe_s3 (tok8 + periodic σ0.3 + aux + isab++). P5/P6 = SEPARATE scripts (own code).
    # P1 — stack the two OOD-proven per-donor readout regularizers (F22 sparse + F27 geno_query).
    "inc4_p1_stack": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--geno_query", "--out_subdir", "inc4_p1_stack", *ep],
    # P2 — P1 + STRONG per-donor readout: ID reads LOCAL (non-global-mixed) features (decoder_source=local),
    #      cutting the global-mixing combo-leak that feeds the ID decision (F29). ISAB still feeds aux/reject/NOC.
    "inc4_p2_local": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--geno_query", "--decoder_source", "local", "--out_subdir", "inc4_p2_local", *ep],
    # P3 — IRM re-judged AT N5 (F23 dropped it on aggregate EM which masks the combo gap). On adopted base (+sparse).
    "inc4_p3_irm": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--irm", "--irm_lambda", "1.0", "--out_subdir", "inc4_p3_irm", *ep],
    # P4 — DECORRELATED data (balance donor co-occurrence; kill the bind-combo shortcut). Same train cmd as the
    #      pe_s3+sparse base, but STR_DATA_DIR is swapped to the decorrelated dir (special-cased in the loop).
    "inc4_p4_decorr": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--out_subdir", "inc4_p4_decorr", *ep],
    # P5 — NOC intrinsic, decoupled from the ID-profile bottleneck (F30: count combo-overfits via that coupling).
    #      SEPARATE CODE (train_p5_noc_intrinsic.py) so existing code is untouched.
    "inc4_p5_noc_intrinsic": [PY, "train_p5_noc_intrinsic.py", *ep],
    # P6 — per-CONTRIBUTOR slot set-prediction unifying ID+NOC (DETR/DeepSetNet; count = #non-∅ slots).
    #      SEPARATE CODE (train_p6_slot.py).
    "inc4_p6_slot": [PY, "train_p6_slot.py", *ep],

    # ── INCREMENT 5 (reports/design_increment5_peeling.md): RESIDUAL-AWARE (peeling) decoder for the
    #    N5 oracle wall. Base = pe_s3 + sparse (SAME flags as inc4_p4), but trained on RESIDUAL-AUGMENTED
    #    data (build_residual_aug.py → data_res_<mode>; STR_DATA_DIR swapped, special-cased in the loop)
    #    and decoded with GATED RECURSIVE PEEL (eval_peel_decode.py). Fixes the frozen-model OOD-on-
    #    residual failure (neural_peel.py) so the trained model can score residuals at decode.
    #  P1 rand1 = residual aug peels 1 random donor/sample (gentle); P2 randk = peels random 1..K-1.
    "inc5_res_rand1": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--out_subdir", "inc5_res_rand1", *ep],
    "inc5_res_randk": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--out_subdir", "inc5_res_randk", *ep],

    # ── INCREMENT 6 (reports/design_increment6_root_levers.md): ROOT anti-memorization levers for the
    #    combinatorial-generalization wall (F29/F30). Peeling/residual-aug only chip the symptom; these
    #    attack the mechanism (decoder MEMORIZES combos). Base = pe_s3 + sparse (SAME flags as inc5).
    #    Judge EVERY arm on DEV per-NOC oracle (measure_insilico_oracle.py), NOT test EM after peel.
    #    SCREEN = 1 seed, direction-finder (N5 var ±.18 → confirm with 3-seed CIs before any conclusion).
    # D-mask  — input-peak dropout: decoder can't template-match a memorized combo (anti-memorization).
    "inc6_maskp":   S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--mask_peaks", "0.15", "--out_subdir", "inc6_maskp", *ep],
    # D-andmask — ILC AND-mask (Parascandolo 2020) on cls grad across low/high-NOC envs; invariance
    #             WITHOUT a penalty term (so aux/φ heads stay intact, unlike IRM F23).
    "inc6_andmask": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--and_mask", "--out_subdir", "inc6_andmask", *ep],
    # C slot-fix — DETR eos_coef + slot-diversity to break the ∅-collapse of the P6 slot binder
    #              (SEPARATE CODE train_p6_slot.py --slot_fix; self-measures setEM/NOC per NOC).
    "inc6_slotfix": [PY, "train_p6_slot.py", "--slot_fix", *ep],
    # B meta — Reptile (Nichol 2018) donor-context episodic finetune of a trained base (Lake 2019 proxy);
    #          warm-starts from inc5_res_rand1_seed42 (seed-matched; must exist). SEPARATE CODE.
    "inc6_meta":    [PY, "train_inc6_meta.py", "--init_from", "inc5_res_rand1_seed42", *ep],
    # ── INCREMENT 6 batch-2: the remaining blind-spot GO/untested levers (feasibility_inc6{,b}.py).
    # SAM (4a) — Sharpness-Aware Minimization (Foret 2021), flat minima; no-train probe found novel-N5
    #            2.1x sharper than seen. SEPARATE CODE (two forward-backward/step).
    "inc6_sam":     [PY, "train_inc6_sam.py", *ep],
    # masked-pretrain (3e) — self-supervised masked-feature reconstruction pre-phase, then normal supervised.
    "inc6_maskpre": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--masked_pretrain_epochs", "20", "--out_subdir", "inc6_maskpre", *ep],
    # VIB (2f) — variational information bottleneck on per-donor latents (disentangle, drop combo nuisance).
    "inc6_vib":     S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--vib", "--out_subdir", "inc6_vib", *ep],
    # FOMAML (4c+) — first-order MAML (query objective), stronger than Reptile; warm-starts inc5 base.
    "inc6_fomaml":  [PY, "train_inc6_meta.py", "--fomaml", "--init_from", "inc5_res_rand1_seed42", *ep],
    # ── INCREMENT 6 batch-3: VERIFIED root cause = minor-contributor sensitivity (missed dev-N5 donors
    # have mean mixture-frac 0.059 vs 0.21 for hits; 97% of misses <0.15; 62% sit at rank 6-10 = recoverable).
    # Lever = cost-weight low-phi POSITIVES, capped (NOT blind gamma_pos which chases the phi<0.05 noise floor).
    # Base = the best arm (maskp = pe_s3+sparse+mask_peaks). Run 3 seeds (1/machine) for a CI vs maskp(.68).
    "inc6_minorw":  S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--mask_peaks", "0.15", "--minor_weight", "--minor_phi_ref", "0.2", "--minor_cap", "4.0", "--out_subdir", "inc6_minorw", *ep],

    # ── INCREMENT 7 (design_increment7_masspool.md): F31 ROOT lever = MAJOR-PEAK MASS DOMINANCE.
    #    3-seed proof (probe_context/probe_root_levers/probe_height_decouple): the ISAB inducing-point
    #    compression pools peaks by a softmax WEIGHTED-MEAN → many tall MAJOR peaks wash the few faint
    #    MINOR peaks in H (attr .07; recovers to .81 only when majors are removed). Height-feature decouple
    #    is INSUFFICIENT (.07→.16) — it's the mass of the major MAJORITY, not the feature. Lever = mass-
    #    preserving inducing compression (scaled weighted-SUM, Fischer&Gartner 2024 arXiv 2407.04170).
    #    Base = pe_s3 + sparse (= the working base). Judge on DEV N5 oracle vs base ~.65 (3-seed CI).
    "inc7_masspool": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--mass_pool", "--out_subdir", "inc7_masspool", *ep],
    # ── INCREMENT 8 (F32): VICReg (arXiv 2105.04906) VERBATIM on per-donor reps. Base = inc2_2d_sparse
    #    (pe_s3+sparse; DEV N5 oracle 0.609 3-seed). N1-SAFE by the variance term (hard centering BROKE N1:
    #    1.0→0.02). V1 = VICReg variance+covariance (anti-collapse decorr = de-smooth carrier). V2 = + the
    #    invariance term (same donor across combos = combo-invariance). Judge DEV N5 oracle vs 0.609 + guard
    #    N1/2/3; 1-seed promise check then 3-seed. Probe ckpt: probe_encoder_entangle/seen_vs_novel/decompose_wall.
    "inc8_v1_vicreg":     S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--vicreg", "--out_subdir", "inc8_v1_vicreg", *ep],
    "inc8_v2_vicreg_inv": S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++", "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3", "--sparse_attn", "--vicreg", "--vicreg_inv", "--out_subdir", "inc8_v2_vicreg_inv", *ep],
    # ── INCREMENT 9 (F33): post-inc8 mechanism levers, each = the established base (pe_s3 + sparse =
    #    inc2_2d_sparse, DEV N5 oracle ~0.59) + ONE published method (clean isolation, project methodology).
    #    Run with EARLY_ABORT=1 so a clearly-failing arm self-cancels (guard/diverge/N5-below-base 2 checks).
    #    Judge DEV N5 oracle vs 0.59 + guard N1/2/3 (measure_insilico_oracle, auto); then probe ckpts.
    #    DECODER under-read family (A1/A2/A3 — ASL gamma_neg=4 over-suppresses faint positives, probe .84->.14):
    "inc9_a1_ldam":   S + BASE9 + ["--cls_loss", "ldam",   "--out_subdir", "inc9_a1_ldam", *ep],
    "inc9_a2_dbloss": S + BASE9 + ["--cls_loss", "dbloss", "--out_subdir", "inc9_a2_dbloss", *ep],
    "inc9_a3_bal":    S + BASE9 + ["--cls_loss", "bal",    "--out_subdir", "inc9_a3_bal", *ep],
    #    ENCODER anti-absorption (B1 source-recon, B4 softmax-1/attn-sink) + INTER-LOCUS (geno_query):
    "inc9_b1_recon":  S + BASE9 + ["--donor_recon",        "--out_subdir", "inc9_b1_recon", *ep],
    "inc9_b4_sink":   S + BASE9 + ["--attn_sink", "1",     "--out_subdir", "inc9_b4_sink", *ep],
    "inc9_inter_genoq": S + BASE9 + ["--geno_query",       "--out_subdir", "inc9_inter_genoq", *ep],
    #    A4 = DN-DETR query denoising (needs --geno_query for the noised anchor); B2 = IFM (adversarial
    #    per-donor-rep perturbation). Self-contained 2nd-pass/aux losses, smoked (eval-inert, finite grad).
    "inc9_a4_qdenoise": S + BASE9 + ["--geno_query", "--query_denoise", "--out_subdir", "inc9_a4_qdenoise", *ep],
    "inc9_b2_ifm":      S + BASE9 + ["--ifm",         "--out_subdir", "inc9_b2_ifm", *ep],
    # ── INCREMENT 10 (F34): the VERIFIED mechanism lever. The N5 miss = the model under-credits DECISIVE
    #    (panel-rare) reference-matching alleles (probe4-6, causal). probe7/8 (no-train ensemble): adding a
    #    rarity-weighted reference-allele match to the per-donor logit lifts N5 set-EM .72->.83 (the
    #    recoverable gap; floor ~.84). Base = inc2_2d_sparse (= BASE9). Needs donor_geno.npy in the data dir.
    #    V1 = fixed 1/panel-rarity weight (= the ensemble internalised, learnable scale only) — GUARANTEED to
    #    expose the +.11 signal. V2 = LEARN the per-allele discriminability weight (AFIA) — can exceed V1.
    #    JUDGE: DEV N5 oracle vs base 0.59 AND test set-EM vs the .83 ensemble baseline (does training keep it?);
    #    guard N1-N4 not dropping. 3-seed before any verdict (F14/F16). Probe: ensemble was set-EM .831.
    "inc10_v1_refmatch":       S + BASE9 + ["--ref_match", "--out_subdir", "inc10_v1_refmatch", *ep],
    "inc10_v2_refmatch_learn": S + BASE9 + ["--ref_match", "--ref_match_learn", "--out_subdir", "inc10_v2_refmatch_learn", *ep],

    # ── INCREMENT 11 (F35b): NON-COMPETITIVE SIGMOID ENCODER ATTENTION (Ramapuram et al. 2024, arXiv 2409.04431).
    #    The N5 encoder wall (~.20, 3-seed) = softmax COMPETITION in mab0 (inducing points softmax-attend over
    #    PEAKS) → tall MAJOR peaks monopolize each inducing summary → faint MINOR peaks washed/absorbed into
    #    majors (probe_encoder_isolate: missed-tail private peaks →major 72-88%). No-train ceiling probe
    #    (probe_noncompetitive_attn.py) LOCALIZED the lever to mab0 (relaxing it lifts missed-tail per-peak
    #    isolation .266→.46-.58; mab1 relaxation HURTS) but no-train can't deliver donor recall (frozen OOD)
    #    → TRAIN it. σ(QKᵀ/√d − log n)V, no row-norm = each peak an independent gate, no competition.
    #    Base = inc2_2d_sparse (= BASE9). M1 = mab0-only (hypothesis) · M2 = both (placement control: probe
    #    predicts ≤ mab0). JUDGE: DEV N5 oracle vs base ~.59 + guard N1/2/3; then re-run probe_encoder_isolate
    #    on the ckpt (does →major drop?). 3-seed (SEEDS=42,43,44) before any verdict. Readout 2nd wall (probe6)
    #    likely still needs a separate lever.
    "inc11_nc_mab0": S + BASE9 + ["--nc_attn", "mab0", "--out_subdir", "inc11_nc_mab0", *ep],
    "inc11_nc_both": S + BASE9 + ["--nc_attn", "both", "--out_subdir", "inc11_nc_both", *ep],

    # ── INCREMENT 12 — DECODER READOUT (close the +.10 N5 under-read), on the mab0 base ──────────────
    #   Probes (probe_inc11_decode / _identifiable / _additive_parts) proved: NO info floor (N5 97% identifiable);
    #   the mab0 encoder ceiling (soft-vote .849) >> the per_donor sparsemax decode (.747) = +.10 STRANDED. The
    #   sparsemax (hard) readout drops the faint donor's DIFFUSE evidence (the ~2.9 decisive alleles). additive
    #   was REJECTED (it collapses the encoder, ceiling N5 .005). Fix = SOFTER/adaptive per-donor aggregation
    #   (keeps per_donor → encoder stays good). Lit: entmax (Peters 2019) / ASEntmax learnable-temp (2508.17821)
    #   / DSMIL dual-stream (Li 2021). JUDGE: DEV N5 oracle vs mab0 (.639) + guard N1/2/3; 3-seed before verdict.
    "inc12_entmax15":    S + BASE9 + ["--nc_attn", "mab0", "--dec_aggr", "entmax15",
                                      "--out_subdir", "inc12_entmax15", *ep],
    "inc12_entmax_temp": S + BASE9 + ["--nc_attn", "mab0", "--dec_aggr", "entmax_temp",
                                      "--out_subdir", "inc12_entmax_temp", *ep],
    "inc12_dsmil":       S + ["--n_token_feats", "8", "--cls_decoder", "dsmil", "--encoder", "isab++",
                              "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                              "--nc_attn", "mab0", "--out_subdir", "inc12_dsmil", *ep],

    # ── INCREMENT 13 — GENERATOR-PROPORTION 3-way (root-cause fix for the synth->real N5 gap) ─────────
    #   Session 2026-06-13: the N5 wall is NOT encoder/decoder — a readout re-fit on REAL H reaches N5
    #   oracle 0.94 vs the synth-trained 0.79 (info IS in H). The gap = DOMAIN SHIFT in the mixture-
    #   proportion model: original r_max=6+5(k-2) gives N5 min-phi 0.054 / ratio 8.1 (minors ~2x too
    #   faint) vs real 0.125 / 4.0; real's balanced 40% (ratio<=2) is ~1% of synth (real min-phi = orig's
    #   97th pct). EACH ARM REGENERATES its own 50k train (same seed/noc_weights) with a different
    #   proportion model, then trains the SAME inc6_maskp config and evals on the SAME real test.
    #     gen_orig = original skew (baseline = reproduces inc6_maskp ~0.79)
    #     gen_real = realistic (matches real; expect best real-test N5 but NARROW = OOD risk)
    #     gen_wide = SUPERSET (skew U balanced; expect >= gen_real on real test AND best OOD = the
    #                "easy->hard beyond reality done right" — keeps the knowledge curriculum).
    #   JUDGE: real-test per_noc_oracle N5 (measure_insilico_oracle) vs 0.79; guard N1-N4 no regression.
    #   3-seed (SEEDS=42,43,44) before any verdict. The MASKP flags below MUST match inc6_maskp exactly.
    "gen_orig": S + GENMASKP + ["--out_subdir", "genprop_orig", *ep],
    "gen_real": S + GENMASKP + ["--out_subdir", "genprop_real", *ep],
    "gen_wide": S + GENMASKP + ["--out_subdir", "genprop_wide", *ep],

    # ── INCREMENT 13b — GENERALIZATION LEVERS (the 3 building blocks), on the inc6_maskp base + DATA_W
    #   (clean comparison to inc6_maskp 0.788). Each closes a different part of the combinatorial wall;
    #   all are train-time losses, zero/ramped-in so the model starts == maskp (no regression at init).
    #     B (distill)  = LUPI/generalized distillation: combo-invariant attr_head soft-vote -> decoder.
    #     A (cf-inv)   = counterfactual leave-one-donor-out consistency (combo-invariance, Veitch/NCM).
    #     C (ladder)   = difficulty-ladder, easy-teaches-hard faint-minor curriculum (FixMatch/FlexMatch).
    #   JUDGE: real-test per_noc_oracle N5 vs 0.788 + guard N1-N4; 3-seed before verdict. The combined
    #   ABC is runnable NOW; the MLC / soft-hybrid combos (matrix M1/M2 run-2, M3 run-2) come after these
    #   building blocks are validated alone (MLC/soft-hybrid not yet implemented).
    "inc13_B":   S + GENMASKP + ["--distill_attr",  "--out_subdir", "inc13_B_distill", *ep],
    "inc13_A":   S + GENMASKP + ["--cf_invariance", "--out_subdir", "inc13_A_cfinv", *ep],
    "inc13_C":   S + GENMASKP + ["--diff_ladder",   "--out_subdir", "inc13_C_ladder", *ep],
    "inc13_ABC": S + GENMASKP + ["--distill_attr", "--cf_invariance", "--diff_ladder",
                                 "--out_subdir", "inc13_ABC", *ep],
    # Increment 14 — root-cause levers (forensic-grounded; see reports/ + session notes):
    #   B-v2 = multi-label genotype CO-MEMBERSHIP head (CPI/inclusion: a peak owned by EVERY present donor
    #          carrying its allele — softmax attr_head can't, 75% shared) + MLD per-class binary KD distill
    #          (Multi-Label KD, Yang 2023). Teacher-ceiling probe: .82 (softmax) -> .96 (multi-label).
    #   A-v2 = additive-SUBTRACTION counterfactual (EuroForMix additive model: subtract a contributor's
    #          gamma deposit, shared peaks REMAIN) + conditional MMD on the identity sub-rep (CIP/HSCIC,
    #          NOT pointwise -> anti-collapse). Fixes A-v1's wrong invariance (deleted 41% shared evidence).
    #   JUDGE: real-test per_noc_oracle N5 vs base 0.788 + guard N1-N4; 3-seed before verdict.
    "inc14_Bv2": S + GENMASKP + ["--ml_attr", "--ml_distill", "--out_subdir", "inc14_Bv2", *ep],
    "inc14_Av2": S + GENMASKP + ["--add_invar", "--out_subdir", "inc14_Av2", *ep],
    # Increment 17 — NOISE-SPECTRUM: height-stratified multi-noise aug + cross-level consistency. The real faint
    #   band is ~91% ARTIFACT (measure_faint_peaks.py); true alleles are noise-ROBUST, decoy fodder noise-SENSITIVE
    #   -> learn invariance instead of hard-filtering (which dodges hard cases). A/B = height-aware noise-spectrum +
    #   consistency (inc17_ns) vs the current best uniform aug mask_peaks (inc17_base). Trains on DATA_W (runner preps).
    #   Built on the TWO best configs: inc6_maskp (=GENMASKP) and inc13_B_distill (=GENMASKP+distill_attr).
    #   noise_spectrum REPLACES mask_peaks (code elif) -> the A/B is height-aware-aug+consistency vs uniform dropout.
    #   Run (1 machine, sequential):  EPOCHS=40 RUNS=inc17_base,inc17_ns python kaggle_run_increment1.py
    #     (inc17_base re-trains the inc6_maskp config in-run for a same-conditions A/B; or compare to the existing
    #      inc6_maskp_seed42 directly. inc17_ns_distill = noise-spectrum on the OTHER best, inc13_B_distill.)
    #   JUDGE: DEV N5 oracle (generalization.dev_oracle["5"]) inc17_ns vs inc17_base/inc6_maskp + guard N1-N4. (add_recon
    #   Inc16 left for a separate full run: add --add_recon to any S-arm; decide after.)
    "inc17_base":       S + GENMASKP + ["--out_subdir", "inc17_base", *ep],
    "inc17_ns":         S + GENMASKP + ["--noise_spectrum", "--xnoise_consistency", "--out_subdir", "inc17_ns", *ep],
    "inc17_ns_distill": S + GENMASKP + ["--distill_attr", "--noise_spectrum", "--xnoise_consistency", "--out_subdir", "inc17_ns_distill", *ep],
    # --- Increment 16: additive-recon (Wiedemer compositional-gen) on the best config inc6_maskp (=GENMASKP) ---
    #   --add_recon: additive grid log-recon over present+absent bins (w=0.1); predicts 0 at damning-absent
    #   positions -> penalizes decoy inclusion. Orthogonal to mask_peaks (kept). A/B vs existing inc6_maskp_seed42.
    #   JUDGE: DEV N5 oracle (generalization.dev_oracle["5"]) inc16_addrecon vs inc6_maskp 0.6804 + guard N1-N4 + test N5.
    "inc16_addrecon":   S + GENMASKP + ["--add_recon", "--out_subdir", "inc16_addrecon", *ep],

    # ── INCREMENT 18: φ-injection (V1) + soft-genotype-attr (V2), base = inc6_maskp (GENMASKP) ──
    #   V1 phi_inject: inject GT phi (privileged) into per-donor decoder queries at train time;
    #          at inference, inject EM-uniform phi (NITER=20, same as probe_phi_rerank.py).
    #          Model learns FiLM-style "when phi_c is high, donor c is more likely" directly in cls logit.
    #          Key advantage over post-hoc rerank: phi signal inside logit → card-head also benefits.
    #   Inc18: soft_geno_attr + feas_filter (phi_inject dropped — EM-uniform beats neural phi at current scale).
    #   soft_geno_attr: private peaks hard-assign, shared soft-mask at INFERENCE only (unmasked CE at train).
    #   feas_filter: remove peaks with 0 carriers before encoder — both train and inference (no covariate shift).
    #   JUDGE: DEV N5 oracle vs maskp baseline + probe_phi_rerank.py after training; 3-seed before verdict.
    #   ab_soft_label_compare.py tests whether soft label (GT phi x Cn) improves attr head separately.
    #   M1 = inc18_soft_attr (soft_geno_attr + feas_filter)
    "inc18_soft_attr":  S + GENMASKP + ["--soft_geno_attr", "--feas_filter", "--soft_attr_label",
                                         "--out_subdir", "inc18_soft_attr",  *ep],

    # ── INCREMENT 19: Set-of-Set decoder-level (WRONG DESIGN — kept for comparison) ────
    #   ⚠ DEPRECATED: --cls_decoder sos splits private/shared AFTER the encoder (H already
    #   mixes all peaks via ISAB → explaining-away already baked in).  Correct design = Inc20.
    #   Kept to measure decoder-level vs encoder-level SoS delta.
    "inc19_sos":          S + GENMASKP + ["--cls_decoder", "sos", "--feas_filter",
                                           "--out_subdir", "inc19_sos", *ep],
    "inc19_sos_softattr": S + GENMASKP + ["--cls_decoder", "sos", "--feas_filter",
                                           "--soft_attr_label", "--out_subdir", "inc19_sos_softattr", *ep],

    # ── INCREMENT 20: Set-of-Set ENCODER-LEVEL (CORRECT DESIGN) ──────────────
    #   --set_of_set splits peaks into private (n_car==1) and shared/unknown (n_car!=1) BEFORE
    #   the ISAB encoder.  Two independent encoder passes (same weights = lite, no extra params),
    #   then element-wise merge.  Private peaks attend ONLY to other private peaks → minor
    #   donor's unambiguous private evidence is no longer drowned by the shared-peak mass.
    #   n_car==0 (panel-unknown alleles, e.g. OOD kit-shift) → routed into shared encoder,
    #   NOT excluded — keeps SoS OOD-safe unlike feas_filter which blinds the decoder to them.
    #
    #   OOD issue with feas_filter: panel LUT built from in-domain genotypes; kit-shifted
    #   alleles (allele calling δ ~0.1) → bin mismatch → n_car=0 → filtered = decoder blind.
    #   inc18_soft_attr was OOD-worst for this reason.  inc20_sos does NOT use feas_filter
    #   by default; inc20_sos_feas adds it as an ablation (in-domain ceiling, OOD floor).
    #
    #   JUDGE: DEV N5 oracle vs maskp baseline .6498 / mab0 .639 + OOD eval (eval_cross_inc3.py).
    #   Guard N1-N4 oracle; 3-seed CI before verdict.
    #   M1 = inc20_sos           (SoS only, OOD-safe — recommended)
    #   M2 = inc20_sos_softattr  (SoS + soft_attr_label — stacks inc18 improvement)
    #   ablation (via RUNS=): inc20_sos_feas (SoS + feas_filter, in-domain upper bound)
    "inc20_sos":          S + GENMASKP + ["--set_of_set",
                                           "--out_subdir", "inc20_sos", *ep],
    "inc20_sos_softattr": S + GENMASKP + ["--set_of_set", "--soft_attr_label",
                                           "--out_subdir", "inc20_sos_softattr", *ep],
    "inc20_sos_feas":     S + GENMASKP + ["--set_of_set", "--feas_filter",
                                           "--out_subdir", "inc20_sos_feas", *ep],

    # ── SMALL-SCALE COMPARISON (d_model=64, m_inducing=16, epochs=60) ────────
    #   Exploratory 3-way: base vs SoS vs AdaptiveSlot (aslot) at half model size.
    #   Purpose: check aslot for collapse / promise WITHOUT full Kaggle compute cost.
    #   JUDGE: compare DEV N5 oracle among the three (relative ordering, not absolute).
    #   Run:  RUNS=sm_base,sm_sos,sm_aslot EPOCHS=60 SEEDS=42 python kaggle_run_increment1.py
    #         (or one per machine: RUNS=sm_base MACHINE=M1, RUNS=sm_sos MACHINE=M2, etc.)
    #
    #   sm_base:  per_donor decoder (current best config, small scale) — control
    #   sm_sos:   encoder-level SoS split (inc20 design), small scale
    #   sm_aslot: CoSA+GSANet+MESH+AdaSlot, small scale; requires aux_heads for GSANet
    #
    #   All three share the same ISAB++ encoder (pe_s3 + sparse), d_model=64.
    #   sm_aslot uses cls_decoder=aslot + aux_heads; GSANet activated automatically.
    "sm_base":  S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++",
                     "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                     "--sparse_attn", "--mask_peaks", "0.15",
                     "--d_model", "64", "--m_inducing", "16",
                     "--epochs", "60", "--out_subdir", "sm_base"],
    "sm_sos":   S + ["--n_token_feats", "8", "--cls_decoder", "per_donor", "--encoder", "isab++",
                     "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                     "--sparse_attn", "--mask_peaks", "0.15",
                     "--d_model", "64", "--m_inducing", "16",
                     "--set_of_set",
                     "--epochs", "60", "--out_subdir", "sm_sos"],
    "sm_aslot": S + ["--n_token_feats", "8", "--cls_decoder", "aslot", "--encoder", "isab++",
                     "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                     "--sparse_attn", "--mask_peaks", "0.15",
                     "--d_model", "64", "--m_inducing", "16",
                     "--n_slot_iters", "3", "--ot_eps", "0.05", "--ot_iters", "5",
                     "--epochs", "60", "--out_subdir", "sm_aslot"],
    "sm_spen":  S + ["--n_token_feats", "8", "--cls_decoder", "spen", "--encoder", "isab++",
                     "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                     "--sparse_attn", "--mask_peaks", "0.15",
                     "--d_model", "64", "--m_inducing", "16",
                     "--n_inf_steps", "10", "--inf_lr", "0.5", "--spen_global_hidden", "64",
                     "--epochs", "60", "--out_subdir", "sm_spen"],

    # ── INCREMENT 21: Full-scale 6-way 150-epoch comparison ───────────────────
    #   M1 = base (control) + spen (EBM joint energy)
    #   M2 = sos (encoder-level SoS) + sos_softattr (SoS + soft attr label)
    #   M3 = aslot (CoSA+GSANet+MESH+AdaSlot) + sos_aslot (SoS encoder + aslot decoder)
    #   Run: MACHINE=M1|M2|M3 python kaggle_run_increment1.py  (3 machines in parallel)
    #   Judge: DEV N5 oracle vs mab0 .639 + guard N1-4; real-test N5 oracle vs .747.
    "inc21_base": S + GENMASKP + ["--soft_geno_attr", "--feas_filter", "--soft_attr_label", "--epochs", "150", "--out_subdir", "inc21_base"],
    "inc21_spen": S + GENMASKP + ["--soft_geno_attr", "--feas_filter", "--soft_attr_label", "--cls_decoder", "spen",
                                   "--n_inf_steps", "10", "--inf_lr", "0.5",
                                   "--spen_global_hidden", "128",
                                   "--epochs", "150", "--out_subdir", "inc21_spen"],
    # ── INCREMENT 21: SoS arms (M2) ────────────────────────────────────────────
    "inc21_sos":          S + GENMASKP + ["--soft_geno_attr", "--feas_filter", "--soft_attr_label", "--set_of_set",
                                           "--epochs", "150", "--out_subdir", "inc21_sos"],
    "inc21_sos_softattr": S + GENMASKP + ["--set_of_set", "--soft_geno_attr", "--feas_filter", "--soft_attr_label",
                                           "--epochs", "150", "--out_subdir", "inc21_sos_softattr"],
    # ── INCREMENT 21: AdaptiveSlot arms (M3) ───────────────────────────────────
    "inc21_aslot":     S + ["--soft_geno_attr", "--feas_filter", "--soft_attr_label", "--n_token_feats", "8", "--cls_decoder", "aslot", "--encoder", "isab++",
                             "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                             "--sparse_attn", "--mask_peaks", "0.15",
                             "--n_slot_iters", "3", "--ot_eps", "0.05", "--ot_iters", "5",
                             "--epochs", "150", "--out_subdir", "inc21_aslot"],
    "inc21_sos_aslot": S + ["--soft_geno_attr", "--feas_filter", "--soft_attr_label", "--n_token_feats", "8", "--cls_decoder", "aslot", "--encoder", "isab++",
                             "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                             "--sparse_attn", "--mask_peaks", "0.15", "--set_of_set",
                             "--n_slot_iters", "3", "--ot_eps", "0.05", "--ot_iters", "5",
                             "--epochs", "150", "--out_subdir", "inc21_sos_aslot"],
    # ── INCREMENT 22: fixed AdaptiveSlot arms ───────────────────────────────────
    "inc22_fixed_aslot": S + ["--noc_head_v2", "--nc_attn", "mab0", "--feas_filter", "--soft_attr_label", 
                             "--n_token_feats", "8", "--cls_decoder", "aslot", "--encoder", "isab++",
                             "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                             "--mask_peaks", "0.15", "--set_of_set",
                             "--n_slot_iters", "3", "--ot_eps", "0.05", "--ot_iters", "5",
                             "--epochs", "150", "--out_subdir", "inc22_fixed_aslot"],
    # ── INCREMENT 22: refmatch AdaptiveSlot arms ───────────────────────────────────
    "inc22_refmatch_aslot": S + ["--ref_match", "--noc_head_v2", "--nc_attn", "mab0","--soft_geno_attr", "--feas_filter", "--soft_attr_label", "--n_token_feats", "8", "--cls_decoder", "aslot", "--encoder", "isab++",
                             "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                             "--sparse_attn", "--mask_peaks", "0.15", "--set_of_set",
                             "--n_slot_iters", "3", "--ot_eps", "0.05", "--ot_iters", "5",
                             "--epochs", "150", "--out_subdir", "inc22_refmatch_aslot"],
    # ── INCREMENT 23: emphi AdaptiveSlot arms ───────────────────────────────────
    "inc23_emphi_aslot": S + ["--em_phi_feature", "--noc_head_v2", "--nc_attn", "mab0","--soft_geno_attr", "--feas_filter", "--soft_attr_label", "--n_token_feats", "8", "--cls_decoder", "aslot", "--encoder", "isab++",
                             "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                             "--sparse_attn", "--mask_peaks", "0.15", "--set_of_set",
                             "--n_slot_iters", "3", "--ot_eps", "0.05", "--ot_iters", "5",
                             "--epochs", "150", "--out_subdir", "inc23_emphi_aslot"],
    # ── INCREMENT LUPI: privileged PHYSICAL supervision on data_lupi (overlay+drop-in regen, leak-safe).
    #   Run with INSILICO_W=data_lupi.  CUMULATIVE ablation (each = previous + one head); eval on REAL test.
    #     v1 = data only (isolate the realism + drop-in effect, no new head)
    #     v2 = + degradation-β head            v3 = + clean-mu denoise head (β-NLL, Seitzer 2022)
    #     v4 = + variance supervision (Gaussian, Nix-Weigend/Kendall-Gal)   v5 = + drop-in (phantom) head
    #   JUDGE: real-test per_noc_oracle / em vs inc22_fixed_aslot (F10: physical supervision → transfer).
    "lupi_v1_data":   S + LUPIB + ["--out_subdir", "lupi_v1_data"],
    "lupi_v2_degr":   S + LUPIB + ["--lupi_phys", "--lupi_degr", "--out_subdir", "lupi_v2_degr"],
    "lupi_v3_mu":     S + LUPIB + ["--lupi_phys", "--lupi_degr", "--lupi_mu", "--out_subdir", "lupi_v3_mu"],
    "lupi_v4_var":    S + LUPIB + ["--lupi_phys", "--lupi_degr", "--lupi_mu", "--lupi_var", "--out_subdir", "lupi_v4_var"],
    "lupi_v5_dropin": S + LUPIB + ["--lupi_phys", "--lupi_degr", "--lupi_mu", "--lupi_var", "--lupi_dropin", "--out_subdir", "lupi_v5_dropin"],
    # ── INCREMENT 24: SOFT per-peak noise/stutter gate (decoy suppression; inc22_fixed + --noise_gate) ──
    "inc24_noisegate_aslot": S + ["--noise_gate", "--noise_gate_w", "0.3", "--noc_head_v2", "--nc_attn", "mab0", "--feas_filter", "--soft_attr_label", "--n_token_feats", "8", "--cls_decoder", "aslot", "--encoder", "isab++",
                             "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                             "--mask_peaks", "0.15", "--set_of_set",
                             "--n_slot_iters", "3", "--ot_eps", "0.05", "--ot_iters", "5",
                             "--epochs", "150", "--out_subdir", "inc24_noisegate_aslot"],
    # ── INCREMENT 25: RICHER DATA — R=3 replicate amplifications pooled per mixture (inc22_fixed + --replicates).
    #    PREREQ: build the _rep3 peak arrays first:  STR_REPLICATES=3 python make_replicates.py  (writes into the data dir).
    "inc25_replicates_aslot": S + ["--replicates", "3", "--noc_head_v2", "--nc_attn", "mab0", "--feas_filter", "--soft_attr_label", "--n_token_feats", "8", "--cls_decoder", "aslot", "--encoder", "isab++",
                             "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                             "--mask_peaks", "0.15", "--set_of_set",
                             "--n_slot_iters", "3", "--ot_eps", "0.05", "--ot_iters", "5",
                             "--epochs", "150", "--out_subdir", "inc25_replicates_aslot"],
    # ── INCREMENT 26: FULL HYBRID — richer data (replicates) feeding BOTH channels + math LOP rerank +
    #    count-on-reranked-score. Math owns deconvolution (phi_rerank, replicate-pooled); neural owns the
    #    cross-locus residual; rerank BEFORE count. PREREQ: STR_REPLICATES=3 python make_replicates.py
    "inc26_hybrid_aslot": S + ["--replicates", "3", "--phi_rerank", "--count_on_rerank",
                             "--noc_head_v2", "--nc_attn", "mab0", "--feas_filter", "--soft_attr_label", 
                             "--n_token_feats", "8", "--cls_decoder", "aslot", "--encoder", "isab++",
                             "--aux_heads", "--num_embed", "periodic", "--periodic_sigma", "0.3",
                             "--mask_peaks", "0.15", "--set_of_set",
                             "--n_slot_iters", "3", "--ot_eps", "0.05", "--ot_iters", "5",
                             "--epochs", "150", "--out_subdir", "inc26_hybrid_aslot"],
}
sub_of = {"inc17_base": "inc17_base", "inc17_ns": "inc17_ns", "inc17_ns_distill": "inc17_ns_distill",
          "inc16_addrecon": "inc16_addrecon",
          "hybrid3": "inc1_hybrid_tok3", "hybrid8": "inc1_hybrid_tok8",
          "st_pd3": "inc1_st_perdonor_tok3", "st_pd8": "inc1_st_perdonor_tok8",
          "st_pd_pp8": "inc1_st_perdonor_tok8_pp", "st_pool8": "inc1_st_pooled_tok8",
          "hybrid8_pp": "inc1_hybrid_tok8_pp", "st_pd_pp3": "inc1_st_perdonor_tok3_pp",
          "st_pd_pp8_pe": "inc1_st_perdonor_tok8_pp_pe", "st_pd8_pe": "inc1_st_perdonor_tok8_pe",
          "hybrid8_pp_pe": "inc1_hybrid_tok8_pp_pe",
          "inc2_base": "inc2_base", "inc2_2a": "inc2_2a_fwdstutter",
          "inc2_2b": "inc2_2b_privsup", "inc2_2ab": "inc2_2ab",
          "inc2_2a_full": "inc2_2a_full", "inc2_2ab_full": "inc2_2ab_full",
          "inc2_2c": "inc2_2c_contrast",
          "inc2_2b_pe05": "inc2_2b_pe_s05", "inc2_2b_pe1": "inc2_2b_pe_s1",
          "inc2_2b_pe3": "inc2_2b_pe_s3", "inc2_2b_pe9": "inc2_2b_pe_tok9",
          "inc2_2b_pe11": "inc2_2b_pe_tok11",
          "inc2_pe3_tok9": "inc2_2b_pe3_tok9", "inc2_pe3_tok11": "inc2_2b_pe3_tok11",
          "inc2_2c_shared": "inc2_2c_pe3_shared", "inc2_2c_detach": "inc2_2c_pe3_detach",
          "inc2_2c_pcgrad": "inc2_2c_pe3_pcgrad",
          "inc2_2c_fix_sh": "inc2_2c_fix_sh", "inc2_2c_fix_pg": "inc2_2c_fix_pg",
          "inc2_2d_sparse": "inc2_2d_sparse", "inc2_2d_irm": "inc2_2d_irm",
          "inc2_2d_irm10": "inc2_2d_irm10",
          "inc3_repA_genoq": "inc3_repA_genoq", "inc3_repB_donorcon": "inc3_repB_donorcon",
          "inc3_repC_additive": "inc3_repC_additive", "inc3_nocV1_ordrnc": "inc3_nocV1_ordrnc",
          "inc3_nocV2_ordrnc_pcgrad": "inc3_nocV2_ordrnc_pcgrad",
          "inc3_nocV3_ordreplace": "inc3_nocV3_ordreplace",
          "inc4_p1_stack": "inc4_p1_stack", "inc4_p2_local": "inc4_p2_local",
          "inc4_p3_irm": "inc4_p3_irm", "inc4_p4_decorr": "inc4_p4_decorr",
          "inc4_p5_noc_intrinsic": "inc4_p5_noc_intrinsic", "inc4_p6_slot": "inc4_p6_slot",
          "inc5_res_rand1": "inc5_res_rand1", "inc5_res_randk": "inc5_res_randk",
          "inc6_maskp": "inc6_maskp", "inc6_andmask": "inc6_andmask",
          "inc6_slotfix": "inc6_slotfix", "inc6_meta": "inc6_meta",
          "inc6_sam": "inc6_sam", "inc6_maskpre": "inc6_maskpre",
          "inc6_vib": "inc6_vib", "inc6_fomaml": "inc6_fomaml",
          "inc6_minorw": "inc6_minorw",
          "inc7_masspool": "inc7_masspool",
          "inc8_v1_vicreg": "inc8_v1_vicreg",
          "inc8_v2_vicreg_inv": "inc8_v2_vicreg_inv",
          "inc9_a1_ldam": "inc9_a1_ldam", "inc9_a2_dbloss": "inc9_a2_dbloss",
          "inc9_a3_bal": "inc9_a3_bal", "inc9_b1_recon": "inc9_b1_recon",
          "inc9_b4_sink": "inc9_b4_sink", "inc9_inter_genoq": "inc9_inter_genoq",
          "inc9_a4_qdenoise": "inc9_a4_qdenoise", "inc9_b2_ifm": "inc9_b2_ifm",
          "inc10_v1_refmatch": "inc10_v1_refmatch", "inc10_v2_refmatch_learn": "inc10_v2_refmatch_learn",
          "inc11_nc_mab0": "inc11_nc_mab0", "inc11_nc_both": "inc11_nc_both",
          "inc12_entmax15": "inc12_entmax15", "inc12_entmax_temp": "inc12_entmax_temp",
          "inc12_dsmil": "inc12_dsmil",
          "gen_orig": "genprop_orig", "gen_real": "genprop_real", "gen_wide": "genprop_wide",
          "inc13_B": "inc13_B_distill", "inc13_A": "inc13_A_cfinv",
          "inc13_C": "inc13_C_ladder", "inc13_ABC": "inc13_ABC",
          "inc14_Bv2": "inc14_Bv2", "inc14_Av2": "inc14_Av2",
          # Inc18: phi injection (V1) + soft genotype-constrained attr (V2), base = inc6_maskp (GENMASKP)
          # inc18_phi_inject dropped (EM-uniform dominates neural phi at current scale)
          "inc18_soft_attr":  "inc18_soft_attr",
          # Inc19: set-of-set decoder-level (WRONG — post-encoder split, kept as control)
          "inc19_sos":          "inc19_sos",
          "inc19_sos_softattr": "inc19_sos_softattr",
          # Inc20: Set-of-Set ENCODER-LEVEL (CORRECT — pre-encoder split, OOD-safe)
          "inc20_sos":          "inc20_sos",
          "inc20_sos_softattr": "inc20_sos_softattr",
          "inc20_sos_feas":     "inc20_sos_feas",
          # Small-scale 3-way comparison
          "sm_base":  "sm_base",
          "sm_sos":   "sm_sos",
          "sm_aslot": "sm_aslot",
          "sm_spen":  "sm_spen",
          # Inc21: full-scale 6-way (150 epochs) — M1: base/spen · M2: sos/sos+attr · M3: aslot/sos+aslot
          "inc21_base":         "inc21_base",
          "inc21_spen":         "inc21_spen",
          "inc21_sos":          "inc21_sos",
          "inc21_sos_softattr": "inc21_sos_softattr",
          "inc21_aslot":        "inc21_aslot",
          "inc21_sos_aslot":    "inc21_sos_aslot",
          "inc22_fixed_aslot":    "inc22_fixed_aslot",
          "inc22_refmatch_aslot":    "inc22_refmatch_aslot",
          "inc23_emphi_aslot":    "inc23_emphi_aslot",
          "lupi_v1_data": "lupi_v1_data", "lupi_v2_degr": "lupi_v2_degr", "lupi_v3_mu": "lupi_v3_mu",
          "lupi_v4_var": "lupi_v4_var", "lupi_v5_dropin": "lupi_v5_dropin",
          "inc24_noisegate_aslot":    "inc24_noisegate_aslot",
          "inc25_replicates_aslot":    "inc25_replicates_aslot",
          "inc26_hybrid_aslot":    "inc26_hybrid_aslot",
          }

# ── Kaggle-machine split (seed 42 first; add seeds later only for arms that look promising) ──
#   MACHINE=M1|M2 picks the arm group; or set RUNS explicitly to override.
#   CURRENT INCREMENT = 5 (residual-aware peeling): only 2 independent trainings → 2 machines
#   (or ONE machine sequential: RUNS=inc5_res_rand1,inc5_res_randk). M1=rand1 · M2=randk.
#   Prior increments are still reachable via explicit RUNS (e.g. RUNS=inc4_p3_irm).
MACHINES = {
    # CURRENT INCREMENT = 11 (F35b): NON-COMPETITIVE SIGMOID ENCODER ATTENTION (Ramapuram 2024). 2 machines.
    #   M1 = inc11_nc_mab0 (sigmoid in mab0 only = the probe-localized lever) ·
    #   M2 = inc11_nc_both  (sigmoid in mab0+mab1 = placement control; probe predicts ≤ M1).
    #   Run:  MACHINE=M1 SEEDS=42 python kaggle_run_increment1.py   (and MACHINE=M2 on a 2nd machine, in PARALLEL)
    #   For 3-seed (after the seed-42 promise-check): SEEDS=42,43,44 MACHINE=M1 ... / SEEDS=42,43,44 MACHINE=M2 ...
    #   JUDGE: DEV N5 oracle vs base ~.59 + guard N1/2/3 oracle (no regression); then on the ckpt re-run
    #   probe_encoder_isolate (does missed-tail →major drop from .72-.88?) and probe_cleanprobe_maxsum (does
    #   the encoder ceiling rise above .80?). Expect the readout 2nd wall (probe6, conjunctive under-weight)
    #   to still need a separate lever even if the encoder ceiling rises.
    # CURRENT INCREMENT = 12 (decoder readout, on the mab0 base): close the +.10 N5 under-read (no info floor,
    #   N5 97% identifiable; mab0 ceiling .849 >> per_donor sparsemax decode .747). Softer/adaptive per-donor
    #   aggregation (keeps per_donor → encoder stays good; additive REJECTED, it collapses the encoder).
    #   M1 = inc12_entmax15, inc12_entmax_temp (entmax ladder: fixed-soft α=1.5 → learnable-temp ASEntmax) ·
    #   M2 = inc12_dsmil (DSMIL dual-stream: max instance + dense bag). JUDGE DEV N5 oracle vs mab0 (.639) +
    #   guard N1/2/3; 3-seed (SEEDS=42,43,44) before any verdict.
    # CURRENT INCREMENT = 13 (generator-proportion 3-way root-cause fix). 3 machines IN PARALLEL:
    #   M1 = gen_orig (original skew, baseline ~0.79) · M2 = gen_real (matches real) · M3 = gen_wide (superset).
    #   Each REGENERATES its own 50k train (make_insilico --build 50000 --noc_weights 1,1.5,2.5,2 --seed 42
    #   with its proportion mode) then trains the inc6_maskp config; eval = real-test per_noc_oracle N5.
    #   Run (3 Kaggle machines, parallel):  MACHINE=M1 python kaggle_run_increment1.py   (M2, M3 likewise)
    #   3-seed after the seed-42 promise-check: SEEDS=42,43,44 MACHINE=M1 ... (and M2, M3).
    #   JUDGE: real-test N5 oracle vs 0.79; expect gen_real & gen_wide > gen_orig; gen_wide ~ gen_real on
    #   real test but more robust OOD. Guard N1-N4 (measure_insilico_oracle writes generalization.test_oracle).
    # CURRENT = INCREMENT 13b: the 3 generalization-lever building blocks, validated ALONE first
    # (1 per machine, in PARALLEL), on the inc6_maskp base + DATA_W. Combined ABC reachable via RUNS=.
    #   Run:  MACHINE=M1 python kaggle_run_increment1.py   (and M2, M3 on 2 more machines)
    #   3-seed after the seed-42 promise-check:  SEEDS=42,43,44 MACHINE=M1 ...
    #   JUDGE: real-test per_noc_oracle N5 vs 0.788 + guard N1-N4. After these, layer MLC/soft-hybrid
    #   for the full matrix (M1: A+B+MLC · M2: A+B+soft-hybrid · M3: C+A+B+MLC) — not yet implemented.
    # CURRENT = INCREMENT 14: forensic-grounded root levers, validated ALONE first, 2 machines IN PARALLEL.
    #   M1 = B-v2 (multi-label genotype co-membership head + MLD distill; teacher-ceiling .82->.96)
    #   M2 = A-v2 (additive-subtraction counterfactual + conditional MMD; fixes A-v1's wrong invariance)
    #   Run:  MACHINE=M1 python kaggle_run_increment1.py   (and M2 on a 2nd machine)
    #   3-seed after seed-42 promise-check:  SEEDS=42,43,44 MACHINE=M1 ...
    #   JUDGE: real-test per_noc_oracle N5 vs base 0.788 + guard N1-N4; 3-seed before verdict.
    # CURRENT = INCREMENT 16/17 (model-side compositional-gen levers on the best config inc6_maskp):
    #   M1 = inc16_addrecon (additive log-recon over present+absent, penalizes decoy inclusion) ·
    #   M2 = inc17_ns (height-aware noise-spectrum + x-noise consistency; NOTE inc17_mlc seed42 was NEGATIVE,
    #        N5 .54 < maskp .788 — keep only as a control). A/B vs existing inc6_maskp_seed42 (DEV N5 .6804 / test N5 .788).
    #   Run:  RUNS=inc16_addrecon python kaggle_run_increment1.py   (or MACHINE=M1). Trains on DATA_W (runner preps).
    #   JUDGE: DEV N5 oracle (generalization.dev_oracle["5"]) vs .6804 + guard N1-N4 + real-test N5 oracle vs .788.
    # INCREMENT 21: Full-scale 6-way comparison (3 machines, 150 epochs, seed42 first)
    #   M1 = base (control per_donor) + SPEN (EBM joint energy decoder)
    #   M2 = sos (encoder-level SoS) + sos_softattr (SoS + soft attr label)
    #   M3 = aslot (CoSA+GSANet+MESH+AdaSlot) + sos_aslot (SoS encoder + aslot decoder)
    #   Run:  MACHINE=M1 python kaggle_run_increment1.py  (M2, M3 likewise in parallel)
    #   Judge: DEV N5 oracle vs mab0 .639 + guard N1-4; real-test N5 oracle vs .747.
    #   Small-scale screen (reachable via RUNS=): sm_base | sm_sos | sm_aslot | sm_spen
    "M1": "inc21_base,inc21_spen",
    "M2": "inc21_sos,inc21_sos_softattr",
    "M3": "inc21_aslot,inc21_sos_aslot",
    # (Inc20 full-scale, reachable via RUNS=): inc20_sos | inc20_sos_softattr | inc20_sos_feas
    # (Inc19 decoder-level control, reachable via RUNS=): inc19_sos | inc19_sos_softattr
    # INCREMENT 18 (reachable via RUNS=): inc18_soft_attr
    # (Inc13 reachable via RUNS=): inc13_C
    # (prev Increment-13 building blocks, reachable via RUNS=): inc13_B | inc13_A | inc13_C | inc13_ABC
    # (Increment-13 generator 3-way, reachable via RUNS=): gen_orig | gen_real | gen_wide
    # (Increment-12 split, reachable via RUNS=): M1=inc12_entmax15 | inc12_entmax_temp | M3=inc12_dsmil
    # (Increment-11 split, reachable via RUNS=): inc11_nc_mab0 (mab0-only) | inc11_nc_both (placement ctl)
    # (Increment-10 split, reachable via RUNS=): M1=inc10_v1_refmatch | M2=inc10_v2_refmatch_learn
    # (Increment-9 split, reachable via RUNS=): M1=inc9_a1_ldam,inc9_a2_dbloss,inc9_a3_bal,inc9_a4_qdenoise |
    #   M2=inc9_b1_recon,inc9_b4_sink,inc9_b2_ifm,inc9_inter_genoq
    # (prev Increment-8 split, reachable via RUNS=): inc8_v1_vicreg | inc8_v2_vicreg_inv
    # (prev Increment-7 mass-pool split, reachable via RUNS=): M1/M2/M3 = inc7_masspool × seeds 42/43/44
    # (prev Increment-6 split, reachable via explicit RUNS=)
    #   M1=inc6_maskp,inc6_andmask,inc6_sam  M2=inc6_vib,inc6_maskpre,inc6_slotfix  M3=inc6_meta,inc6_fomaml
    # ── batch-3 minor-weight (VERIFIED root cause) = SAME arm, 1 seed/machine for a 3-seed CI.
    #   Seed comes from SEEDS env (not MACHINE), so run explicitly:
    #     machine1:  RUNS=inc6_minorw SEEDS=42 python kaggle_run_increment1.py
    #     machine2:  RUNS=inc6_minorw SEEDS=43 python kaggle_run_increment1.py
    #     machine3:  RUNS=inc6_minorw SEEDS=44 python kaggle_run_increment1.py
    #   (compare to maskp .68 dev-N5; eval prints DEV per-NOC oracle via measure_insilico_oracle).
    # (Inc5 split, run with RUNS=...): inc5_res_rand1 | inc5_res_randk
    # (Inc4 split, run with RUNS=...): inc4_p1_stack,inc4_p2_local | inc4_p3_irm,inc4_p4_decorr | inc4_p5_noc_intrinsic,inc4_p6_slot
}
if os.environ.get("MACHINE") in MACHINES and not os.environ.get("RUNS"):
    RUNS = MACHINES[os.environ["MACHINE"]].split(",")
    print(f"MACHINE={os.environ['MACHINE']} -> RUNS={RUNS}  SEEDS={SEEDS}")
# Each (arm × seed) writes results/<base_subdir>_seed<seed>/ (so seeds are aggregatable).
# NOTE suffix is "_seed<N>" (not "_s<N>") to avoid colliding with the sigma dirs (inc2_2b_pe_s05 etc.).
def sub_for(r, seed):
    return f"{sub_of[r.strip()]}_seed{seed}"

# P4 needs a DECORRELATED data dir (balanced donor co-occurrence): build from SRC -> decorrelate ->
# dev-split -> enrich, once, on first use. Trained with STR_DATA_DIR pointed at it.
DATA_DECORR = WORK / "data_decorr"
def ensure_decorr():
    if (DATA_DECORR / "tokens8_dev.npy").exists():
        return
    DATA_DECORR.mkdir(parents=True, exist_ok=True)
    for f in list(SRC.glob("*.npy")) + list(SRC.glob("*.json")):
        shutil.copy(f, DATA_DECORR / f.name)
    for nm in ("donor_geno.npy", "donor_geno_mask.npy"):
        if (DATA_W / nm).exists():
            shutil.copy(DATA_W / nm, DATA_DECORR / nm)
    run([PY, "decorrelate_combos.py", str(DATA_DECORR)])
    run([PY, "make_dev_split.py", str(DATA_DECORR)])
    run([PY, "features/enrich.py", str(DATA_DECORR)])

# Inc5 needs a RESIDUAL-AUGMENTED train dir: build from DATA_W (which already has dev_split + enrich)
# via build_residual_aug.py. dev/val/test copied unchanged; only TRAIN arrays get residual rows.
# Built once per mode on first use. Trained with STR_DATA_DIR pointed at it; eval uses CLEAN DATA_W.
def ensure_residual(mode):
    d = WORK / f"data_res_{mode}"
    if (d / "tokens8_train.npy").exists():
        return d
    run([PY, "build_residual_aug.py", str(DATA_W), str(d), "--mode", mode])
    return d

# Inc13 needs a RE-GENERATED train dir per proportion mode (original|realistic|wide). make_insilico
# rebuilds 50k synthetic mixtures from the SAME real single-source pool (in SRC) with that mode's
# proportion/template model, KEEPS the real SS, and copies the SAME real val/test (so the real-test
# oracle is comparable across modes). Then dev_split + enrich. Built once per mode on first use.
GEN_BUILD = os.environ.get("GEN_BUILD", "50000")            # match the shipped 50k synthetic mixtures
GEN_NOCW  = os.environ.get("GEN_NOCW", "1,1.5,2.5,2")       # match the shipped NOC sampling weights
GEN_SEED  = os.environ.get("GEN_SEED", "42")               # SAME gen seed for all 3 modes (controlled)
def ensure_regen(mode):
    d = WORK / f"data_gen_{mode}"
    if (d / "tokens8_dev.npy").exists():
        return d
    genv = os.environ.copy()
    genv["STR_DATA_DIR"] = str(SRC); genv["STR_OUT_DIR"] = str(d); genv["STR_PROP_MODE"] = mode
    run([PY, "make_insilico.py", "--build", GEN_BUILD, "--noc_weights", GEN_NOCW, "--seed", GEN_SEED], genv)
    run([PY, "make_dev_split.py", str(d)])
    run([PY, "features/enrich.py", str(d)])
    return d

arm_data, arm_is_st = {}, {}   # per-arm data dir + whether it's a train_set_transformer arm
run_subs = []  # (r, seed, subdir) actually launched
for r in RUNS:
    rk = r.strip()
    if rk not in jobs:
        continue
    arm_is_st[rk] = (len(jobs[rk]) > 1 and str(jobs[rk][1]).endswith("train_set_transformer.py"))
    run_env, arm_data[rk] = env, DATA_W
    if rk == "inc4_p4_decorr":
        ensure_decorr()
        run_env = os.environ.copy(); run_env["STR_DATA_DIR"] = str(DATA_DECORR)
        arm_data[rk] = DATA_DECORR
    elif rk in ("inc5_res_rand1", "inc5_res_randk"):
        mode = "rand1" if rk.endswith("rand1") else "randk"
        d_res = ensure_residual(mode)
        run_env = os.environ.copy(); run_env["STR_DATA_DIR"] = str(d_res)
        arm_data[rk] = d_res                      # train on residual-aug; eval_peel uses CLEAN DATA_W
    elif rk in ("gen_orig", "gen_real", "gen_wide"):
        gmode = {"gen_orig": "original", "gen_real": "realistic", "gen_wide": "wide"}[rk]
        d_gen = ensure_regen(gmode)
        run_env = os.environ.copy(); run_env["STR_DATA_DIR"] = str(d_gen)
        arm_data[rk] = d_gen                      # train AND eval on this mode's data (real test is shared)
    # EARLY_ABORT=1 -> fail-fast lever screening (train_set_transformer self-cancels a clearly-failing run).
    # ST arms only (the separate-code arms don't take the flag). Optional BASE_N5 overrides the trajectory ref.
    ea = []
    if os.environ.get("EARLY_ABORT") == "1" and arm_is_st.get(rk):
        ea = ["--early_abort"] + (["--early_abort_base_n5", os.environ["BASE_N5"]] if os.environ.get("BASE_N5") else [])
    for s in SEEDS:
        cmd = list(jobs[rk]) + ea + ["--seed", s, "--out_subdir", sub_for(rk, s)]  # last --out_subdir wins
        run(cmd, run_env)
        run_subs.append((rk, s, sub_for(rk, s)))

# 3b) Increment-2 real eval (A=condition-stratified EM, B=phi/Mx, C=template/Q, D=attribution)
#     for any inc2 arm. Parts B/D auto-skip when the arm has no aux heads.
for rk, s, sub in run_subs:
    sub_dir = WORK / "results" / sub
    # Inc5 trains on residual-aug data but DEV/TEST are clean copies → eval on CLEAN DATA_W
    # (clean single-source G for the peel decode + clean train-oracle reference).
    is_inc5 = sub_of[rk].startswith("inc5")
    d_arm = DATA_W if is_inc5 else arm_data.get(rk, DATA_W)
    if sub_of[rk].startswith(("inc2", "inc3", "inc4", "inc5", "inc6", "inc7", "inc8")) and (sub_dir / "y_test_pred.npy").exists():
        run([PY, "eval_phi_condition.py", "--results", sub, "--data", str(d_arm)])
    # F19 manipulation check: probe RNC arms (those that dumped z_noc_proj) for ordinal structure.
    if (sub_dir / "z_noc_proj_test.npy").exists():
        run([PY, "probe_noc_structure.py", "--results", sub, "--data", str(d_arm),
             "--results_root", str(WORK / "results")])
    # F29 COMBO-GENERALIZATION judge (the decisive metric): per-NOC oracle + count on DEV vs TRAIN vs
    # REAL, written into metrics.json. SetTransformer arms only (P5/P6 self-measure in their scripts).
    if arm_is_st.get(rk) and (sub_dir / "metrics.json").exists():
        try:
            run([PY, "measure_insilico_oracle.py", str(sub_dir), str(d_arm)])
        except Exception as e:
            print("  measure_insilico_oracle skipped:", e)
    # Inc5: GATED RECURSIVE-PEEL decode (the lever) — writes a "peel" block (plain vs peel per-NOC
    # oracle, tau sweep, on DEV + REAL) into metrics.json. Uses CLEAN DATA_W for G + test/dev tokens.
    if is_inc5 and (sub_dir / "metrics.json").exists():
        try:
            run([PY, "eval_peel_decode.py", sub, str(DATA_W)])
        except Exception as e:
            print("  eval_peel_decode skipped:", e)

# 4) Compact comparison from saved metrics (one line per arm × seed)
import json
print("\n=== INCREMENT SUMMARY (per arm × seed: oracle = ranking ceiling; decoded EM; reject) ===")
for rk, s, sub in run_subs:
    mf = WORK / "results" / sub / "metrics.json"
    if mf.exists():
        M = json.load(open(mf))
        pn = M.get("per_noc") or {}
        pno = {k: v.get("em") for k, v in pn.items()} if pn else {}
        ab = M.get("early_abort")
        tag = f"  [ABORTED ep{ab['epoch']}: {ab['reason']}]" if ab else ""
        print(f"  {sub:<30} oracle={M.get('oracle_em')} per_noc_em={pno} "
              f"reject={round(M.get('reject_auroc',0),4) if M.get('reject_auroc') else 'NA'} "
              f"test_EM={M.get('test',{}).get('exact_match')}{tag}")

# 4b) Per-arm aggregation across seeds (mean ± 95% CI) — separates real effects from run noise.
if len(SEEDS) > 1:
    run([PY, "aggregate_seeds.py", "--results", str(WORK / "results"),
         "--arms", ",".join(sub_of[r.strip()] for r in RUNS if r.strip() in jobs)])

print("\nDONE. Download /kaggle/working/results/ and send the metrics.json files back.")

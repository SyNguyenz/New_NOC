"""
kaggle_run_increment1.py — CLEAN orchestrator for the `inc22_fixed_aslot` pipeline, RAW -> OUTPUT, one archive fold.

Replaces kaggle_run_increment1.py for the inc22 case: ONE arm, no MACHINE/decorr/residual/regen/
replicate branching, no 60-arm job table.  Runs the full chain:

  [--from-raw]  data_raw/  ->  (proven extractors)  ->  data/  -> make_insilico ->  data_insilico_w/
  (always)      data_insilico_w/  -> copy ->  data_w/  -> make_dev_split -> features/enrich
  (always)      train_set_transformer.py (Phase 1 or --init_from -> phi-rerank ID + NoC count (decoder LoRA) -> fold report)
  -> results/<out_subdir>_seed<seed>/{best_model.pt, best_model.fold.json, metrics.json, report.json, ...}
  -> outputs/fold<F>/  fold<F>_seed<S>.json  metrics.json  best_model.pt  best_model.fold.json

The DATA-PREP scripts (extract_phi_condition, synth/extract_genotypes, build_real_attr, extract_size,
make_insilico, make_dev_split, features/enrich) are the PROVEN, shared dataset generators — they are
INVOKED UNCHANGED from the project root (not rewritten), because they define the exact in-silico
dataset and rewriting them would change the data / break reproducibility.  The clean REWRITE is the
inc22 training + model (train_set_transformer.py, models/set_transformer.py), which is where the complexity lived.

Usage:
  # full chain from the PROVEDIt CSVs (needs data_raw/; rebuilds data/ + data_insilico_w/):
  python kaggle_run_increment1.py --from-raw --seed 42
  # one archive fold from its dataset (what the Kaggle notebooks run; ~3.9 h on a GPU):
  python kaggle_run_increment1.py --fold 0 --data /kaggle/input/noc-inc22-fold0
  # same fold, reusing its saved Phase 1 backbone (~40 min):
  python kaggle_run_increment1.py --fold 1 --data /kaggle/input/noc-inc22-fold1 --init_from <best_model.pt>
ENV (optional, Kaggle): INSILICO_W=<dir to data_insilico_w>  WORK_DIR=<writable root for data_w/outputs>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # inc22_clean/
PROJ = HERE                                      # SELF-CONTAINED: prep scripts + data dirs live IN inc22_clean/
PY = sys.executable

SRC  = Path(os.environ.get("INSILICO_W", str(PROJ / "data_insilico_w")))   # the built in-silico dataset
WORK = Path(os.environ.get("WORK_DIR", str(PROJ)))                         # writable root
DATA_W = WORK / "data_w_inc22"                                             # enriched, dev-split working copy


def run(cmd, env=None, cwd=PROJ):
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    env = dict(env if env is not None else os.environ)
    env["PYTHONUNBUFFERED"] = "1"          # child logs arrive in order, not after the buffer fills
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env, check=True)


def find_fold_dir(data):
    """The dataset root, or the one sub-directory under it that holds fold_info.json."""
    data = Path(data)
    if (data / "fold_info.json").exists():
        return data
    hits = sorted(p.parent for p in data.rglob("fold_info.json"))
    if len(hits) != 1:
        raise SystemExit(f"--data {data}: expected one fold_info.json, found {len(hits)}: {hits}")
    return hits[0]


def stage_raw_to_insilico():
    """data_raw/ -> data/ (real phi/condition/genotypes/attr/size) -> make_insilico -> data_insilico_w/.
    Exactly the proven LOCAL prep sequence (kaggle_run_increment1.py header)."""
    assert (PROJ / "data_raw").exists(), f"data_raw/ not found at {PROJ/'data_raw'} (needs the PROVEDIt CSVs)"
    run([PY, "prepare_data_set.py"])               # raw GF29cycles CSVs -> base tokens/mask/Xflat/y/noc/meta_set
    run([PY, "build_donor_geno.py"])               # raw Known-Genotypes xlsx -> donor_geno.npy + donor_geno_mask.npy
    run([PY, "extract_phi_condition.py"])          # real phi/condition/template/Q  -> data/
    run([PY, "synth/extract_genotypes.py"])        # consensus donor genotypes      -> data/donor_geno*.npy
    run([PY, "build_real_attr.py"])                # real allele->donor labels       -> attr_{val,test}
    run([PY, "extract_size.py"])                   # real per-peak size(bp)          -> size_*
    env = os.environ.copy()
    env["STR_DATA_DIR"] = str(PROJ / "data"); env["STR_OUT_DIR"] = str(SRC)
    run([PY, "make_insilico.py", "--build", "50000", "--noc_weights", "1,1.5,2.5,2", "--seed", "42"], env=env)
    print(f"[from-raw] built in-silico dataset -> {SRC}")


def stage_prep_data_w(src):
    """Copy the built dataset to a WRITABLE dir, carve the combo-disjoint DEV split, build tokens8."""
    assert (src / "meta_set.json").exists(), f"in-silico dataset not found at {src} (run with --from-raw first)"
    if DATA_W.exists():
        shutil.rmtree(DATA_W)                      # a previous fold must not leak files into this one
    DATA_W.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.npy"):
        shutil.copy(f, DATA_W / f.name)
    for f in src.glob("*.json"):
        shutil.copy(f, DATA_W / f.name)
    for nm in ("donor_geno.npy", "donor_geno_mask.npy"):                    # CoSA/feas_filter need these
        for cand in (src / nm, PROJ / "data" / nm, PROJ / nm):
            if cand.exists():
                shutil.copy(cand, DATA_W / nm); break
    print(f"copied dataset -> {DATA_W}")
    run([PY, "make_dev_split.py", str(DATA_W)])    # carve combo-disjoint balanced DEV (selection set)
    run([PY, "features/enrich.py", str(DATA_W)])   # build tokens8_{train,val,test,open,dev}.npy


def stage_train(seed, out_subdir, init_from=None, extra=()):
    env = os.environ.copy(); env["STR_DATA_DIR"] = str(DATA_W)
    cmd = [PY, "train_set_transformer.py", "--seed", str(seed), "--out_subdir", out_subdir] + list(extra)
    if init_from:
        cmd += ["--init_from", str(init_from)]
    run(cmd, env=env, cwd=HERE)


def collect(seed, out_subdir, fold):
    """Copy the fold's report and backbone to outputs/fold<F>/ and fail loudly if a check did not pass."""
    res = HERE / "results" / f"{out_subdir}_seed{seed}"
    rep = json.load(open(res / "report.json"))
    out = WORK / "outputs" / f"fold{rep['fold']}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(res / "report.json", out / f"fold{rep['fold']}_seed{seed}.json")
    for nm in ("metrics.json", "best_model.pt", "best_model.fold.json", "noc_lora_proba.npy"):
        if (res / nm).exists():
            shutil.copy(res / nm, out / nm)
    c = rep["checks"]
    ok = ((fold is None or rep["fold"] == fold) and c["params_unchanged"] and c["split_audit_shared_combos"] == 0
          and c["lora_folds_combination_disjoint"] and rep["noc"]["source"].startswith("noc_"))
    print(f"\n{'=' * 72}\nFOLD {rep['fold']} DONE | CODE {rep['code']} | checks {'OK' if ok else 'FAILED'} | "
          f"backbone stamp {'present' if (out / 'best_model.fold.json').exists() else 'MISSING'}")
    print(f"  NoC macroF1 {rep['noc']['macro_f1']} | acc {rep['noc']['accuracy']} | F1 NOC1..5 "
          + " ".join(str(rep['noc']['per_noc'][str(v)]['f1']) for v in range(1, 6)))
    print(f"  ID  oracle EM {rep['id']['oracle_em']['overall']} | EM@k {rep['id']['em_at_pred_k']['overall']} | "
          f"donor F1@k {rep['id']['donor_at_pred_k']['overall']['f1']} | reject AUROC {rep['id']['reject_auroc']}")
    print(f"  outputs -> {out}: " + ", ".join(sorted(p.name for p in out.iterdir())))
    if not ok:
        raise SystemExit("fold checks FAILED — do not use this report")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean inc22_fixed_aslot pipeline, raw -> output, one archive fold.")
    ap.add_argument("--from-raw", action="store_true",
                    help="rebuild data/ + data_insilico_w/ from data_raw/ (PROVEDIt CSVs) first")
    ap.add_argument("--data", type=str, default=None,
                    help="fold dataset dir holding fold_info.json (default: INSILICO_W / data_insilico_w)")
    ap.add_argument("--fold", type=int, default=None,
                    help="archive fold 0..9; must match the dataset's fold_info.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_subdir", type=str, default=None, help="default: fold<F>")
    ap.add_argument("--init_from", type=str, default=None,
                    help="THIS fold's Phase 1 best_model.pt (best_model.fold.json beside it); skips Phase 1")
    ap.add_argument("--skip-prep", action="store_true",
                    help="reuse an existing data_w_inc22/ (skip copy/dev-split/enrich)")
    ap.add_argument("--epochs", type=int, default=150, help="Phase 1 epochs (lower only for smoke tests)")
    # NO choices= here on purpose: the trainer owns the list and validates it. A restricted list here
    # rejected --count/--compare gate_cal twice after the trainer had already gained it (2026-09-30).
    ap.add_argument("--count", default="lora_inv",
                    help="cach dem, do train_set_transformer.py dinh nghia (clone, clone_inv, lora, "
                         "lora_inv, gate_cal, ...); mac dinh lora_inv")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="cach dem chay them tren cung backbone, chi bao cao (vd --compare gate_cal)")
    ap.add_argument("--lora_kfold", type=int, default=5, help="K of the combination-disjoint count folds")
    ap.add_argument("--lora_steps", type=int, default=3000)
    ap.add_argument("--lora_scope", choices=["all", "encoder", "encoder_attn", "decoder"], default="decoder")
    ap.add_argument("--lora_epochs", type=int, default=5)
    ap.add_argument("--inner_val", action="store_true")
    # every other train_set_transformer.py flag (--count_decode, ...) passes through
    # Unknown flags go straight to train_set_transformer.py: a new trainer flag then works from here
    # without editing this file too (2026-09-23: a new trainer flag died here).
    args, passthrough = ap.parse_known_args()

    if args.from_raw:
        stage_raw_to_insilico()
    src = find_fold_dir(args.data) if args.data else SRC
    fi = json.load(open(src / "fold_info.json")) if (src / "fold_info.json").exists() else {}
    if args.fold is not None and fi.get("fold") != args.fold:
        raise SystemExit(f"--fold {args.fold} but {src}/fold_info.json says fold {fi.get('fold')}: wrong dataset attached")
    print(f"fold {fi.get('fold')}/{fi.get('n_folds')} | unknown donors {fi.get('unknown_donors')} | data {src}")
    if args.init_from:
        bb = Path(args.init_from)
        bb = bb / "best_model.pt" if bb.is_dir() else bb
        if not bb.exists():
            raise SystemExit(f"--init_from not found: {bb}")
        print(f"backbone {bb} | stamp {'present' if bb.with_suffix('.fold.json').exists() else 'MISSING (fold cannot be verified)'}")
    if not args.skip_prep:
        stage_prep_data_w(src)
    out_subdir = args.out_subdir or f"fold{fi.get('fold', 'NA')}"
    stage_train(args.seed, out_subdir, args.init_from,
                ["--epochs", str(args.epochs), "--count", args.count,
                 "--count_kfold", str(args.lora_kfold), "--lora_steps", str(args.lora_steps),
                 "--lora_scope", args.lora_scope, "--lora_epochs", str(args.lora_epochs)]
                + (["--inner_val"] if args.inner_val else []) + (["--compare", *args.compare] if args.compare else [])
                + passthrough)
    collect(args.seed, out_subdir, args.fold)
    print("\nDONE.")

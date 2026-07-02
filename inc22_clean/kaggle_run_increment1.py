"""
kaggle_run_increment1.py — CLEAN orchestrator for the `inc22_fixed_aslot` pipeline, RAW -> OUTPUT.

Replaces kaggle_run_increment1.py for the inc22 case: ONE arm, no MACHINE/decorr/residual/regen/
replicate branching, no 60-arm job table.  Runs the full chain:

  [--from-raw]  data_raw/  ->  (proven extractors)  ->  data/  -> make_insilico ->  data_insilico_w/
  (always)      data_insilico_w/  -> copy ->  data_w/  -> make_dev_split -> features/enrich
  (always)      train_set_transformer.py (model + deployable phi-rerank -> post-hoc-RF-count decode)
  -> results/<out_subdir>_seed<seed>/{best_model.pt, metrics.json, y_test_pred.npy, y_test_true.npy}

The DATA-PREP scripts (extract_phi_condition, synth/extract_genotypes, build_real_attr, extract_size,
make_insilico, make_dev_split, features/enrich) are the PROVEN, shared dataset generators — they are
INVOKED UNCHANGED from the project root (not rewritten), because they define the exact in-silico
dataset and rewriting them would change the data / break reproducibility.  The clean REWRITE is the
inc22 training + model (train_set_transformer.py, models/set_transformer.py), which is where the complexity lived.

Usage:
  # full chain from the PROVEDIt CSVs (needs data_raw/; rebuilds data/ + data_insilico_w/):
  python inc22_clean/kaggle_run_increment1.py --from-raw --seed 42
  # from the already-built data_insilico_w (the common case; what Kaggle training uses):
  python inc22_clean/kaggle_run_increment1.py --seed 42
ENV (optional, Kaggle): INSILICO_W=<dir to data_insilico_w>  WORK_DIR=<writable root for data_w/results>
"""
import argparse
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
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env, check=True)


def stage_raw_to_insilico():
    """data_raw/ -> data/ (real phi/condition/genotypes/attr/size) -> make_insilico -> data_insilico_w/.
    Exactly the proven LOCAL prep sequence (kaggle_run_increment1.py header)."""
    assert (PROJ / "data_raw").exists(), f"data_raw/ not found at {PROJ/'data_raw'} (needs the PROVEDIt CSVs)"
    run([PY, "data/prepare_data_set.py"])          # raw GF29cycles CSVs -> base tokens/mask/Xflat/y/noc/meta_set
    run([PY, "build_donor_geno.py"])               # raw Known-Genotypes xlsx -> donor_geno.npy + donor_geno_mask.npy
    run([PY, "extract_phi_condition.py"])          # real phi/condition/template/Q  -> data/
    run([PY, "synth/extract_genotypes.py"])        # consensus donor genotypes      -> data/donor_geno*.npy
    run([PY, "build_real_attr.py"])                # real allele->donor labels       -> attr_{val,test}
    run([PY, "extract_size.py"])                   # real per-peak size(bp)          -> size_*
    env = os.environ.copy()
    env["STR_DATA_DIR"] = str(PROJ / "data"); env["STR_OUT_DIR"] = str(SRC)
    run([PY, "make_insilico.py", "--build", "50000", "--noc_weights", "1,1.5,2.5,2", "--seed", "42"], env=env)
    print(f"[from-raw] built in-silico dataset -> {SRC}")


def stage_prep_data_w():
    """Copy the built dataset to a WRITABLE dir, carve the combo-disjoint DEV split, build tokens8."""
    assert (SRC / "meta_set.json").exists(), f"in-silico dataset not found at {SRC} (run with --from-raw first)"
    DATA_W.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob("*.npy"):
        shutil.copy(f, DATA_W / f.name)
    for f in SRC.glob("*.json"):
        shutil.copy(f, DATA_W / f.name)
    for nm in ("donor_geno.npy", "donor_geno_mask.npy"):                    # CoSA/feas_filter need these
        for cand in (SRC / nm, PROJ / "data" / nm, PROJ / nm):
            if cand.exists():
                shutil.copy(cand, DATA_W / nm); break
    print(f"copied dataset -> {DATA_W}")
    run([PY, "make_dev_split.py", str(DATA_W)])    # carve combo-disjoint balanced DEV (selection set)
    run([PY, "features/enrich.py", str(DATA_W)])   # build tokens8_{train,val,test,open,dev}.npy


def stage_train(seed, out_subdir):
    env = os.environ.copy(); env["STR_DATA_DIR"] = str(DATA_W)
    run([PY, "train_set_transformer.py", "--seed", str(seed), "--out_subdir", out_subdir], env=env, cwd=HERE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean inc22_fixed_aslot pipeline, raw -> output.")
    ap.add_argument("--from-raw", action="store_true",
                    help="rebuild data/ + data_insilico_w/ from data_raw/ (PROVEDIt CSVs) first")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_subdir", type=str, default="inc22_fixed_aslot")
    ap.add_argument("--skip-prep", action="store_true",
                    help="reuse an existing data_w_inc22/ (skip copy/dev-split/enrich)")
    args = ap.parse_args()

    if args.from_raw:
        stage_raw_to_insilico()
    if not args.skip_prep:
        stage_prep_data_w()
    stage_train(args.seed, args.out_subdir)
    print("\nDONE.")

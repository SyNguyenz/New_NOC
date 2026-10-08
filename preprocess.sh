#!/usr/bin/env bash
# Preprocess one or more folds from data_raw and keep each fold's dataset in data/foldK (next to code/).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$ROOT/code"
OUT="$ROOT/data"

usage() {
    cat <<EOF
Usage: bash preprocess.sh [--override] <fold> [<fold> ...]

  <fold> can be:
    K        one fold, 0..9                 bash preprocess.sh 0
    K L M    several folds                  bash preprocess.sh 0 2 5
    X-Y      folds X to Y (inclusive)       bash preprocess.sh 3-6
    all      every fold, 0 to 9             bash preprocess.sh all
  Forms can be mixed:                       bash preprocess.sh 0 3-5 9

  A fold whose data/foldK already exists is SKIPPED.
  --override  rebuild every listed fold and overwrite data/foldK

Each fold is rebuilt from data_raw by code/preprocess.py (~30 min per fold) and its
data_insilico_w is copied to data/foldK. Logs go to data/logs/foldK.log.
Python: \$PY if set, else venv/, else python3 / python.
Kit: STR_KIT (default 3500_GF29cycles), e.g. STR_KIT=3500_F6C29cycles_hlfrxn bash preprocess.sh 0
  (data/foldK is per run - keep different kits apart).
EOF
}

[ $# -eq 0 ] && { usage; exit 0; }

# python
if [ -n "${PY:-}" ]; then :
elif [ -x "$ROOT/venv/Scripts/python.exe" ]; then PY="$ROOT/venv/Scripts/python.exe"
elif [ -x "$ROOT/venv/bin/python" ]; then PY="$ROOT/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python
fi

# folds from the arguments
FOLDS=(); OVERRIDE=0
for a in "$@"; do
    if [ "$a" = "--override" ]; then
        OVERRIDE=1
    elif [ "$a" = "all" ]; then
        for k in $(seq 0 9); do FOLDS+=("$k"); done
    elif [[ "$a" =~ ^[0-9]+$ ]]; then
        FOLDS+=("$a")
    elif [[ "$a" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        x=${BASH_REMATCH[1]}; y=${BASH_REMATCH[2]}
        [ "$x" -le "$y" ] || { echo "error: range '$a' must be low-high"; exit 1; }
        for k in $(seq "$x" "$y"); do FOLDS+=("$k"); done
    else
        echo "error: bad argument '$a'"; echo; usage; exit 1
    fi
done
[ ${#FOLDS[@]} -gt 0 ] || { echo "error: no fold given"; echo; usage; exit 1; }
for k in "${FOLDS[@]}"; do
    [ "$k" -ge 0 ] && [ "$k" -le 9 ] || { echo "error: fold $k is outside 0..9"; exit 1; }
done
FOLDS=($(printf "%s\n" "${FOLDS[@]}" | awk '!seen[$0]++'))        # drop repeats, keep order

mkdir -p "$OUT/logs"
echo "python: $PY"
echo "folds : ${FOLDS[*]}  ->  $OUT/fold<K>$([ $OVERRIDE -eq 1 ] && echo '  (--override)')"

for k in "${FOLDS[@]}"; do
    if [ $OVERRIDE -eq 0 ] && [ -d "$OUT/fold$k" ]; then
        echo "=== fold $k  skipped: $OUT/fold$k already exists (--override to rebuild)"
        continue
    fi
    log="$OUT/logs/fold$k.log"
    echo "=== fold $k  start $(date +%H:%M:%S)  (log: $log)"
    ( cd "$CODE" && STR_FOLD="$k" PYTHONUNBUFFERED=1 "$PY" preprocess.py ) > "$log" 2>&1 \
        || { echo "fold $k FAILED - last lines of $log:"; tail -20 "$log"; exit 1; }
    got=$(cd "$CODE" && "$PY" -c "import json; print(json.load(open('data/fold_info.json'))['fold'])")
    [ "$got" = "$k" ] || { echo "fold $k: data/fold_info.json says fold $got - stopping"; exit 1; }
    rm -rf "$OUT/fold$k.tmp"                                        # copy, then rename: data/foldK only exists
    cp -r "$CODE/data_insilico_w" "$OUT/fold$k.tmp" || { echo "fold $k: copy failed"; exit 1; }   # once complete
    rm -rf "$OUT/fold$k"
    mv "$OUT/fold$k.tmp" "$OUT/fold$k"
    echo "=== fold $k  done  $(date +%H:%M:%S)  -> $OUT/fold$k ($(du -sh "$OUT/fold$k" | cut -f1))"
done
echo "all done"

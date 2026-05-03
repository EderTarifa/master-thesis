#!/usr/bin/env bash
# scripts/launch_full_optimal.sh
# Usage: nohup taskpolicy -t 0 bash scripts/launch_full_optimal.sh 13 > logs/full_optimal/main.log 2>&1 &
set -euo pipefail

# Forzar locale C para evitar problemas con printf y separador decimal
export LC_ALL=C
export LANG=C

JOBS="${1:-13}"
CONFIG="${2:-src/configs/full_optimal.yaml}"
RUN_NAME="full_optimal"

MARKETS=(DJIA SP50 IBEX)
VARIANTS=(V0 V1 V2 V3 V4)
SEEDS=(0 1 2 3 4)
N_FOLDS=13

TOTAL_TASKS=$((${#MARKETS[@]} * N_FOLDS * ${#VARIANTS[@]} * ${#SEEDS[@]}))
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${RUN_NAME}"
echo "Config: ${CONFIG}"
echo "Parallel jobs: ${JOBS}"
echo "Total tasks: ${TOTAL_TASKS}"

parallel --jobs "$JOBS" --bar --joblog "logs/${RUN_NAME}/parallel.log" \
  python scripts/run_one.py \
    --config "${CONFIG}" \
    --market {1} --fold {2} --variant {3} --seed {4} \
    --out "results/${RUN_NAME}" \
    ">" "logs/${RUN_NAME}/{1}_f{2}_{3}_s{4}.log" "2>&1" \
  ::: "${MARKETS[@]}" \
  ::: $(seq 0 $((N_FOLDS-1))) \
  ::: "${VARIANTS[@]}" \
  ::: "${SEEDS[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training complete. Aggregating results..."

python -c "
import pandas as pd, glob, os, sys
rows_dir = 'results/${RUN_NAME}/rows'
files = sorted(glob.glob(os.path.join(rows_dir, '*.parquet')))
if not files:
    print(f'Warning: no parquet files found in {rows_dir}', file=sys.stderr)
    sys.exit(0)
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
out_path = 'results/${RUN_NAME}/runs.parquet'
df.to_parquet(out_path)
print(f'Aggregated {len(df)} rows from {len(files)} files -> {out_path}')
"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
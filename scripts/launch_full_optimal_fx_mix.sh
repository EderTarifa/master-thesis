#!/usr/bin/env bash
# scripts/launch_full_optimal_fx_mix.sh
# Lanza full_optimal SOLO para FX_MIX (universo de divisas).
#
# Uso:
#   nohup bash scripts/launch_full_optimal_fx_mix.sh 13 \
#         > logs/full_optimal/main_fx_mix.log 2>&1 &
#   disown
set -euo pipefail
export LC_ALL=C
export LANG=C

JOBS="${1:-13}"
CONFIG="${2:-src/configs/full_optimal_fx_mix.yaml}"
RUN_NAME="full_optimal"

MARKETS=(FX_MIX)
VARIANTS=(V0 V1 V2 V3 V4)
SEEDS=(0 1 2 3 4)
N_FOLDS=13   # FX majors disponibles desde 2003-2004 en Yahoo

mkdir -p "logs/${RUN_NAME}"

TOTAL_TASKS=$((${#MARKETS[@]} * N_FOLDS * ${#VARIANTS[@]} * ${#SEEDS[@]}))
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${RUN_NAME} (FX_MIX only)"
echo "Config: ${CONFIG}"
echo "Total tasks: ${TOTAL_TASKS}"

parallel --jobs "$JOBS" --bar --joblog "logs/${RUN_NAME}/parallel_fx_mix.log" \
  python scripts/run_one.py \
    --config "${CONFIG}" \
    --market {1} --fold {2} --variant {3} --seed {4} \
    --out "results/${RUN_NAME}" \
    ">" "logs/${RUN_NAME}/{1}_f{2}_{3}_s{4}.log" "2>&1" \
  ::: "${MARKETS[@]}" \
  ::: $(seq 0 $((N_FOLDS-1))) \
  ::: "${VARIANTS[@]}" \
  ::: "${SEEDS[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] FX_MIX training complete. Re-aggregating runs.parquet..."

python -c "
import pandas as pd, glob, os
rows_dir = 'results/${RUN_NAME}/rows'
files = sorted(glob.glob(os.path.join(rows_dir, '*.parquet')))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df.to_parquet('results/${RUN_NAME}/runs.parquet')
print(f'Aggregated {len(df)} rows ({df[\"market\"].value_counts().to_dict()}) -> runs.parquet')
"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
#!/usr/bin/env bash
# Usage: bash scripts/launch_parallel.sh
# Requires GNU parallel: sudo apt install parallel
set -euo pipefail

MARKETS=(DJIA SP50 IBEX)
VARIANTS=(V0 V1 V2 V3 V4)
SEEDS=(0 1 2 3 4)
N_FOLDS=13         # generate_splits over 2008-2025 with the default schedule
JOBS=6             # 6 parallel processes (you have 12 logical cores)

mkdir -p logs results/full/rows

parallel --jobs "$JOBS" --bar --joblog logs/parallel.log \
  python scripts/run_one.py \
    --market {1} --fold {2} --variant {3} --seed {4} \
    ">" "logs/{1}_f{2}_{3}_s{4}.log" "2>&1" \
  ::: "${MARKETS[@]}" \
  ::: $(seq 0 $((N_FOLDS-1))) \
  ::: "${VARIANTS[@]}" \
  ::: "${SEEDS[@]}"

# Concatenate all per-row parquets into a single runs.parquet.
python -c "
import pandas as pd, glob
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('results/full/rows/*.parquet'))],
               ignore_index=True)
df.to_parquet('results/full/runs.parquet')
print(f'Combined: {len(df)} rows')
"
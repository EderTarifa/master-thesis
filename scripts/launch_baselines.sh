#!/usr/bin/env bash
# scripts/launch_baselines.sh
#
# Launches POTPG + EXDRL baselines using GNU parallel.
#
# Output layout:
#   results/baselines/rows/{market}_f{fold}_{baseline}_s{seed}.parquet
#   results/baselines/telemetry/{market}_f{fold}_{baseline}_s{seed}.parquet
#   logs/baselines/{market}_f{fold}_{baseline}_s{seed}.log

set -euo pipefail
export LC_ALL=C LANG=C

JOBS="${1:-15}"
CONFIG="${2:-src/configs/baselines.yaml}"
OUTDIR="results/baselines"
LOGDIR="logs/baselines"

MARKETS=(${MARKETS:-DJIA SP50 IBEX})
BASELINES=(${BASELINES:-POTPG EXDRL})
SEEDS=(${SEEDS:-0 1 2 3 4})
FOLDS=(${FOLDS:-0 1 2 3 4 5 6 7 8 9 10 11 12})

mkdir -p "$OUTDIR/rows" "$OUTDIR/telemetry" "$LOGDIR"

TOTAL=$((${#MARKETS[@]} * ${#FOLDS[@]} * ${#BASELINES[@]} * ${#SEEDS[@]}))
echo "[$(date '+%F %T')] Starting baselines run"
echo "  markets  : ${MARKETS[*]}"
echo "  baselines: ${BASELINES[*]}"
echo "  seeds    : ${SEEDS[*]}"
echo "  folds    : ${FOLDS[*]}"
echo "  parallel : $JOBS"
echo "  total    : $TOTAL cells"
echo "  config   : $CONFIG"
echo "  outdir   : $OUTDIR  (rows/, telemetry/)"
echo "  logdir   : $LOGDIR"

run_cell () {
    local market=$1 fold=$2 baseline=$3 seed=$4
    local logfile="$LOGDIR/${market}_f${fold}_${baseline}_s${seed}.log"
    python scripts/run_one_baseline.py \
        --market "$market" --fold "$fold" \
        --baseline "$baseline" --seed "$seed" \
        --config "$CONFIG" --out "$OUTDIR" \
        >"$logfile" 2>&1
}
export -f run_cell
export OUTDIR LOGDIR CONFIG

parallel --bar --jobs "$JOBS" --joblog "$LOGDIR/parallel.log" --colsep ' ' \
    run_cell {1} {2} {3} {4} \
    ::: "${MARKETS[@]}" \
    ::: "${FOLDS[@]}" \
    ::: "${BASELINES[@]}" \
    ::: "${SEEDS[@]}"

echo "[$(date '+%F %T')] Aggregating into $OUTDIR/runs.parquet ..."
python - <<'PYEOF'
import os
from pathlib import Path
import pandas as pd
out = Path(os.environ["OUTDIR"])
rows = sorted((out / "rows").glob("*.parquet"))
if not rows:
    print(f"ERROR: no rows in {out}/rows/"); raise SystemExit(1)
df = pd.concat([pd.read_parquet(p) for p in rows], ignore_index=True)
df.to_parquet(out / "runs.parquet")
print(f"  {len(df)} runs aggregated -> {out/'runs.parquet'}")
print()
print(df.groupby(['market','variant'])['mdd'].agg(['mean','std','count']))
PYEOF
echo "[$(date '+%F %T')] Done."

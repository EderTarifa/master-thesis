#!/usr/bin/env bash
# Sensibilidad a lambda_1 y lambda_2.
#
# Uso:
#   bash scripts/launch_sens_lambda.sh
#   JOBS=8 bash scripts/launch_sens_lambda.sh
set -euo pipefail

# Forzar locale C para evitar problemas con printf y separador decimal
export LC_ALL=C
export LANG=C

JOBS="${JOBS:-10}"
CONFIG="src/configs/intermediate_B.yaml"
OUT="results/sens_lambda"
LOGS="logs/sens_lambda"

mkdir -p "$LOGS" "${OUT}/rows"

JOBLIST=$(mktemp)
trap "rm -f $JOBLIST" EXIT

LAMBDAS_DD=(0.5 1.0 2.0 5.0)
LAMBDAS_EVT=(0.1 0.5 1.0 2.0 5.0)
SEEDS=(0 1 2)

for s in "${SEEDS[@]}"; do
    # V1: solo lambda_dd
    for l1 in "${LAMBDAS_DD[@]}"; do
        l1f=$(printf '%.2f' "$l1")
        marker="${OUT}/rows/DJIA_f00_V1_s${s}_l1_${l1f}_l2_0.00.parquet"
        if [ -f "$marker" ]; then continue; fi
        echo "DJIA 0 V1 $s $l1 0.0" >> "$JOBLIST"
    done
    # V4: lambda_dd x lambda_evt
    for l1 in "${LAMBDAS_DD[@]}"; do
        for l2 in "${LAMBDAS_EVT[@]}"; do
            l1f=$(printf '%.2f' "$l1")
            l2f=$(printf '%.2f' "$l2")
            marker="${OUT}/rows/DJIA_f00_V4_s${s}_l1_${l1f}_l2_${l2f}.parquet"
            if [ -f "$marker" ]; then continue; fi
            echo "DJIA 0 V4 $s $l1 $l2" >> "$JOBLIST"
        done
    done
done

PENDING=$(wc -l < "$JOBLIST" | tr -d ' ')
echo "Pending: $PENDING entrenamientos"
echo "JOBS: $JOBS"
echo "Output: $OUT"

if [ "$PENDING" -eq 0 ]; then
    echo "Nothing to run."
    exit 0
fi

parallel --jobs "$JOBS" --bar --joblog "${LOGS}/parallel.log" --colsep ' ' \
    "python scripts/run_one_sens.py \
        --market {1} --fold {2} --variant {3} --seed {4} \
        --lambda-dd {5} --lambda-evt {6} \
        --total-timesteps 100000 \
        --config $CONFIG --out $OUT \
        > ${LOGS}/{1}_f{2}_{3}_s{4}_l1_{5}_l2_{6}.log 2>&1" \
    :::: "$JOBLIST"

echo
echo "Done. Analizar con:"
echo "  python scripts/analyze_sens_lambda.py --rows-dir ${OUT}/rows --out-dir ${OUT}/analysis"
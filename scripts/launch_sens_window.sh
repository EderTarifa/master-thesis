#!/usr/bin/env bash
# scripts/launch_sens_window.sh
# Sensibilidad a la ventana de estimación EVT (W).
#
# Diseño:
#   - 1 mercado (DJIA), 3 folds estratégicos (0=2013 tranquilo, 5=2018 volátil, 9=2022 bear)
#   - V4 con lambda_dd=2.0, lambda_evt=2.0 (óptimos del sensitivity)
#   - 3 semillas
#   - W in {120, 250, 500} días
# Total: 27 entrenamientos
# Tiempo M4 Max con JOBS=13: ~5h
#
# Usage:
#   nohup taskpolicy -t 0 bash scripts/launch_sens_window.sh > logs/sens_window/main.log 2>&1 &
set -euo pipefail

export LC_ALL=C
export LANG=C

JOBS="${1:-14}"
CONFIG="${2:-src/configs/intermediate_B.yaml}"
OUT="results/sens_window"
LOGS="logs/sens_window"

mkdir -p "$LOGS" "${OUT}/rows"

# Joblist con todas las combinaciones, saltando las ya hechas
JOBLIST=$(mktemp)
trap "rm -f $JOBLIST" EXIT

WINDOWS=(120 250 500)
FOLDS=(0 5 9)
SEEDS=(0 1 2)

PENDING=0
SKIPPED=0
for f in "${FOLDS[@]}"; do
    f_padded=$(printf "%02d" "$f")
    for s in "${SEEDS[@]}"; do
        for W in "${WINDOWS[@]}"; do
            marker="${OUT}/rows/DJIA_f${f_padded}_V4_s${s}_l1_2.00_l2_2.00_W${W}.parquet"
            if [ -f "$marker" ]; then
                SKIPPED=$((SKIPPED+1))
                continue
            fi
            echo "DJIA $f V4 $s $W" >> "$JOBLIST"
            PENDING=$((PENDING+1))
        done
    done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sensitivity window experiment"
echo "Pending: $PENDING entrenamientos"
echo "Skipped: $SKIPPED (ya completados)"
echo "JOBS: $JOBS"
echo "Output: $OUT"

if [ "$PENDING" -eq 0 ]; then
    echo "Nothing to run."
    exit 0
fi

parallel --jobs "$JOBS" --bar --joblog "${LOGS}/parallel.log" --colsep ' ' \
    "python scripts/run_one_sens.py \
        --market {1} --fold {2} --variant {3} --seed {4} \
        --lambda-dd 2.0 --lambda-evt 2.0 \
        --evt-window {5} \
        --total-timesteps 100000 \
        --config $CONFIG --out $OUT \
        > ${LOGS}/{1}_f{2}_{3}_s{4}_W{5}.log 2>&1" \
    :::: "$JOBLIST"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
echo
echo "Analizar con:"
echo "  python scripts/analyze_sens_window.py --rows-dir $OUT/rows --out-dir $OUT/analysis"
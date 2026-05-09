#!/usr/bin/env bash
# rerun_v2_only.sh
#
# Re-ejecuta ÚNICAMENTE la variante V2 del experimento full_optimal.
#
# Flujo:
#   1. Borra los parquets de V2 existentes (los de la config incorrecta).
#   2. Lanza GNU parallel solo para V2, con la config corregida.
#   3. Como run_one.py es idempotente, V0/V1/V3/V4 no se tocan.
#
# Uso:
#   bash scripts/rerun_v2_only.sh [JOBS]
#   bash scripts/rerun_v2_only.sh 12    # 12 procesos en paralelo
#
set -euo pipefail

JOBS="${1:-9}"
CONFIG="src/configs/full_optimal.yaml"
OUT_DIR="results/full_optimal"
LOG_DIR="logs/rerun_v2"

echo "============================================================"
echo "  Re-ejecución de V2 con VARIANT_PRESETS corregidos"
echo "  JOBS=$JOBS | CONFIG=$CONFIG | OUT=$OUT_DIR"
echo "============================================================"

# ---------------------------------------------------------------
# Paso 1: Verificar que VARIANT_PRESETS tiene V2 con lambda_dd=2.0
# ---------------------------------------------------------------
echo ""
echo "[1/4] Verificando VARIANT_PRESETS para V2..."
V2_LINE=$(grep -A 1 '"V2"' src/evt_ppo/experiment.py | head -2)
echo "  $V2_LINE"

if echo "$V2_LINE" | grep -q "lambda_dd.*0\.0"; then
    echo "  ❌ ERROR: V2 todavía tiene lambda_dd=0.0. Corrige experiment.py primero."
    exit 1
fi
if echo "$V2_LINE" | grep -q "lambda_dd.*2\.0"; then
    echo "  ✅ V2 tiene lambda_dd=2.0 (correcto)"
else
    echo "  ⚠️  No puedo confirmar lambda_dd. Verifica manualmente."
    read -p "  ¿Continuar? (y/n) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

# ---------------------------------------------------------------
# Paso 2: Contar y borrar parquets V2 existentes
# ---------------------------------------------------------------
echo ""
echo "[2/4] Borrando parquets V2 existentes..."
V2_COUNT=$(ls "$OUT_DIR"/rows/*_V2_*.parquet 2>/dev/null | wc -l)
echo "  Parquets V2 encontrados: $V2_COUNT"

if [ "$V2_COUNT" -gt 0 ]; then
    rm "$OUT_DIR"/rows/*_V2_*.parquet
    echo "  ✅ Borrados $V2_COUNT parquets de V2"
else
    echo "  (no había parquets V2 que borrar)"
fi

# Verificar que V0/V1/V3/V4 siguen intactos
for v in V0 V1 V3 V4; do
    n=$(ls "$OUT_DIR"/rows/*_${v}_*.parquet 2>/dev/null | wc -l)
    echo "  $v: $n parquets (intactos)"
done

# ---------------------------------------------------------------
# Paso 3: Lanzar entrenamiento SOLO para V2
# ---------------------------------------------------------------
echo ""
echo "[3/4] Lanzando entrenamiento de V2..."
mkdir -p "$LOG_DIR"

parallel --jobs "$JOBS" --bar --joblog "$LOG_DIR/parallel.log" \
    "python scripts/run_one.py \
        --config $CONFIG \
        --market {1} --fold {2} --variant V2 --seed {3} \
        --out $OUT_DIR \
        > $LOG_DIR/{1}_f{2}_V2_s{3}.log 2>&1" \
    ::: DJIA SP50 IBEX \
    ::: $(seq 0 12) \
    ::: 0 1 2 3 4

# ---------------------------------------------------------------
# Paso 4: Verificación final
# ---------------------------------------------------------------
echo ""
echo "[4/4] Verificación final..."
V2_NEW=$(ls "$OUT_DIR"/rows/*_V2_*.parquet 2>/dev/null | wc -l)
TOTAL=$(ls "$OUT_DIR"/rows/*.parquet 2>/dev/null | wc -l)
echo "  Parquets V2 nuevos: $V2_NEW (esperado: 195)"
echo "  Parquets totales:   $TOTAL (esperado: 975)"

if [ "$V2_NEW" -eq 195 ] && [ "$TOTAL" -eq 975 ]; then
    echo "  ✅ Re-ejecución completada con éxito"
else
    echo "  ⚠️  Cifras inesperadas. Revisa los logs en $LOG_DIR/"
    # Mostrar jobs que fallaron
    if [ -f "$LOG_DIR/parallel.log" ]; then
        FAILED=$(awk '$7 != 0' "$LOG_DIR/parallel.log" | wc -l)
        echo "  Jobs fallidos según joblog: $FAILED"
    fi
fi

echo ""
echo "============================================================"
echo "  Siguiente paso: re-ejecutar el análisis"
echo "  python scripts/04_analyze_results.py --results-dir $OUT_DIR"
echo "============================================================"
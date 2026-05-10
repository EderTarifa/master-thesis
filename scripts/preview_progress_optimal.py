"""Vista preliminar del progreso del experimento full_optimal."""
import pandas as pd
import glob
from pathlib import Path
from datetime import datetime, timedelta

RUN_NAME = "full_optimal"
ROWS_DIR = f"results/{RUN_NAME}/rows"

# 3 mercados × 13 folds × 5 variantes × 5 seeds = 975
TOTAL_TASKS = 13 * 5 * 5 # 3 * 13 * 5 * 5

files = sorted(glob.glob(f"{ROWS_DIR}/FX_*.parquet"))
print(f"{'='*60}")
print(f"Progreso: {len(files)} / {TOTAL_TASKS} tareas ({100*len(files)/TOTAL_TASKS:.1f}%)")
print(f"{'='*60}")

if not files:
    raise SystemExit(0)

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# --- Progreso por mercado/variante ---
print(f"\nFolds completados por mercado/variante:")
folds_done = df.groupby(['market', 'variant'])['fold'].nunique().unstack(fill_value=0)
print(folds_done)

# --- ETA estimada ---
if 'train_time_seconds' in df.columns:
    avg_time = df['train_time_seconds'].mean()
    remaining = TOTAL_TASKS - len(files)
    eta_seconds = remaining * avg_time / 13  # dividido por jobs paralelos
    eta_finish = datetime.now() + timedelta(seconds=eta_seconds)
    print(f"\nTiempo medio por job: {avg_time/60:.1f} min")
    print(f"ETA (aprox, 13 jobs): {eta_finish.strftime('%Y-%m-%d %H:%M')}")

# --- Métricas por variante ---
print(f"\nMedia de MDD por variante:")
print(df.groupby('variant').agg(
    mdd_mean=('mdd', 'mean'),
    mdd_std=('mdd', 'std'),
    calmar_mean=('calmar', 'mean'),
    n=('mdd', 'count'),
).round(4))

# --- Últimos jobs completados ---
print(f"\nÚltimos 5 jobs completados:")
print(df.sort_values('train_time_seconds', ascending=False).head()[
    ['market', 'fold', 'variant', 'seed', 'mdd', 'calmar', 'train_time_seconds']
].to_string())

# --- Comparación pareada V4 vs V1 (EVT óptimo vs baseline) ---
if {'V1', 'V4'}.issubset(df['variant'].unique()):
    paired = df.pivot_table(
        index=['market', 'fold', 'seed'],
        columns='variant',
        values='mdd'
    ).dropna()
    
    if len(paired) > 5 and 'V1' in paired.columns and 'V4' in paired.columns:
        diff = paired['V4'] - paired['V1']
        print(f"\n{'='*60}")
        print(f"Diferencia pareada V4 - V1 (MDD):")
        print(f"  n = {len(paired)} observaciones")
        print(f"  mean(V4 - V1) = {diff.mean():+.5f} pp")
        print(f"  median(V4 - V1) = {diff.median():+.5f} pp")
        print(f"  V4 mejor que V1 (MDD menor): {(diff < 0).mean() * 100:.1f}%")
        print(f"{'='*60}")

# --- Comparación V3 vs V0 (double EVT vs no-EVT) ---
if {'V0', 'V3'}.issubset(df['variant'].unique()):
    paired_v3 = df.pivot_table(
        index=['market', 'fold', 'seed'],
        columns='variant',
        values='mdd'
    ).dropna()
    
    if len(paired_v3) > 5 and 'V0' in paired_v3.columns and 'V3' in paired_v3.columns:
        diff_v3 = paired_v3['V3'] - paired_v3['V0']
        print(f"\nDiferencia pareada V3 - V0 (MDD):")
        print(f"  n = {len(paired_v3)}")
        print(f"  mean = {diff_v3.mean():+.5f} pp")
        print(f"  V3 mejor que V0: {(diff_v3 < 0).mean() * 100:.1f}%")
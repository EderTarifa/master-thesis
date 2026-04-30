"""Vista preliminar del progreso del experimento full."""
import pandas as pd
import glob
from pathlib import Path

files = sorted(glob.glob('results/full/rows/*.parquet'))
print(f"Total parquets: {len(files)}")
if not files:
    raise SystemExit(0)

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"\nFolds completados por mercado/variante:")
print(df.groupby(['market', 'variant'])['fold'].nunique().unstack())

print(f"\nMedia de MDD por variante (con folds completos):")
# Solo considerar folds donde TODAS las variantes tengan al menos una semilla
n_seeds = df.groupby(['market', 'fold', 'variant'])['seed'].nunique().reset_index()
print(df.groupby('variant').agg(
    mdd_mean=('mdd', 'mean'),
    mdd_std=('mdd', 'std'),
    n=('mdd', 'count'),
).round(4))

print(f"\nÚltimos 5 jobs completados:")
print(df.sort_values('train_time_seconds', ascending=False).head().to_string())

# Si tienes V1 y V4 con suficientes datos, una vista pareada rápida
if {'V1', 'V4'}.issubset(df['variant'].unique()):
    paired = df.pivot_table(index=['market', 'fold', 'seed'],
                              columns='variant', values='mdd').dropna()
    if len(paired) > 5 and 'V1' in paired and 'V4' in paired:
        diff = paired['V4'] - paired['V1']
        print(f"\nDiferencia pareada V4 - V1:")
        print(f"  n = {len(paired)}")
        print(f"  mean = {diff.mean():+.5f}")
        print(f"  median = {diff.median():+.5f}")
        print(f"  pct V4 < V1 = {(diff < 0).mean() * 100:.1f}%")
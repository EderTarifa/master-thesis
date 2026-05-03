import sys
sys.path.insert(0, 'src')
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob

# Load runs and regime classification
runs = pd.concat([pd.read_parquet(f)
                   for f in sorted(glob.glob('results/full/rows/*.parquet'))],
                 ignore_index=True)
regimes = pd.read_csv('results/sidequest_evt/regime_classification.csv')

# Map test_year via fold ordering
runs['test_year'] = runs['fold'].map({i: 2013+i for i in range(13)})

# Merge
runs = runs.merge(regimes[['market','fold','regime','evt_xi','kurtosis']],
                   on=['market','fold'], how='left')

print("=== Analisis condicional por regimen ===")
for (baseline, treatment) in [('V0','V1'), ('V1','V4'), ('V1','V3'),
                                ('V3','V4')]:
    sub = runs[runs['variant'].isin([baseline, treatment])]
    pivot = sub.pivot_table(
        index=['market','fold','seed','regime','evt_xi'],
        columns='variant', values='mdd'
    ).dropna(how='any').reset_index()
    if pivot.empty or baseline not in pivot.columns:
        continue
    pivot['diff'] = pivot[treatment] - pivot[baseline]

    print(f"\n--- {baseline} vs {treatment} ---")
    by_regime = pivot.groupby('regime').agg(
        mean_base=(baseline, 'mean'),
        mean_treat=(treatment, 'mean'),
        mean_diff=('diff', 'mean'),
        median_diff=('diff', 'median'),
        pct_treat_better=('diff', lambda x: (x < 0).mean() * 100),
        n=('diff', 'count'),
    ).round(4)
    print(by_regime)

# Plot diff vs xi (only V1 vs V4)
sub = runs[runs['variant'].isin(['V1','V4'])]
pivot = sub.pivot_table(
    index=['market','fold','seed','evt_xi'],
    columns='variant', values='mdd'
).dropna(how='any').reset_index()
if not pivot.empty and 'V4' in pivot.columns:
    pivot['diff'] = pivot['V4'] - pivot['V1']
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(pivot['evt_xi'], pivot['diff'], alpha=0.5)
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    # Tendencia
    z = np.polyfit(pivot['evt_xi'].values, pivot['diff'].values, 1)
    xs = np.linspace(pivot['evt_xi'].min(), pivot['evt_xi'].max(), 50)
    ax.plot(xs, np.polyval(z, xs), 'r-', lw=2,
             label=f'slope = {z[0]:+.4f}')
    ax.set_xlabel('EVT shape (xi) of test fold')
    ax.set_ylabel('MDD difference: V4 - V1')
    ax.set_title('Effect of EVT depends on tail heaviness?')
    ax.legend()
    ax.grid(alpha=0.3)
    Path('results/sidequest_evt/figs').mkdir(parents=True, exist_ok=True)
    fig.savefig('results/sidequest_evt/figs/diff_vs_xi.png', dpi=150)
    print('Saved diff_vs_xi.png')
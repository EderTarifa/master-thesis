"""
Análisis de sensibilidad a la ventana EVT (W).

Lee parquets en results/sens_window/rows/ y produce:
  - tables/sens_window_summary.csv: agregado por (W, fold) con métricas
  - figs/mdd_by_window.png: MDD vs W por fold
  - figs/calmar_by_window.png: Calmar vs W por fold
  - tables/best_window_per_fold.csv: la W óptima por fold

Uso:
    python scripts/analyze_sens_window.py \
        --rows-dir results/sens_window/rows \
        --out-dir results/sens_window/analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows-dir", default="results/sens_window/rows")
    p.add_argument("--out-dir", default="results/sens_window/analysis")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.rows_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquets in {args.rows_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} runs from {len(files)} parquets")

    # Asegurar tipo
    df["evt_window"] = df["evt_window"].astype(int)

    # Agregado por (fold, W)
    agg = df.groupby(["fold", "evt_window"]).agg(
        mdd_mean=("mdd", "mean"),
        mdd_std=("mdd", "std"),
        cagr_mean=("cagr", "mean"),
        sharpe_mean=("sharpe", "mean"),
        calmar_mean=("calmar", "mean"),
        n_seeds=("seed", "nunique"),
    ).round(4).reset_index()
    agg.to_csv(out_dir / "tables" / "sens_window_summary.csv", index=False)
    print("\n=== Summary by (fold, W) ===")
    print(agg.to_string(index=False))

    # Mejor W por fold (mínimo MDD)
    best_w = agg.loc[agg.groupby("fold")["mdd_mean"].idxmin()]
    best_w.to_csv(out_dir / "tables" / "best_window_per_fold.csv", index=False)
    print("\n=== Best W per fold (min MDD) ===")
    print(best_w.to_string(index=False))

    # Plot MDD vs W
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    folds = sorted(agg["fold"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(folds)))

    for ax_, metric, ylabel, lower_better in [
        (axes[0], "mdd_mean", "Mean MDD", True),
        (axes[1], "calmar_mean", "Mean Calmar", False),
    ]:
        for f, c in zip(folds, colors):
            sub = agg[agg["fold"] == f].sort_values("evt_window")
            err = df[df["fold"] == f].groupby("evt_window")["mdd"].std().reindex(
                sub["evt_window"]).values
            ax_.errorbar(
                sub["evt_window"], sub[metric],
                yerr=err if metric == "mdd_mean" else None,
                marker="o", lw=2, ms=8,
                label=f"Fold {f}", color=c, capsize=3,
            )
        ax_.set_xlabel(r"EVT window $W$ (days)")
        ax_.set_ylabel(ylabel)
        ax_.set_title(f"{ylabel} by W and fold "
                       f"({'lower=better' if lower_better else 'higher=better'})")
        ax_.set_xticks([120, 250, 500])
        ax_.legend()
        ax_.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "sens_window_combined.png", dpi=150)
    plt.close(fig)

    # Heatmap rápido
    pivot = agg.pivot_table(index="fold", columns="evt_window", values="mdd_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Fold {f}" for f in pivot.index])
    ax.set_xlabel("EVT window W")
    ax.set_title("Mean MDD across (fold, W)\nlower is better")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                     color="white" if v > pivot.values.mean() else "black",
                     fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean MDD")
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "heatmap_mdd_window.png", dpi=150)
    plt.close(fig)

    print(f"\nOutputs in {out_dir}/")


if __name__ == "__main__":
    main()
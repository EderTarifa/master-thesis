"""
Análisis de los resultados del sensitivity de lambda.

Lee los parquets en results/sens_lambda/rows/ producidos por
run_one_sens.py, y produce:

  - tables/sens_lambda_summary.csv: tabla agregada media+std por (variant, l1, l2)
  - tables/sens_lambda_pivot_mdd.csv: tabla pivot para V4 con l1 vs l2
  - figs/heatmap_mdd_V4.png: heatmap MDD por (l1, l2) en V4
  - figs/heatmap_calmar_V4.png: heatmap Calmar por (l1, l2) en V4
  - figs/v1_vs_v4_by_lambda.png: comparativa V1 vs V4 a igualdad de l1
  - tables/best_config.csv: la mejor combinación según múltiples criterios

Uso:
    python scripts/analyze_sens_lambda.py \
        --rows-dir results/sens_lambda/rows \
        --out-dir results/sens_lambda/analysis
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


def load_data(rows_dir: Path) -> pd.DataFrame:
    files = sorted(rows_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquets in {rows_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} runs from {len(files)} parquets")

    # Asegurar columnas lambdas como floats (a veces vienen como object)
    df["lambda_dd"] = df["lambda_dd"].astype(float)
    df["lambda_evt"] = df["lambda_evt"].astype(float)

    return df


def summary_table(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Agregado por (variant, lambda_dd, lambda_evt)."""
    agg = df.groupby(["variant", "lambda_dd", "lambda_evt"]).agg(
        mdd_mean=("mdd", "mean"),
        mdd_std=("mdd", "std"),
        mdd_min=("mdd", "min"),
        mdd_max=("mdd", "max"),
        cagr_mean=("cagr", "mean"),
        sharpe_mean=("sharpe", "mean"),
        calmar_mean=("calmar", "mean"),
        sortino_mean=("sortino", "mean"),
        cdar_95_mean=("cdar_95", "mean"),
        n_seeds=("seed", "nunique"),
    ).round(4).reset_index()

    out_path = out_dir / "tables" / "sens_lambda_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    print(f"\n=== Summary table ===")
    print(agg.to_string())
    return agg


def heatmaps_v4(df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmaps de MDD y Calmar para V4."""
    v4 = df[df["variant"] == "V4"]
    if v4.empty:
        print("No V4 runs found, skipping heatmaps")
        return

    # Pivot para heatmap. Filas = lambda_dd, columnas = lambda_evt
    pivot_mdd = v4.pivot_table(
        index="lambda_dd", columns="lambda_evt",
        values="mdd", aggfunc="mean",
    )
    pivot_calmar = v4.pivot_table(
        index="lambda_dd", columns="lambda_evt",
        values="calmar", aggfunc="mean",
    )
    pivot_sharpe = v4.pivot_table(
        index="lambda_dd", columns="lambda_evt",
        values="sharpe", aggfunc="mean",
    )

    pivot_mdd.to_csv(out_dir / "tables" / "sens_lambda_pivot_mdd.csv")
    pivot_calmar.to_csv(out_dir / "tables" / "sens_lambda_pivot_calmar.csv")

    # MDD heatmap (menor es mejor)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_mdd.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pivot_mdd.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pivot_mdd.columns])
    ax.set_yticks(range(len(pivot_mdd.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in pivot_mdd.index])
    ax.set_xlabel(r"$\lambda_{evt}$ (peso CVaR-EVT)")
    ax.set_ylabel(r"$\lambda_{dd}$ (peso drawdown)")
    ax.set_title("V4: Mean MDD across (lambda_dd, lambda_evt)\n"
                  "lower is better")
    plt.colorbar(im, ax=ax, label="Mean MDD")
    # Anotar valores
    for i in range(len(pivot_mdd.index)):
        for j in range(len(pivot_mdd.columns)):
            v = pivot_mdd.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                     color="white" if v > pivot_mdd.values.mean() else "black",
                     fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "heatmap_mdd_V4.png", dpi=150)
    plt.close(fig)

    # Calmar heatmap (mayor es mejor)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_calmar.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot_calmar.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pivot_calmar.columns])
    ax.set_yticks(range(len(pivot_calmar.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in pivot_calmar.index])
    ax.set_xlabel(r"$\lambda_{evt}$ (peso CVaR-EVT)")
    ax.set_ylabel(r"$\lambda_{dd}$ (peso drawdown)")
    ax.set_title("V4: Mean Calmar across (lambda_dd, lambda_evt)\n"
                  "higher is better")
    plt.colorbar(im, ax=ax, label="Mean Calmar")
    for i in range(len(pivot_calmar.index)):
        for j in range(len(pivot_calmar.columns)):
            v = pivot_calmar.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v < pivot_calmar.values.mean() else "black",
                     fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "heatmap_calmar_V4.png", dpi=150)
    plt.close(fig)

    # Sharpe heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_sharpe.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot_sharpe.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pivot_sharpe.columns])
    ax.set_yticks(range(len(pivot_sharpe.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in pivot_sharpe.index])
    ax.set_xlabel(r"$\lambda_{evt}$")
    ax.set_ylabel(r"$\lambda_{dd}$")
    ax.set_title("V4: Mean Sharpe ratio")
    plt.colorbar(im, ax=ax, label="Mean Sharpe")
    for i in range(len(pivot_sharpe.index)):
        for j in range(len(pivot_sharpe.columns)):
            v = pivot_sharpe.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v < pivot_sharpe.values.mean() else "black",
                     fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "heatmap_sharpe_V4.png", dpi=150)
    plt.close(fig)

    print(f"Saved heatmaps to {out_dir / 'figs'}")


def v1_vs_v4_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Comparar V1 (sin EVT) vs V4 (con EVT) a igualdad de lambda_dd."""
    v1 = df[df["variant"] == "V1"]
    v4 = df[df["variant"] == "V4"]
    if v1.empty or v4.empty:
        return

    v1_agg = v1.groupby("lambda_dd")["mdd"].mean().reset_index()
    v1_agg.columns = ["lambda_dd", "mdd_V1"]

    # V4: para cada lambda_dd, tomar el mejor lambda_evt
    v4_best = v4.loc[v4.groupby("lambda_dd")["mdd"].idxmin()]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(v1_agg["lambda_dd"], v1_agg["mdd_V1"],
             "o-", lw=2, ms=10, label="V1 (no EVT)", color="C0")
    ax.plot(v4_best["lambda_dd"], v4_best["mdd"],
             "s-", lw=2, ms=10, label="V4 (best lambda_evt)", color="C3")
    # Anotar el lambda_evt óptimo en V4
    for _, r in v4_best.iterrows():
        ax.annotate(f"$\\lambda_{{evt}}={r['lambda_evt']:.1f}$",
                     xy=(r["lambda_dd"], r["mdd"]),
                     xytext=(0, -15), textcoords="offset points",
                     ha="center", fontsize=9)
    ax.set_xlabel(r"$\lambda_{dd}$")
    ax.set_ylabel("Mean MDD")
    ax.set_title("V1 vs V4 (best $\\lambda_{evt}$) by drawdown weight")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "v1_vs_v4_by_lambda.png", dpi=150)
    plt.close(fig)
    print(f"Saved V1 vs V4 comparison")


def find_best_configs(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Encontrar las mejores configuraciones según múltiples criterios."""
    agg = df.groupby(["variant", "lambda_dd", "lambda_evt"]).agg(
        mdd_mean=("mdd", "mean"),
        cagr_mean=("cagr", "mean"),
        sharpe_mean=("sharpe", "mean"),
        calmar_mean=("calmar", "mean"),
    ).reset_index()

    rows = []
    # Mejor MDD (objetivo del TFM)
    best_mdd = agg.loc[agg["mdd_mean"].idxmin()]
    rows.append({"criterion": "min_MDD", **best_mdd.to_dict()})
    # Mejor Calmar (CAGR/MDD)
    best_calmar = agg.loc[agg["calmar_mean"].idxmax()]
    rows.append({"criterion": "max_Calmar", **best_calmar.to_dict()})
    # Mejor Sharpe
    best_sharpe = agg.loc[agg["sharpe_mean"].idxmax()]
    rows.append({"criterion": "max_Sharpe", **best_sharpe.to_dict()})
    # Mejor MDD restringido a CAGR > 0
    pos_cagr = agg[agg["cagr_mean"] > 0]
    if not pos_cagr.empty:
        best_constrained = pos_cagr.loc[pos_cagr["mdd_mean"].idxmin()]
        rows.append({"criterion": "min_MDD_with_positive_CAGR",
                     **best_constrained.to_dict()})

    df_best = pd.DataFrame(rows)
    df_best.to_csv(out_dir / "tables" / "best_config.csv", index=False)
    print(f"\n=== Best configurations ===")
    print(df_best.to_string(index=False))
    return df_best


def stats_per_lambda(df: pd.DataFrame, out_dir: Path) -> None:
    """Test estadístico V4 vs V1 a igualdad de lambda_dd."""
    print("\n=== V4 vs V1 paired comparison by lambda_dd ===")
    rows = []
    for l1 in sorted(df["lambda_dd"].unique()):
        v1 = df[(df["variant"] == "V1") & (df["lambda_dd"] == l1)]
        v4 = df[(df["variant"] == "V4") & (df["lambda_dd"] == l1)]
        if v1.empty or v4.empty:
            continue
        # Para V4 con varios lambda_evt, agregar por seed con cada lambda_evt
        # y tomar la mejor por seed (podría ser otro criterio).
        # Aquí compararemos V1(l1) contra V4(l1, best l2 average).
        v4_avg = v4.groupby("lambda_evt")["mdd"].mean()
        best_l2 = v4_avg.idxmin()
        v4_best = v4[v4["lambda_evt"] == best_l2]
        # Pareados por seed
        paired = pd.merge(
            v1[["seed", "mdd"]].rename(columns={"mdd": "V1"}),
            v4_best[["seed", "mdd"]].rename(columns={"mdd": "V4"}),
            on="seed",
        )
        if len(paired) < 3:
            continue
        from scipy import stats as sp_stats
        t = sp_stats.ttest_rel(paired["V4"], paired["V1"], alternative="less")
        rows.append({
            "lambda_dd": l1,
            "best_lambda_evt": best_l2,
            "n_pairs": len(paired),
            "mean_V1": paired["V1"].mean(),
            "mean_V4": paired["V4"].mean(),
            "mean_diff": (paired["V4"] - paired["V1"]).mean(),
            "t_pvalue": t.pvalue,
        })

    df_stats = pd.DataFrame(rows).round(4)
    df_stats.to_csv(out_dir / "tables" / "v4_vs_v1_by_lambda.csv", index=False)
    print(df_stats.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows-dir", default="results/sens_lambda/rows")
    p.add_argument("--out-dir", default="results/sens_lambda/analysis")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.rows_dir))
    summary_table(df, out_dir)
    heatmaps_v4(df, out_dir)
    v1_vs_v4_comparison(df, out_dir)
    find_best_configs(df, out_dir)
    stats_per_lambda(df, out_dir)

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
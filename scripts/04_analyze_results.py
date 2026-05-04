"""
Analyse a finished experiment and produce all tables, plots, and
hypothesis tests required by the TFM.

Usage:
    python scripts/04_analyze_results.py --results-dir results/full/

Outputs (under <results-dir>/analysis/):
    - tables/agg_by_variant.csv
    - tables/agg_by_market_variant.csv
    - tables/paired_V1_V4.csv
    - tables/hypothesis_tests.csv         <- the headline table for the TFM
    - tables/benchmarks_summary.csv
    - plots/mdd_boxplot_global.{png,svg}
    - plots/mdd_boxplot_by_market_<market>.{png,svg}
    - plots/paired_diff_V1_V4.{png,svg}
    - plots/equity_<market>_fold<fold>.{png,svg}     # one per fold
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from evt_ppo import plots as P
from evt_ppo.experiment import aggregate_runs, paired_table
from evt_ppo.statistics import full_comparison


def make_aggregated_tables(runs: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # By variant globally.
    agg_v = aggregate_runs(runs, group_by=("variant",))
    agg_v.to_csv(out_dir / "agg_by_variant.csv")

    # By market x variant.
    agg_mv = aggregate_runs(runs, group_by=("market", "variant"))
    agg_mv.to_csv(out_dir / "agg_by_market_variant.csv")


def hypothesis_test_table(runs: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Run hypothesis tests for V1 vs V2/V3/V4 and V0 vs V1, store CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    contrasts = [
        ("V0", "V1"), ("V1", "V2"), ("V1", "V3"), ("V1", "V4"),
        ("V2", "V4"), ("V3", "V4"),
    ]
    for baseline, treatment in contrasts:
        if not {baseline, treatment}.issubset(runs["variant"].unique()):
            continue
        paired = paired_table(runs, baseline=baseline, treatment=treatment)
        if len(paired) < 5:
            continue
        result = full_comparison(
            paired[baseline].values, paired[treatment].values,
            alpha=0.05, n_bootstrap=5000, seed=0,
        )
        rows.append({
            "baseline": baseline,
            "treatment": treatment,
            **result,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hypothesis_tests.csv", index=False)
    return df


def benchmarks_summary(bench: pd.DataFrame, out_dir: Path) -> None:
    if bench is None or bench.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [c for c in
                   ("mdd", "calmar", "sharpe", "cagr", "vol_annualised", "cdar_95")
                   if c in bench.columns]
    summary = bench.groupby(["market", "benchmark"])[metric_cols].agg(["mean", "std"])
    summary.to_csv(out_dir / "benchmarks_summary.csv")


def make_global_plots(runs: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    variants_sorted = sorted(runs["variant"].unique())

    P.plot_mdd_boxplots(
        runs, variant_col="variant", mdd_col="mdd",
        title="Maximum Drawdown distribution by variant (all markets, all folds, all seeds)",
        save_path=out_dir / "mdd_boxplot_global",
        order=variants_sorted,
    )

    for market in runs["market"].unique():
        sub = runs[runs["market"] == market]
        P.plot_mdd_boxplots(
            sub, variant_col="variant", mdd_col="mdd",
            title=f"MDD by variant — {market}",
            save_path=out_dir / f"mdd_boxplot_by_market_{market}",
            order=variants_sorted,
        )

    if {"V1", "V4"}.issubset(set(runs["variant"].unique())):
        paired = paired_table(runs, "V1", "V4")
        if len(paired) > 0:
            P.plot_paired_diffs(
                paired["diff"].values,
                name_left="V1", name_right="V4",
                title="Paired MDD difference: V4 (EVT) - V1 (no EVT)",
                save_path=out_dir / "paired_diff_V1_V4",
            )


def make_per_fold_equity_plots(results_dir: Path, runs: pd.DataFrame,
                                out_dir: Path) -> None:
    """For each (market, fold), overlay equity curves of V0..V4 (seed 0)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold_dir = results_dir / "per_fold"
    if not per_fold_dir.exists():
        return

    for market in runs["market"].unique():
        market_dir = per_fold_dir / market
        if not market_dir.exists():
            continue
        for fold_dir in sorted(market_dir.glob("fold*")):
            series = {}
            for v in sorted(runs["variant"].unique()):
                f = fold_dir / f"{v}_seed0.npz"
                if f.exists():
                    arr = np.load(f)
                    series[v] = arr["values"]
            if series:
                P.plot_value_and_drawdown(
                    series,
                    title=f"{market} — {fold_dir.name} — equity & drawdown",
                    save_path=out_dir / f"equity_{market}_{fold_dir.name}",
                )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    args = p.parse_args()

    rd = args.results_dir
    if not (rd / "runs.parquet").exists():
        raise SystemExit(f"No runs.parquet in {rd}")
    runs = pd.read_parquet(rd / "runs.parquet")
    bench_path = rd / "benchmarks.parquet"
    bench = pd.read_parquet(bench_path) if bench_path.exists() else None

    analysis_dir = rd / "analysis"
    tables_dir = analysis_dir / "tables"
    plots_dir = analysis_dir / "plots"

    print(f"Loaded {len(runs)} runs from {rd}")
    make_aggregated_tables(runs, tables_dir)
    df_tests = hypothesis_test_table(runs, tables_dir)
    benchmarks_summary(bench, tables_dir)
    make_global_plots(runs, plots_dir)
    make_per_fold_equity_plots(rd, runs, plots_dir)

    if not df_tests.empty:
        print("\n=== Hypothesis tests ===")
        print(df_tests.to_string(index=False))

    print(f"\nAll outputs saved under: {analysis_dir}")


if __name__ == "__main__":
    main()

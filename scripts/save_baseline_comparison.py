"""
V4 vs baseline competitors (POTPG, EX-DRL): paired comparison with correct
filtering, saved to CSV.

Filters V4 to the intersection with the baseline runs:
  - markets in {DJIA, SP50, IBEX} (the three equity universes the baselines
    were trained on)
  - seeds in {0,1,2,3,4} (the seeds the baselines were trained with)
This yields exactly 3 x 13 x 5 = 195 paired (market, fold, seed) episodes,
reproducing the V4 numbers reported in the paper (MDD 0.1158, Calmar 1.153).

Outputs:
  - v4_vs_baselines_metrics.csv  : mean metrics per method (V4, POTPG, EXDRL)
  - v4_vs_baselines_paired.csv   : paired-contrast statistics on MDD
  - v4_vs_baselines_by_market.csv: per-market breakdown of the paired diff

Usage
-----
    python scripts/save_baseline_comparison.py \
        --v4-results results/full_optimal/runs.parquet \
        --baseline-results results/baselines/runs.parquet \
        --out-dir results/baselines/comparison
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


EQUITY_MARKETS = ["DJIA", "SP50", "IBEX"]
BASELINE_SEEDS = [0, 1, 2, 3, 4]
METRIC_COLS = ["mdd", "cdar_95", "sharpe", "calmar", "cagr", "turnover_mean"]


def block_bootstrap_ci(values: np.ndarray, block_size: int,
                       n_resamples: int = 5000, alpha: float = 0.05,
                       seed: int = 42) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean of a paired-difference vector."""
    rng = np.random.default_rng(seed)
    n = len(values)
    n_blocks = int(np.ceil(n / block_size))
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([values[s:s + block_size] for s in starts])[:n]
        means[i] = sample.mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_contrast(v4: pd.DataFrame, bs: pd.DataFrame, variant: str,
                    metric: str = "mdd") -> dict:
    """Paired V4-vs-baseline contrast on `metric`. Diff = V4 - baseline."""
    b = bs[bs.variant == variant][["market", "fold", "seed", metric]].rename(
        columns={metric: f"{metric}_b"})
    merged = v4[["market", "fold", "seed", metric]].rename(
        columns={metric: f"{metric}_v4"}).merge(b, on=["market", "fold", "seed"])
    if merged.empty:
        return {"baseline": variant, "n_pairs": 0}

    d = (merged[f"{metric}_v4"] - merged[f"{metric}_b"]).values
    t_stat, t_p_two = stats.ttest_rel(merged[f"{metric}_v4"], merged[f"{metric}_b"])
    t_p_one = t_p_two / 2 if t_stat < 0 else 1 - t_p_two / 2
    try:
        _, w_p_one = stats.wilcoxon(d, alternative="less")
    except ValueError:
        w_p_one = float("nan")

    lo1, hi1 = block_bootstrap_ci(d, 1)
    lo5, hi5 = block_bootstrap_ci(d, 5)
    lo20, hi20 = block_bootstrap_ci(d, 20)

    return {
        "baseline": variant,
        "n_pairs": int(len(d)),
        "mean_d_pp": float(d.mean() * 100),
        "median_d_pp": float(np.median(d) * 100),
        "std_d_pp": float(d.std(ddof=1) * 100),
        "pct_V4_better": float((d < 0).mean() * 100),
        "p_t_one_sided": float(t_p_one),
        "p_wilcoxon_one_sided": float(w_p_one),
        "ci95_b1_lo_pp": lo1 * 100, "ci95_b1_hi_pp": hi1 * 100,
        "ci95_b5_lo_pp": lo5 * 100, "ci95_b5_hi_pp": hi5 * 100,
        "ci95_b20_lo_pp": lo20 * 100, "ci95_b20_hi_pp": hi20 * 100,
    }


def per_market_breakdown(v4: pd.DataFrame, bs: pd.DataFrame,
                         variant: str, metric: str = "mdd") -> pd.DataFrame:
    """Per-market mean paired diff and % V4 better."""
    b = bs[bs.variant == variant][["market", "fold", "seed", metric]].rename(
        columns={metric: f"{metric}_b"})
    merged = v4[["market", "fold", "seed", metric]].rename(
        columns={metric: f"{metric}_v4"}).merge(b, on=["market", "fold", "seed"])
    merged["diff_pp"] = (merged[f"{metric}_v4"] - merged[f"{metric}_b"]) * 100
    out = merged.groupby("market").agg(
        n_pairs=("diff_pp", "size"),
        mean_diff_pp=("diff_pp", "mean"),
        pct_V4_better=("diff_pp", lambda x: (x < 0).mean() * 100),
    ).round(3).reset_index()
    out["baseline"] = variant
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-results", default="results/full_optimal/runs.parquet")
    ap.add_argument("--baseline-results", default="results/baselines/runs.parquet")
    ap.add_argument("--out-dir", default="results/baselines")
    ap.add_argument("--baselines", nargs="+", default=["POTPG", "EXDRL"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v4_all = pd.read_parquet(args.v4_results)
    v4 = v4_all[
        (v4_all.variant == "V4")
        & (v4_all.market.isin(EQUITY_MARKETS))
        & (v4_all.seed.isin(BASELINE_SEEDS))
    ].copy()
    bs = pd.read_parquet(args.baseline_results)

    print(f"V4 filtered: {len(v4)} runs (expected 195)")
    print(f"Baselines: {len(bs)} runs\n")

    # --- 1. Mean metrics per method ---
    metric_rows = []
    row_v4 = {"method": "V4"}
    row_v4.update(v4[METRIC_COLS].mean().round(4).to_dict())
    metric_rows.append(row_v4)
    for variant in args.baselines:
        r = {"method": variant}
        r.update(bs[bs.variant == variant][METRIC_COLS].mean().round(4).to_dict())
        metric_rows.append(r)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "v4_vs_baselines_metrics.csv", index=False)
    print("=== Mean metrics per method ===")
    print(metrics_df.to_string(index=False))
    print(f"\n-> {out_dir/'v4_vs_baselines_metrics.csv'}\n")

    # --- 2. Paired contrasts on MDD ---
    contrasts = [paired_contrast(v4, bs, v, "mdd") for v in args.baselines]
    contrasts_df = pd.DataFrame(contrasts)
    contrasts_df.to_csv(out_dir / "v4_vs_baselines_paired.csv", index=False)
    print("=== Paired contrasts on MDD (V4 - baseline) ===")
    print(contrasts_df.to_string(index=False))
    print(f"\n-> {out_dir/'v4_vs_baselines_paired.csv'}\n")

    # --- 3. Per-market breakdown ---
    breakdowns = pd.concat(
        [per_market_breakdown(v4, bs, v, "mdd") for v in args.baselines],
        ignore_index=True,
    )
    breakdowns.to_csv(out_dir / "v4_vs_baselines_by_market.csv", index=False)
    print("=== Per-market breakdown of paired diff (MDD) ===")
    print(breakdowns.to_string(index=False))
    print(f"\n-> {out_dir/'v4_vs_baselines_by_market.csv'}")


if __name__ == "__main__":
    main()
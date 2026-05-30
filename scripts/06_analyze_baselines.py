"""
Pair-wise comparison V4 PPO vs baseline competitors (POTPG, EXDRL).

Builds a paired table over (market, fold, seed) and reports:
- Per-baseline aggregate stats vs V4 (mean MDD, CDaR, Calmar)
- One-sided paired t-test (treatment < V4) and Wilcoxon signed-rank test
- Block-bootstrap 95% CIs with block sizes 1, 5, 20
- Forest plot with all baselines side-by-side

Outputs:
- results/baselines_optimal/v4_vs_baselines.csv  (summary table)
- results/baselines_optimal/forest_v4_vs_baselines.svg

The naming convention is: lower = better (MDD), so V4 "wins" if its
mean MDD is below the baseline's, and the paired diff (V4 - baseline)
is negative.

Usage
-----
    python scripts/06_analyze_baselines.py \\
        --v4-results results/full_optimal/runs.parquet \\
        --baseline-results results/baselines_optimal/runs.parquet \\
        --out-dir results/baselines_optimal
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def block_bootstrap_ci(values: np.ndarray, block_size: int,
                       n_resamples: int = 5000, alpha: float = 0.05,
                       seed: int = 42) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    n = len(values)
    n_blocks = int(np.ceil(n / block_size))
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        blocks = [values[s:s + block_size] for s in starts]
        sample = np.concatenate(blocks)[:n]
        means[i] = sample.mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_contrast(v4_df: pd.DataFrame, baseline_df: pd.DataFrame,
                    baseline_name: str, metric: str = "mdd") -> dict:
    """
    Build paired (market, fold, seed) table and run tests.
    Treatment = V4. Comparator = baseline. Diff = V4 - baseline.
    If diff < 0 then V4 wins.
    """
    v4 = v4_df[v4_df["variant"] == "V4"][["market", "fold", "seed", metric]].copy()
    v4 = v4.rename(columns={metric: f"{metric}_V4"})
    bs = baseline_df[baseline_df["variant"] == baseline_name][
        ["market", "fold", "seed", metric]
    ].copy()
    bs = bs.rename(columns={metric: f"{metric}_{baseline_name}"})
    merged = v4.merge(bs, on=["market", "fold", "seed"], how="inner")
    if merged.empty:
        return {"baseline": baseline_name, "n_pairs": 0}

    d = (merged[f"{metric}_V4"] - merged[f"{metric}_{baseline_name}"]).values
    # One-sided test: H1: V4 < baseline => d < 0 => t / W left-tailed
    t_stat, t_p_two = stats.ttest_rel(merged[f"{metric}_V4"],
                                      merged[f"{metric}_{baseline_name}"])
    t_p_one = t_p_two / 2 if t_stat < 0 else 1 - t_p_two / 2
    try:
        w_stat, w_p_one = stats.wilcoxon(d, alternative="less")
    except ValueError:
        w_p_one = float("nan")

    lo1, hi1 = block_bootstrap_ci(d, block_size=1)
    lo5, hi5 = block_bootstrap_ci(d, block_size=5)
    lo20, hi20 = block_bootstrap_ci(d, block_size=20)

    return {
        "baseline": baseline_name,
        "n_pairs": len(d),
        "mean_d_pp": float(d.mean() * 100),
        "median_d_pp": float(np.median(d) * 100),
        "pct_V4_better": float((d < 0).mean() * 100),
        "p_t_one_sided": float(t_p_one),
        "p_wilcoxon_one_sided": float(w_p_one),
        "ci95_b1_lo_pp": lo1 * 100, "ci95_b1_hi_pp": hi1 * 100,
        "ci95_b5_lo_pp": lo5 * 100, "ci95_b5_hi_pp": hi5 * 100,
        "ci95_b20_lo_pp": lo20 * 100, "ci95_b20_hi_pp": hi20 * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-results", default="results/full_optimal/runs.parquet")
    ap.add_argument("--baseline-results",
                    default="results/baselines_optimal/runs.parquet")
    ap.add_argument("--out-dir", default="results/baselines_optimal")
    ap.add_argument("--baselines", nargs="+", default=["POTPG", "EXDRL"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v4_df = pd.read_parquet(args.v4_results)
    bs_df = pd.read_parquet(args.baseline_results)

    print(f"V4 results: {len(v4_df[v4_df['variant'] == 'V4'])} runs across "
          f"{v4_df['market'].nunique()} markets")
    print(f"Baseline results: {len(bs_df)} runs, "
          f"variants={sorted(bs_df['variant'].unique())}")

    summaries = []
    for b in args.baselines:
        s = paired_contrast(v4_df, bs_df, b, metric="mdd")
        summaries.append(s)
        print(f"\n=== V4 vs {b} (n={s.get('n_pairs', 0)} paired episodes) ===")
        if s["n_pairs"] == 0:
            print("  no paired observations -> SKIP")
            continue
        print(f"  mean Δ (pp): {s['mean_d_pp']:+.2f}")
        print(f"  median Δ (pp): {s['median_d_pp']:+.2f}")
        print(f"  % V4 better: {s['pct_V4_better']:.1f}%")
        print(f"  paired t-test (V4 < baseline): p = {s['p_t_one_sided']:.3e}")
        print(f"  Wilcoxon       (V4 < baseline): p = {s['p_wilcoxon_one_sided']:.3e}")
        print(f"  CI95 (block=1):  [{s['ci95_b1_lo_pp']:+.2f}, {s['ci95_b1_hi_pp']:+.2f}] pp")
        print(f"  CI95 (block=5):  [{s['ci95_b5_lo_pp']:+.2f}, {s['ci95_b5_hi_pp']:+.2f}] pp")
        print(f"  CI95 (block=20): [{s['ci95_b20_lo_pp']:+.2f}, {s['ci95_b20_hi_pp']:+.2f}] pp")

    df_summary = pd.DataFrame(summaries)
    out_csv = out_dir / "v4_vs_baselines.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nWrote summary to {out_csv}")

    # Forest plot
    try:
        import matplotlib.pyplot as plt
        valid = df_summary[df_summary["n_pairs"] > 0]
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(8, 1.2 + 0.6 * len(valid)))
            for i, row in enumerate(valid.itertuples()):
                ax.errorbar(
                    row.mean_d_pp, i,
                    xerr=[[row.mean_d_pp - row.ci95_b5_lo_pp],
                          [row.ci95_b5_hi_pp - row.mean_d_pp]],
                    fmt="o", capsize=4, color="darkblue",
                )
                ax.text(row.ci95_b5_hi_pp + 0.1, i,
                        f"  p_W = {row.p_wilcoxon_one_sided:.2e}, n = {row.n_pairs}",
                        va="center", fontsize=9, color="gray")
            ax.axvline(0, color="gray", linestyle="--", lw=0.8)
            ax.set_yticks(range(len(valid)))
            ax.set_yticklabels([f"V4 vs {b}" for b in valid["baseline"]])
            ax.set_xlabel("Mean paired difference ∆ (pp of MDD), V4 - baseline")
            ax.set_title("V4 PPO vs baseline competitors")
            ax.grid(True, axis="x", linestyle=":", alpha=0.4)
            plt.tight_layout()
            out_svg = out_dir / "forest_v4_vs_baselines.svg"
            fig.savefig(out_svg)
            print(f"Wrote forest plot to {out_svg}")
    except ImportError:
        print("matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()

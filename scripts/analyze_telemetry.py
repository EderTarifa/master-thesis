"""
Analyze convergence telemetry for V0 and V4 (and baselines if requested).

Reads all per-run telemetry parquets and produces:
  1. A summary CSV per (variant, market) with mean, std, IQR of each metric
     in the final 20% of training (= "converged" region), INCLUDING a
     per-action-dimension normalised KL (approx_kl / action_dim) which is
     the metric to report in the paper because it is comparable across
     markets with different N.
  2. Figures with median + IQR per (variant, market) per metric:
       - curve_<metric>.svg          : raw metric
       - curve_approx_kl_per_dim.svg : KL normalised per action dimension
  3. A convergence-check table: variance ratio (last 20% / first 20%).

The key metrics for defending convergence to a Q1 reviewer:
  - approx_kl:           raw value, expected high in multi-dim action spaces
  - approx_kl_per_dim:   normalised, should sit in canonical 0.01-0.03 range
  - explained_variance:  should rise above 0.5 and stay there
  - clip_fraction:       expected high in multi-dim; not a non-convergence flag
  - entropy_loss:        should decrease (policy becomes more deterministic)
  - effective_step_norm: should decay

Action dimension per market (from data.py):
  DJIA: 29, SP50: 46-50, IBEX: 28, HSI: 35,
  BRD_CMDY: 25, CRYPTO: 14, BOND_US: 27, FX_MIX: 27

Usage
-----
    python scripts/analyze_telemetry.py \\
        --telemetry-dir results/full_optimal/telemetry \\
        --out-dir results/full_optimal/telemetry_analysis
"""
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PARQUET_RE = re.compile(
    r"^(?P<market>[A-Z0-9_]+)_f(?P<fold>\d{2})_(?P<variant>[A-Z0-9_]+)_s(?P<seed>\d+)\.parquet$"
)

# Number of assets per market (= action dimension of PortfolioEnv).
# Used to normalise approx_kl per action dimension. If a market is not
# listed here a sensible default (29) is used and a warning printed.
ACTION_DIM_BY_MARKET: dict[str, int] = {
    "DJIA": 29,       # 30 listed, 1 typically drops in cleaning
    "SP50": 46,       # paper uses 46-component subset
    "IBEX": 28,       # 35 listed, ~7 drop due to <95% completeness
    "HSI": 35,
    "BRD_CMDY": 25,
    "CRYPTO": 14,
    "BOND_US": 27,
    "FX_MIX": 27,
}
DEFAULT_ACTION_DIM = 29

METRICS = [
    "approx_kl",
    "approx_kl_per_dim",       # NEW: KL normalised per action dimension
    "explained_variance",
    "clip_fraction",
    "entropy_loss",
    "policy_gradient_loss",
    "value_loss",
    "effective_step_norm",
]


def load_all(telemetry_dir: Path) -> pd.DataFrame:
    """Load every telemetry parquet under telemetry_dir, return one long DF.

    Adds an `approx_kl_per_dim` column = approx_kl / action_dim_of_market.
    """
    rows = []
    unknown_markets: set[str] = set()
    for p in sorted(telemetry_dir.glob("*.parquet")):
        m = PARQUET_RE.match(p.name)
        if not m:
            print(f"WARN: skipping unparseable name {p.name}")
            continue
        df = pd.read_parquet(p)
        market = m["market"]
        df["market"] = market
        df["fold"] = int(m["fold"])
        df["variant"] = m["variant"]
        df["seed"] = int(m["seed"])
        df["iteration"] = np.arange(len(df))

        # Per-dimension KL normalisation. This is the metric to report
        # in the paper because the raw KL aggregates contributions across
        # all action components, making it incomparable between markets
        # with different N. Dividing by N gives the average KL per Gaussian
        # component, which is what PPO targets canonically (0.01-0.03).
        n_dim = ACTION_DIM_BY_MARKET.get(market)
        if n_dim is None:
            unknown_markets.add(market)
            n_dim = DEFAULT_ACTION_DIM
        if "approx_kl" in df.columns:
            df["approx_kl_per_dim"] = df["approx_kl"] / n_dim
        else:
            df["approx_kl_per_dim"] = np.nan

        rows.append(df)
    if unknown_markets:
        print(f"WARN: no action_dim configured for {sorted(unknown_markets)}; "
              f"using default {DEFAULT_ACTION_DIM}. Edit ACTION_DIM_BY_MARKET "
              f"at the top of this script to fix.")
    if not rows:
        raise SystemExit(f"No telemetry parquets in {telemetry_dir}")
    return pd.concat(rows, ignore_index=True)


def converged_summary(df: pd.DataFrame, last_frac: float = 0.20) -> pd.DataFrame:
    """For each (variant, market) compute stats over the last `last_frac`."""
    out = []
    for (variant, market), g in df.groupby(["variant", "market"]):
        n_iter = g["iteration"].max() + 1
        last_cut = int(n_iter * (1 - last_frac))
        last = g[g["iteration"] >= last_cut]
        first_cut = int(n_iter * last_frac)
        first = g[g["iteration"] < first_cut]
        row = {
            "variant": variant,
            "market": market,
            "n_runs": g["seed"].nunique(),
            "action_dim": ACTION_DIM_BY_MARKET.get(market, DEFAULT_ACTION_DIM),
        }
        for met in METRICS:
            if met not in g.columns:
                continue
            row[f"{met}_final_mean"] = last[met].mean()
            row[f"{met}_final_std"] = last[met].std()
            v_first = first[met].var()
            v_last = last[met].var()
            row[f"{met}_var_ratio"] = v_last / v_first if v_first > 1e-12 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def plot_metric_curves(df: pd.DataFrame, metric: str, out_path: Path,
                       group_by: tuple = ("variant", "market"),
                       reference_lines: list[tuple[float, str]] | None = None):
    """One subplot per (variant, market): median + IQR across seeds.

    Optional `reference_lines`: list of (y_value, label) to overlay
    horizontal reference lines (e.g. PPO canonical KL target 0.01-0.03).
    """
    grouped = df.groupby(list(group_by))
    n_panels = len(grouped)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                              sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()
    i = -1
    for i, (key, g) in enumerate(grouped):
        ax = axes[i]
        piv = g.pivot_table(index="iteration", columns="seed",
                             values=metric, aggfunc="mean")
        med = piv.median(axis=1)
        q25 = piv.quantile(0.25, axis=1)
        q75 = piv.quantile(0.75, axis=1)
        ax.plot(med.index, med.values, color="C0", lw=1.5, label="median")
        ax.fill_between(med.index, q25.values, q75.values,
                         alpha=0.25, color="C0", label="IQR")
        if reference_lines:
            for y_val, lbl in reference_lines:
                ax.axhline(y_val, color="red", linestyle="--", lw=0.8,
                           alpha=0.6, label=lbl)
        title = "  ".join(f"{k}={v}" for k, v in zip(group_by, key))
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    # Single legend if reference lines present
    if reference_lines and i >= 0:
        handles, labels = axes[0].get_legend_handles_labels()
        # Deduplicate
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        if unique:
            fig.legend([h for h, _ in unique], [l for _, l in unique],
                       loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle(metric, fontsize=12, fontweight="bold")
    fig.supxlabel("PPO rollout iteration")
    fig.supylabel(metric)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry-dir", default="results/full_optimal/telemetry")
    ap.add_argument("--out-dir", default="results/full_optimal/telemetry_analysis")
    args = ap.parse_args()
    tel_dir = Path(args.telemetry_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading telemetry from {tel_dir} ...")
    df = load_all(tel_dir)
    print(f"  {len(df):,} rows across "
          f"{df['variant'].nunique()} variants, "
          f"{df['market'].nunique()} markets, "
          f"{df.groupby(['variant','market','fold'])['seed'].nunique().mean():.1f} avg seeds/cell")

    summary = converged_summary(df, last_frac=0.20)
    summary_path = out_dir / "convergence_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nConvergence summary -> {summary_path}\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nGenerating per-metric figures ...")
    # Reference lines: PPO canonical targets for KL.
    kl_refs = [(0.01, "PPO target lower (0.01)"),
               (0.03, "PPO target upper (0.03)")]
    clip_refs = [(0.05, "Healthy lower (0.05)"),
                 (0.20, "Healthy upper (0.20)")]
    ev_refs = [(0.50, "Convergence threshold (0.5)")]

    for met in METRICS:
        if met not in df.columns:
            print(f"  skip {met} (column absent)")
            continue
        out_svg = out_dir / f"curve_{met}.svg"
        refs = None
        if met == "approx_kl_per_dim":
            refs = kl_refs
        elif met == "clip_fraction":
            refs = clip_refs
        elif met == "explained_variance":
            refs = ev_refs
        plot_metric_curves(df, met, out_svg, reference_lines=refs)
        print(f"  {out_svg}")

    print("\n=== Per-variant summary across markets ===")
    cols_of_interest = [
        "approx_kl_final_mean",
        "approx_kl_per_dim_final_mean",
        "explained_variance_final_mean",
        "clip_fraction_final_mean",
        "entropy_loss_final_mean",
        "effective_step_norm_final_mean",
    ]
    cols_of_interest = [c for c in cols_of_interest if c in summary.columns]
    by_variant = summary.groupby("variant")[cols_of_interest].mean()
    print(by_variant.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== Convergence interpretation ===")
    if "approx_kl_per_dim_final_mean" in summary.columns:
        kl_per_dim_overall = summary["approx_kl_per_dim_final_mean"].mean()
        print(f"  approx_kl_per_dim (mean over all cells): {kl_per_dim_overall:.4f}")
        if 0.005 <= kl_per_dim_overall <= 0.05:
            print("    -> within canonical PPO range (0.01-0.03 ± buffer): HEALTHY")
        elif kl_per_dim_overall > 0.05:
            print("    -> above canonical range: policy may be moving too fast")
        else:
            print("    -> below canonical range: policy may be stagnating")
    ev_overall = summary["explained_variance_final_mean"].mean()
    print(f"  explained_variance (mean over all cells): {ev_overall:.4f}")
    if ev_overall >= 0.7:
        print("    -> value head learned strongly: CONVERGENCE DEFENSIBLE")
    elif ev_overall >= 0.5:
        print("    -> value head learned acceptably")
    elif ev_overall >= 0.0:
        print("    -> value head weak; defence harder")
    else:
        print("    -> value head FAILED: convergence cannot be defended")


if __name__ == "__main__":
    main()
"""
Quick convergence check for the POTPG and EXDRL baselines.

Unlike analyze_telemetry.py (which targets PPO internals: KL, explained
variance, clip fraction), this script reads the baseline-specific telemetry
columns:

  POTPG : fit_mode, gpd_converged, gpd_shape, gpd_scale, cvar_estimate, ...
  EXDRL : critic_loss, actor_loss, qr_loss, gpd_xi, gpd_sigma, ...

It writes a small summary CSV and prints a human-readable convergence verdict
for each baseline. Use it to defend, to a Q1 reviewer, that the competitor
baselines were trained to convergence (POTPG genuinely uses the GPD tail;
EXDRL's distributional critic learns).

Usage
-----
    python scripts/check_baseline_convergence.py \
        --telemetry-dir results/baselines/telemetry \
        --out-dir results/baselines/telemetry_analysis
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def check_potpg(telemetry_dir: Path) -> dict | None:
    """Aggregate POTPG telemetry; return summary dict or None if no files."""
    files = sorted(glob.glob(str(telemetry_dir / "*_POTPG_*.parquet")))
    if not files:
        print("POTPG: no telemetry files found.")
        return None

    allg = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fit_counts = allg["fit_mode"].value_counts().to_dict()
    n_total = len(allg)
    n_conv = int(allg["gpd_converged"].sum())
    gpd_pct = 100.0 * fit_counts.get("gpd", 0) / n_total

    summary = {
        "baseline": "POTPG",
        "n_runs": len(files),
        "n_iterations_total": n_total,
        "fit_mode_gpd": fit_counts.get("gpd", 0),
        "fit_mode_empirical": fit_counts.get("empirical", 0),
        "pct_gpd": round(gpd_pct, 1),
        "gpd_converged_count": n_conv,
        "gpd_converged_pct": round(100.0 * n_conv / n_total, 1),
        "gpd_shape_median": round(float(allg["gpd_shape"].median(skipna=True)), 4),
        "gpd_shape_min": round(float(allg["gpd_shape"].min(skipna=True)), 4),
        "gpd_shape_max": round(float(allg["gpd_shape"].max(skipna=True)), 4),
        "gpd_scale_median": round(float(allg["gpd_scale"].median(skipna=True)), 4),
        "n_excesses_mean": round(float(allg["n_excesses"].mean()), 2),
    }

    print(f"\n=== POTPG ({len(files)} runs, {n_total} iterations) ===")
    print(f"  fit_mode               : {fit_counts}")
    print(f"  GPD used (vs empirical): {gpd_pct:.1f}% of iterations")
    print(f"  gpd_converged          : {n_conv}/{n_total} "
          f"({summary['gpd_converged_pct']:.1f}%)")
    print(f"  gpd_shape (xi)         : median={summary['gpd_shape_median']:.3f}, "
          f"range=[{summary['gpd_shape_min']:.3f}, {summary['gpd_shape_max']:.3f}]")
    print(f"  gpd_scale (sigma)      : median={summary['gpd_scale_median']:.3f}")
    print(f"  n_excesses             : mean={summary['n_excesses_mean']:.1f}")

    # Verdict
    if gpd_pct < 50:
        print("  VERDICT: POTPG mostly falls back to empirical CVaR. "
              "Tail head NOT genuinely active -> consider raising "
              "n_trajectories_per_batch.")
    elif abs(summary["gpd_shape_median"]) < 1e-6:
        print("  VERDICT: GPD fits run but shape is ~0 everywhere "
              "-> MLE may be failing silently (check evt.fit_gpd guard).")
    else:
        print("  VERDICT: POTPG genuinely uses the GPD tail. OK.")
    return summary


def check_exdrl(telemetry_dir: Path) -> dict | None:
    """Aggregate EXDRL telemetry; return summary dict or None if no files."""
    files = sorted(glob.glob(str(telemetry_dir / "*_EXDRL_*.parquet")))
    if not files:
        print("EXDRL: no telemetry files found.")
        return None

    # Per-run critic-loss reduction (first snapshot minus last).
    deltas = []
    qr_deltas = []
    for f in files:
        df = pd.read_parquet(f)
        if len(df) < 2:
            continue
        deltas.append(df["critic_loss"].iloc[0] - df["critic_loss"].iloc[-1])
        if "qr_loss" in df.columns:
            qr_deltas.append(df["qr_loss"].iloc[0] - df["qr_loss"].iloc[-1])
    deltas = np.asarray(deltas, dtype=float)
    qr_deltas = np.asarray(qr_deltas, dtype=float)

    allg = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n_dropped = int((deltas > 0).sum())

    summary = {
        "baseline": "EXDRL",
        "n_runs": len(files),
        "n_snapshots_total": len(allg),
        "critic_decreased_runs": n_dropped,
        "critic_decreased_pct": round(100.0 * n_dropped / max(len(deltas), 1), 1),
        "critic_loss_reduction_mean": round(float(deltas.mean()), 4) if deltas.size else float("nan"),
        "qr_loss_reduction_mean": round(float(qr_deltas.mean()), 4) if qr_deltas.size else float("nan"),
        "gpd_xi_mean": round(float(allg["gpd_xi"].mean()), 4),
        "gpd_xi_min": round(float(allg["gpd_xi"].min()), 4),
        "gpd_xi_max": round(float(allg["gpd_xi"].max()), 4),
        "gpd_sigma_mean": round(float(allg["gpd_sigma"].mean()), 4),
    }

    print(f"\n=== EXDRL ({len(files)} runs, {len(allg)} snapshots) ===")
    print(f"  critic_loss decreased  : {n_dropped}/{len(deltas)} runs "
          f"({summary['critic_decreased_pct']:.1f}%)")
    print(f"  mean critic reduction  : {summary['critic_loss_reduction_mean']:.4f}")
    print(f"  mean qr_loss reduction : {summary['qr_loss_reduction_mean']:.4f}")
    print(f"  gpd_xi                 : mean={summary['gpd_xi_mean']:.3f}, "
          f"range=[{summary['gpd_xi_min']:.3f}, {summary['gpd_xi_max']:.3f}]")
    print(f"  gpd_sigma              : mean={summary['gpd_sigma_mean']:.4f}")

    # Verdict
    default_xi = abs(summary["gpd_xi_mean"]) < 1e-6 and abs(summary["gpd_sigma_mean"] - 1e-3) < 1e-6
    if default_xi:
        print("  VERDICT: GPD stuck at defaults (xi=0, sigma=0.001) "
              "-> tail head never activated. Check buffer fill / refit.")
    elif summary["critic_decreased_pct"] < 50:
        print("  VERDICT: critic loss did not decrease in most runs "
              "-> convergence questionable.")
    else:
        print("  VERDICT: EXDRL critic converges and GPD tail is active. OK.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry-dir", default="results/baselines/telemetry")
    ap.add_argument("--out-dir", default="results/baselines/telemetry_analysis")
    args = ap.parse_args()

    tel_dir = Path(args.telemetry_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    s_potpg = check_potpg(tel_dir)
    if s_potpg:
        rows.append(s_potpg)
    s_exdrl = check_exdrl(tel_dir)
    if s_exdrl:
        rows.append(s_exdrl)

    if rows:
        # Two baselines have different columns; write each to its own CSV
        # plus a combined one with the shared keys.
        for r in rows:
            name = r["baseline"].lower()
            pd.DataFrame([r]).to_csv(out_dir / f"convergence_{name}.csv", index=False)
        print(f"\nWrote per-baseline CSVs to {out_dir}/convergence_*.csv")
    else:
        print("\nNo baseline telemetry found; nothing written.")


if __name__ == "__main__":
    main()  
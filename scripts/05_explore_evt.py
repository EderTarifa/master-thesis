"""
Standalone exploratory script: visualise the EVT features on a synthetic
panel to convince yourself (and the tribunal) that the EVT estimator
behaves sensibly.

Plots produced (in results/evt_exploration/):
    - histogram of portfolio losses with fitted GPD tail.
    - rolling time-series of xi, sigma, VaR99, CVaR99, exceedance freq.
    - threshold sensitivity: how do xi and CVaR change with the threshold
      quantile?

Run:
    python scripts/05_explore_evt.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evt_ppo import data as D
from evt_ppo import evt as E
from evt_ppo import plots as P


def main() -> None:
    out_dir = Path("results/evt_exploration")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build a single equally-weighted "portfolio" from the synthetic panel.
    df = D.synthetic_market(n_assets=10, n_days=2500, seed=7)
    log_ret = np.log(df / df.shift(1)).dropna()
    portfolio_log = log_ret.mean(axis=1).values
    losses = -portfolio_log

    # 2. Fit GPD on the full sample and overlay the empirical loss histogram.
    threshold, fit = E.select_threshold_auto(losses)
    print(f"Auto-selected threshold = {threshold:.4f}")
    print(f"  xi    = {fit.shape:.4f}")
    print(f"  sigma = {fit.scale:.4f}")
    print(f"  VaR99 = {E.gpd_var(fit, 0.99):.4f}")
    print(f"  CVaR99= {E.gpd_cvar(fit, 0.99):.4f}")
    print(f"  n_excesses = {fit.n_excesses}, F_u = {fit.exceedance_prob:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(losses, bins=60, density=True, alpha=0.5, label="empirical")
    ax.axvline(threshold, color="red", lw=2, label=f"threshold={threshold:.4f}")
    ax.axvline(E.gpd_var(fit, 0.99), color="black", ls="--", lw=2,
               label=f"VaR99={E.gpd_var(fit, 0.99):.4f}")
    ax.axvline(E.gpd_cvar(fit, 0.99), color="purple", ls="--", lw=2,
               label=f"CVaR99={E.gpd_cvar(fit, 0.99):.4f}")
    ax.set_xlabel("Daily portfolio loss")
    ax.set_ylabel("Density")
    ax.set_title("EW portfolio losses with EVT-fitted tail markers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_histogram_with_evt.png", dpi=150)
    plt.close(fig)

    # 3. Rolling EVT features.
    window = 250
    n = len(losses)
    feat_history = np.full((n, 5), np.nan, dtype=float)
    for t in range(window, n):
        win = portfolio_log[t - window:t]
        feat_history[t] = E.evt_state_features(win)

    P.plot_evt_features_timeseries(
        feat_history[window:],
        save_path=out_dir / "rolling_evt_features",
        title="Rolling EVT features on portfolio returns (W=250)",
    )

    # 4. Threshold sensitivity.
    quantiles = np.arange(0.80, 0.98, 0.01)
    xis, cvars = [], []
    for q in quantiles:
        f = E.fit_gpd(losses, quantile=q)
        xis.append(f.shape)
        cvars.append(E.gpd_cvar(f, 0.99))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(quantiles, xis, "o-")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_xlabel("Threshold quantile")
    axes[0].set_ylabel("xi (shape)")
    axes[0].set_title("Threshold sensitivity of xi")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(quantiles, cvars, "o-", color="C1")
    axes[1].set_xlabel("Threshold quantile")
    axes[1].set_ylabel("CVaR_99")
    axes[1].set_title("Threshold sensitivity of CVaR_99")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_sensitivity.png", dpi=150)
    plt.close(fig)

    print(f"\nAll plots saved to {out_dir.absolute()}")


if __name__ == "__main__":
    main()

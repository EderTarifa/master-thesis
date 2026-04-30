"""
Plotting utilities for the experimental results.

All plots are saved as PNG and SVG to the supplied output directory.
The visual style follows the financial research conventions: equity
curves on a log scale, drawdown curves underneath in absolute units,
and MDD distribution comparisons via paired boxplots.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .drawdown import underwater_curve


# ---------------------------------------------------------------------------
# Single-window equity / drawdown plots
# ---------------------------------------------------------------------------


def plot_equity_curves(
    series: dict[str, np.ndarray],
    title: str = "Equity curves",
    save_path: Path | str | None = None,
    log_scale: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Plot multiple equity curves on the same axes.

    series : dict
        Maps name -> 1-D array of portfolio values.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, vals in series.items():
        ax.plot(vals, label=name, lw=1.4)
    ax.set_title(title)
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Portfolio value")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


def plot_drawdown_curves(
    series: dict[str, np.ndarray],
    title: str = "Drawdown curves",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    for name, vals in series.items():
        dd = underwater_curve(vals)
        ax.fill_between(np.arange(len(dd)), 0, -dd, alpha=0.25, label=name)
        ax.plot(-dd, lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


def plot_value_and_drawdown(
    series: dict[str, np.ndarray],
    title: str = "",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 7),
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    for name, vals in series.items():
        axes[0].plot(vals, label=name, lw=1.4)
        dd = underwater_curve(vals)
        axes[1].fill_between(np.arange(len(dd)), 0, -dd, alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Portfolio value")
    axes[0].set_title(title)
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Trading day")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Aggregate distribution plots over folds and seeds
# ---------------------------------------------------------------------------


def plot_mdd_boxplots(
    df_long: pd.DataFrame,
    variant_col: str = "variant",
    mdd_col: str = "mdd",
    title: str = "MDD distribution by variant",
    save_path: Path | str | None = None,
    order: list[str] | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Boxplot of MDD across (fold, seed) for each variant.

    df_long must have columns 'variant' and 'mdd'.
    """
    fig, ax = plt.subplots(figsize=figsize)
    groups = order or sorted(df_long[variant_col].unique())
    data = [df_long.loc[df_long[variant_col] == g, mdd_col].values for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a0c4ff")
    ax.set_ylabel("Maximum Drawdown")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


def plot_paired_diffs(
    diffs: np.ndarray,
    name_left: str = "V1 (no EVT)",
    name_right: str = "V4 (EVT)",
    title: str = "Paired difference in MDD: V4 - V1",
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Histogram of paired (per-fold per-seed) MDD differences.

    A negative value means EVT improved over no-EVT for that fold/seed.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(diffs, bins=30, color="#a0c4ff", edgecolor="black")
    ax.axvline(0.0, color="black", lw=1, ls="--")
    ax.axvline(np.mean(diffs), color="red", lw=2, label=f"mean = {np.mean(diffs):+.4f}")
    ax.set_xlabel(f"MDD({name_right}) - MDD({name_left})")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


def plot_metric_table(
    df: pd.DataFrame,
    save_path: Path | str | None = None,
    title: str = "",
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Render a metric summary table as a figure (useful for the TFM PDF)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    cell_text = df.round(4).values.tolist()
    table = ax.table(
        cellText=cell_text,
        rowLabels=df.index.tolist(),
        colLabels=df.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


def plot_evt_features_timeseries(
    evt_features: np.ndarray,
    feature_names: tuple[str, ...] = ("xi", "sigma", "VaR99", "CVaR99", "exc_freq"),
    save_path: Path | str | None = None,
    title: str = "Rolling EVT features on portfolio returns",
    figsize: tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Plot the rolling EVT features as one panel per feature."""
    n_feats = evt_features.shape[1]
    fig, axes = plt.subplots(n_feats, 1, figsize=figsize, sharex=True)
    if n_feats == 1:
        axes = [axes]
    for i in range(n_feats):
        axes[i].plot(evt_features[:, i], lw=1.0)
        axes[i].set_ylabel(feature_names[i])
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Trading day")
    axes[0].set_title(title)
    fig.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


def _save(fig: plt.Figure, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

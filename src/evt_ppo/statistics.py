"""
Hypothesis testing for paired and unpaired comparisons of MDD distributions
across variants and seeds.

The TFM contrasts:

    H0: mean(MDD_evt) >= mean(MDD_base)
    H1: mean(MDD_evt) <  mean(MDD_base)

Recommended test for the TFM: paired by (window, seed), one-sided
Wilcoxon signed-rank because the MDD distribution is bounded and
typically right-skewed. We also report:

    - Paired t-test (one-sided).
    - Block bootstrap confidence interval for the mean difference.

Block bootstrap is needed because adjacent walk-forward windows are
not independent (overlapping market history).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    test_name: str
    statistic: float
    pvalue: float
    n: int
    mean_diff: float
    diff_ci_lower: float
    diff_ci_upper: float
    reject_h0: bool
    alpha: float


def paired_one_sided_t(
    mdd_base: np.ndarray,
    mdd_evt: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """Paired t-test, one-sided alternative mean(evt) < mean(base)."""
    base = np.asarray(mdd_base, dtype=float).ravel()
    evt = np.asarray(mdd_evt, dtype=float).ravel()
    if base.shape != evt.shape:
        raise ValueError(f"Shape mismatch: {base.shape} vs {evt.shape}")
    diff = evt - base  # negative => EVT improves
    res = stats.ttest_rel(evt, base, alternative="less")
    n = base.size
    mean_d = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    crit = stats.t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else 0.0
    return TestResult(
        test_name="paired_t_one_sided",
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        n=n,
        mean_diff=mean_d,
        diff_ci_lower=mean_d - crit * se,
        diff_ci_upper=mean_d + crit * se,
        reject_h0=bool(res.pvalue < alpha),
        alpha=alpha,
    )


def paired_one_sided_wilcoxon(
    mdd_base: np.ndarray,
    mdd_evt: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """Wilcoxon signed-rank, one-sided alternative mean(evt) < mean(base)."""
    base = np.asarray(mdd_base, dtype=float).ravel()
    evt = np.asarray(mdd_evt, dtype=float).ravel()
    if base.shape != evt.shape:
        raise ValueError(f"Shape mismatch: {base.shape} vs {evt.shape}")
    diff = evt - base
    res = stats.wilcoxon(evt, base, alternative="less", zero_method="pratt")
    n = base.size
    mean_d = float(np.mean(diff))
    return TestResult(
        test_name="paired_wilcoxon_one_sided",
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        n=n,
        mean_diff=mean_d,
        diff_ci_lower=float(np.nan),
        diff_ci_upper=float(np.nan),
        reject_h0=bool(res.pvalue < alpha),
        alpha=alpha,
    )


def block_bootstrap_diff_ci(
    mdd_base: np.ndarray,
    mdd_evt: np.ndarray,
    block_size: int = 5,
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Block bootstrap CI for the mean difference (evt - base).

    Used because consecutive walk-forward windows are temporally
    overlapping, breaking independence.

    Returns (mean_diff, ci_lower, ci_upper).
    """
    base = np.asarray(mdd_base, dtype=float).ravel()
    evt = np.asarray(mdd_evt, dtype=float).ravel()
    if base.shape != evt.shape:
        raise ValueError("Shape mismatch.")
    diff = evt - base
    n = diff.size
    if n < block_size:
        block_size = max(1, n // 2)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    means = np.zeros(n_resamples)
    for b in range(n_resamples):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        means[b] = diff[idx].mean()
    mean_d = float(diff.mean())
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return mean_d, lo, hi


def full_comparison(
    mdd_base: np.ndarray,
    mdd_evt: np.ndarray,
    alpha: float = 0.05,
    block_size: int = 5,
    n_bootstrap: int = 5000,
    seed: int = 0,
) -> dict:
    """Run all three tests and return a structured dict."""
    t_res = paired_one_sided_t(mdd_base, mdd_evt, alpha=alpha)
    w_res = paired_one_sided_wilcoxon(mdd_base, mdd_evt, alpha=alpha)
    mean_d, lo, hi = block_bootstrap_diff_ci(
        mdd_base, mdd_evt,
        block_size=block_size, n_resamples=n_bootstrap,
        alpha=alpha, seed=seed,
    )
    return {
        "n": t_res.n,
        "mean_mdd_base": float(np.mean(mdd_base)),
        "mean_mdd_evt": float(np.mean(mdd_evt)),
        "mean_diff": mean_d,
        "median_diff": float(np.median(np.asarray(mdd_evt) - np.asarray(mdd_base))),
        "paired_t_pvalue": t_res.pvalue,
        "paired_t_reject": t_res.reject_h0,
        "wilcoxon_pvalue": w_res.pvalue,
        "wilcoxon_reject": w_res.reject_h0,
        "bootstrap_ci_lower": lo,
        "bootstrap_ci_upper": hi,
        "alpha": alpha,
    }

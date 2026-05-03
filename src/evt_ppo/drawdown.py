"""
Drawdown calculation and related risk metrics.

A drawdown at time t is the relative decline of portfolio value from its
running maximum up to t:
    DD(t) = (max_{s<=t} V(s) - V(t)) / max_{s<=t} V(s) in [0, 1].

The Maximum Drawdown (MDD) over horizon T is max_t DD(t).
Conditional Drawdown-at-Risk (CDaR) is the mean of the worst (1-alpha)*100%
drawdowns of the underwater curve, the drawdown-analogue of CVaR
(Chekhlov, Uryasev & Zabarankin, 2005).
"""
from __future__ import annotations

import numpy as np


def underwater_curve(values: np.ndarray) -> np.ndarray:
    """Return the underwater (drawdown) curve from a series of portfolio values.

    Parameters
    ----------
    values : np.ndarray, shape (T+1,)
        Portfolio values, V_0..V_T (positive).

    Returns
    -------
    dd : np.ndarray, shape (T+1,)
        Drawdown at each time, in [0, 1].
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be 1-D.")
    running_max = np.maximum.accumulate(values)
    # Avoid division by zero: if running_max is 0, treat dd as 0.
    safe_max = np.where(running_max > 0, running_max, 1.0)
    dd = (running_max - values) / safe_max
    return np.clip(dd, 0.0, 1.0)


def maximum_drawdown(values: np.ndarray) -> float:
    """Maximum drawdown of a value series."""
    if len(values) == 0:
        return 0.0
    return float(underwater_curve(values).max())


def current_drawdown(values: np.ndarray) -> float:
    """Drawdown at the last point of the series (used in environment state)."""
    if len(values) == 0:
        return 0.0
    return float(underwater_curve(values)[-1])


def conditional_drawdown_at_risk(values: np.ndarray, alpha: float = 0.95) -> float:
    """Empirical CDaR at level alpha.

    CDaR_alpha = mean of the worst (1-alpha) fraction of drawdowns.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    dd = underwater_curve(values)
    if dd.size == 0:
        return 0.0
    threshold = np.quantile(dd, alpha)
    tail = dd[dd >= threshold]
    return float(tail.mean()) if tail.size > 0 else float(threshold)


def calmar_ratio(values: np.ndarray, periods_per_year: int = 252) -> float:
    """Calmar ratio = annualised return / maximum drawdown.

    Returns 0 if MDD is zero (degenerate cash-only strategy).
    """
    if len(values) < 2:
        return 0.0
    n_periods = len(values) - 1
    total_return = values[-1] / values[0] - 1.0
    annualised = (1.0 + total_return)**(periods_per_year / n_periods) - 1.0
    mdd = maximum_drawdown(values)
    if mdd <= 1e-9:
        return 0.0
    return float(annualised / mdd)


def drawdown_increment(prev_dd: float, new_dd: float) -> float:
    """Non-negative drawdown increment used in step-by-step rewards.

    Returns max(0, new_dd - prev_dd).
    """
    return float(max(0.0, new_dd - prev_dd))

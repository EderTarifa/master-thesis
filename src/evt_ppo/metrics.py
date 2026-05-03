"""
Performance metrics for portfolio evaluation.

All functions accept a 1-D array of portfolio values (V_0 ... V_T) and
return a scalar. The conventions:
    - "annualised" means scaled by `periods_per_year` (252 for daily).
    - Returns are computed as simple returns from the value series.
    - All metrics are robust to all-zero or constant series, returning 0.
"""
from __future__ import annotations

import numpy as np

from .drawdown import (
    conditional_drawdown_at_risk,
    maximum_drawdown,
    underwater_curve,
)


def _simple_returns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.array([])
    return values[1:] / values[:-1] - 1.0


def total_return(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(values[-1] / values[0] - 1.0)


def cagr(values: np.ndarray, periods_per_year: int = 252) -> float:
    """Compound annual growth rate."""
    if len(values) < 2 or values[0] <= 0 or values[-1] <= 0:
        return 0.0
    n_periods = len(values) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float((values[-1] / values[0]) ** (1.0 / years) - 1.0)


def annualised_volatility(values: np.ndarray, periods_per_year: int = 252) -> float:
    r = _simple_returns(values)
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    values: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sharpe ratio. risk_free_rate is annualised."""
    r = _simple_returns(values)
    if r.size < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = r - rf_per_period
    sd = np.std(excess, ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    values: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sortino ratio (downside deviation)."""
    r = _simple_returns(values)
    if r.size < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = r - rf_per_period
    downside = excess[excess < 0]
    if downside.size < 2:
        return 0.0
    dd_std = np.std(downside, ddof=1)
    if dd_std <= 1e-12:
        return 0.0
    return float(np.mean(excess) / dd_std * np.sqrt(periods_per_year))


def calmar_ratio(values: np.ndarray, periods_per_year: int = 252) -> float:
    """Calmar = CAGR / MDD."""
    mdd = maximum_drawdown(values)
    if mdd <= 1e-9:
        return 0.0
    return float(cagr(values, periods_per_year) / mdd)


def cvar_returns(values: np.ndarray, alpha: float = 0.95) -> float:
    """Empirical CVaR at level alpha applied to *losses* (-returns).

    Returns the absolute value of the average of the worst (1-alpha) returns.
    """
    r = _simple_returns(values)
    if r.size == 0:
        return 0.0
    losses = -r
    threshold = np.quantile(losses, alpha)
    tail = losses[losses >= threshold]
    return float(tail.mean()) if tail.size > 0 else float(threshold)


def var_returns(values: np.ndarray, alpha: float = 0.95) -> float:
    """Empirical VaR at level alpha applied to *losses*."""
    r = _simple_returns(values)
    if r.size == 0:
        return 0.0
    return float(np.quantile(-r, alpha))


def drawdown_quantile(values: np.ndarray, q: float = 0.95) -> float:
    """The q-th quantile of the drawdown distribution."""
    dd = underwater_curve(values)
    if dd.size == 0:
        return 0.0
    return float(np.quantile(dd, q))


def turnover_mean(weights_history: np.ndarray) -> float:
    """Average L1 turnover across consecutive weight snapshots.

    Parameters
    ----------
    weights_history : np.ndarray, shape (T+1, N)
    """
    w = np.asarray(weights_history, dtype=float)
    if w.ndim != 2 or w.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(w, axis=0)).sum(axis=1)))


def all_metrics(
    values: np.ndarray,
    weights_history: np.ndarray | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute the full metric panel used in the TFM tables."""
    out: dict[str, float] = {
        "total_return": total_return(values),
        "cagr": cagr(values, periods_per_year),
        "vol_annualised": annualised_volatility(values, periods_per_year),
        "sharpe": sharpe_ratio(values, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(values, risk_free_rate, periods_per_year),
        "mdd": maximum_drawdown(values),
        "calmar": calmar_ratio(values, periods_per_year),
        "cvar_95": cvar_returns(values, 0.95),
        "cvar_99": cvar_returns(values, 0.99),
        "var_95": var_returns(values, 0.95),
        "dd_q95": drawdown_quantile(values, 0.95),
        "cdar_95": conditional_drawdown_at_risk(values, 0.95),
    }
    if weights_history is not None:
        out["turnover_mean"] = turnover_mean(weights_history)
    return out

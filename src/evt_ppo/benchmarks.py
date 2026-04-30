"""
Classical portfolio benchmarks evaluated on the same walk-forward
splits as the RL agents.

Implemented benchmarks:
    - Equal-weight (1/N), the strongest naive baseline (DeMiguel et al.).
    - Minimum-variance: argmin w' Sigma w, s.t. sum w = 1, w >= 0.
    - Mean-variance (Markowitz): argmax w' mu - 0.5 * gamma * w' Sigma w,
      s.t. sum w = 1, w >= 0.
    - Buy-and-hold of equal-weight at t=0.

All benchmarks use only the *training* slice of returns to estimate
parameters, then are rolled forward over the test slice with monthly
rebalancing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize


@dataclass
class BacktestResult:
    name: str
    values: np.ndarray         # (T+1,)
    weights_history: np.ndarray  # (T+1, N)


def _project_simplex(w: np.ndarray) -> np.ndarray:
    """Project onto the unit simplex (long-only, sum=1) for numerical safety."""
    w = np.maximum(w, 0)
    s = w.sum()
    if s <= 1e-12:
        return np.ones_like(w) / w.size
    return w / s


def equal_weight(n_assets: int) -> np.ndarray:
    return np.ones(n_assets) / n_assets


def minimum_variance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    w0 = equal_weight(n)
    # Add a small ridge for numerical stability.
    cov_reg = cov + 1e-6 * np.eye(n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bnds = [(0.0, 1.0)] * n
    res = minimize(
        fun=lambda w: float(w @ cov_reg @ w),
        x0=w0,
        jac=lambda w: 2.0 * cov_reg @ w,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-9, "maxiter": 200},
    )
    if res.success:
        return _project_simplex(res.x)
    return w0


def mean_variance(mean: np.ndarray, cov: np.ndarray, gamma: float = 5.0) -> np.ndarray:
    """Maximise w'mu - 0.5*gamma*w'Sigma w subject to long-only, sum=1."""
    n = cov.shape[0]
    w0 = equal_weight(n)
    cov_reg = cov + 1e-6 * np.eye(n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bnds = [(0.0, 1.0)] * n
    res = minimize(
        fun=lambda w: -float(w @ mean) + 0.5 * gamma * float(w @ cov_reg @ w),
        x0=w0,
        jac=lambda w: -mean + gamma * cov_reg @ w,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-9, "maxiter": 200},
    )
    if res.success:
        return _project_simplex(res.x)
    return w0


def _backtest_strategy(
    log_returns_test: np.ndarray,    # (T_test, N)
    weight_fn: Callable[[int], np.ndarray],
    rebalance_freq: int = 21,
    initial_value: float = 1.0,
    transaction_cost: float = 0.0010,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a strategy with periodic rebalancing.

    `weight_fn(t)` returns the desired target weights at step t (relative
    to the start of the test window).
    """
    T, N = log_returns_test.shape
    simple_returns = np.expm1(log_returns_test)
    values = np.zeros(T + 1)
    values[0] = initial_value
    weights_hist = np.zeros((T + 1, N))
    w = weight_fn(0)
    weights_hist[0] = w
    for t in range(T):
        if t > 0 and t % rebalance_freq == 0:
            new_w = weight_fn(t)
            cost = transaction_cost * np.abs(new_w - w).sum()
            values[t] *= (1.0 - cost)
            w = new_w
        port_ret = float(w @ simple_returns[t])
        values[t + 1] = values[t] * (1.0 + port_ret)
        # Drift weights by realised returns (between rebalances).
        gross = w * (1.0 + simple_returns[t])
        denom = gross.sum()
        if denom > 1e-12:
            w = gross / denom
        weights_hist[t + 1] = w
    return values, weights_hist


def run_benchmarks(
    train_log_returns: np.ndarray,
    test_log_returns: np.ndarray,
    rebalance_freq: int = 21,
    transaction_cost: float = 0.0010,
    gamma_meanvar: float = 5.0,
    initial_value: float = 1.0,
) -> dict[str, BacktestResult]:
    """Run all benchmarks and return a dict keyed by name."""
    n = train_log_returns.shape[1]

    # Plug-in estimates from the training slice (annualised mean for MV).
    mu_daily = train_log_returns.mean(axis=0)
    mu_annual = mu_daily * 252
    cov_annual = np.cov(train_log_returns.T, ddof=1) * 252

    eq_w = equal_weight(n)
    minvar_w = minimum_variance(cov_annual)
    mv_w = mean_variance(mu_annual, cov_annual, gamma=gamma_meanvar)

    results: dict[str, BacktestResult] = {}
    for name, w_target in [
        ("equal_weight", eq_w),
        ("min_variance", minvar_w),
        ("mean_variance", mv_w),
    ]:
        v, wh = _backtest_strategy(
            test_log_returns,
            weight_fn=lambda t, w=w_target: w,
            rebalance_freq=rebalance_freq,
            initial_value=initial_value,
            transaction_cost=transaction_cost,
        )
        results[name] = BacktestResult(name=name, values=v, weights_history=wh)

    # Buy-and-hold of equal weight: weights drift, never rebalance.
    bh_v, bh_wh = _backtest_strategy(
        test_log_returns,
        weight_fn=lambda t: eq_w,
        rebalance_freq=10**9,    # effectively never
        initial_value=initial_value,
        transaction_cost=transaction_cost,
    )
    results["buy_and_hold_eq"] = BacktestResult(
        name="buy_and_hold_eq", values=bh_v, weights_history=bh_wh,
    )

    return results

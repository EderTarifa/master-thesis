"""
State construction for the portfolio management environment.

The agent state is the concatenation of three blocks:

1. Market block X^mkt: a flattened window of recent log-returns for the
   N assets over the last L days, optionally augmented with simple
   indicators. Shape: (N * L,) by default, possibly normalised.

2. Portfolio block x^port: the current portfolio weights, the
   normalised portfolio value, and the current drawdown. Shape: (N + 2,).

3. EVT block x^evt (optional, only if include_evt=True): five EVT
   features computed on the rolling window of *portfolio* returns:
   shape (xi), scale (sigma), VaR_99, CVaR_99, exceedance freq.
   Shape: (5,).

All blocks are returned as float32 arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evt import evt_state_features


@dataclass
class StateConfig:
    """Configuration for state construction."""
    window_length: int = 60          # L: market window size in days
    evt_window: int = 250            # window used for EVT estimation
    include_evt: bool = False        # whether to append EVT features
    normalise_market: bool = True    # subtract per-asset mean, divide by std
    use_auto_threshold: bool = True  # automated threshold for EVT


def market_block(
    returns_window: np.ndarray,  # (L, N)
    cfg: StateConfig,
) -> np.ndarray:
    """Flatten and (optionally) normalise the market window."""
    if cfg.normalise_market:
        # Per-asset z-score using the window itself. Robust to scale
        # differences across assets.
        mu = returns_window.mean(axis=0, keepdims=True)
        sd = returns_window.std(axis=0, keepdims=True) + 1e-8
        norm = (returns_window - mu) / sd
    else:
        norm = returns_window
    return norm.astype(np.float32).ravel()


def portfolio_block(
    weights: np.ndarray,
    value_normalised: float,
    current_dd: float,
) -> np.ndarray:
    """Concatenate weights, value, drawdown into a portfolio block."""
    block = np.concatenate([
        weights.astype(np.float32),
        np.array([value_normalised, current_dd], dtype=np.float32),
    ])
    return block


def evt_block(
    portfolio_returns_window: np.ndarray,
    cfg: StateConfig,
) -> np.ndarray:
    """Compute the 5-dim EVT feature vector from portfolio returns."""
    if portfolio_returns_window.size < 50:
        # Not enough history yet; return zeros. Episode rollouts pad the
        # initial portfolio history with zeros.
        return np.zeros(5, dtype=np.float32)
    return evt_state_features(
        portfolio_returns_window,
        use_auto_threshold=cfg.use_auto_threshold,
    ).astype(np.float32)


def build_state(
    returns_window: np.ndarray,        # (L, N)
    weights: np.ndarray,                # (N,)
    value_normalised: float,
    current_dd: float,
    portfolio_returns_window: np.ndarray,  # (W,)
    cfg: StateConfig,
) -> np.ndarray:
    """Concatenate all blocks into a single 1-D state vector."""
    parts = [
        market_block(returns_window, cfg),
        portfolio_block(weights, value_normalised, current_dd),
    ]
    if cfg.include_evt:
        parts.append(evt_block(portfolio_returns_window, cfg))
    return np.concatenate(parts).astype(np.float32)


def state_dim(n_assets: int, cfg: StateConfig) -> int:
    """Total flat dimension of the state vector for `n_assets` assets."""
    market = n_assets * cfg.window_length
    portfolio = n_assets + 2
    evt = 5 if cfg.include_evt else 0
    return market + portfolio + evt

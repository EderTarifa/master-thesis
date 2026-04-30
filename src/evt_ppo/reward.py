"""
Reward functions for the V0..V4 ablation variants.

V0: PPO-vanilla         — log return only.
V1: PPO-DD              — log return minus quadratic drawdown penalty.
V2: PPO-DD-state        — same reward as V1; EVT enters only via the state.
V3: PPO-DD-evt-reward   — V1 plus EVT-based CVaR penalty on drawdown tail.
V4: PPO-DD-evt-full     — V3 plus EVT in the state.

The functions below compute only the *reward*. Whether EVT enters the
state is controlled by `StateConfig.include_evt` in features.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .evt import GPDFit, fit_gpd, gpd_cvar, select_threshold_auto


@dataclass
class RewardConfig:
    """Configuration of the reward function."""
    variant: str = "V0"           # one of V0..V4
    lambda_dd: float = 1.0        # weight on (DD_{t+1})^2 penalty
    lambda_evt: float = 0.5       # weight on EVT-CVaR-DD penalty
    cvar_alpha: float = 0.95      # alpha used for CVaR-DD
    evt_window: int = 250         # rolling window for EVT-DD estimation
    use_auto_threshold: bool = True
    evt_recompute_every: int = 5  # recalcular cada K steps


    def __post_init__(self) -> None:
        if self.variant not in {"V0", "V1", "V2", "V3", "V4"}:
            raise ValueError(f"Unknown variant: {self.variant}")


def _evt_cvar_drawdown(
    drawdown_history: np.ndarray,
    cfg: RewardConfig,
) -> float:
    """Estimate CVaR_alpha of the drawdown distribution via POT-GPD."""
    dd_history = np.asarray(drawdown_history, dtype=float).ravel()
    # Use only the last `evt_window` observations.
    if dd_history.size > cfg.evt_window:
        dd_history = dd_history[-cfg.evt_window:]
    # Filter to positive drawdowns; if none, return 0.
    positive = dd_history[dd_history > 1e-9]
    if positive.size < 30:
        return 0.0
    # Note: for drawdown the "loss" is the drawdown itself (already positive).
    if cfg.use_auto_threshold:
        _, fit = select_threshold_auto(positive)
    else:
        fit = fit_gpd(positive, quantile=0.80)
    val = gpd_cvar(fit, cfg.cvar_alpha)
    if not np.isfinite(val):
        return 0.0
    # Clip to [0, 1] - drawdowns are bounded.
    return float(np.clip(val, 0.0, 1.0))


def compute_reward(
    log_return: float,
    new_dd: float,
    drawdown_history: np.ndarray,
    cfg: RewardConfig,
    cached_cvar_dd: float | None = None
) -> tuple[float, dict[str, float]]:
    """Compute the per-step reward.

    Parameters
    ----------
    log_return : float
        log(V_{t+1} / V_t) net of transaction costs.
    new_dd : float
        Drawdown at the new step DD_{t+1}.
    drawdown_history : np.ndarray
        Past drawdowns, used by V3/V4 for the EVT-CVaR term. Should
        include DD_{t+1} as the last element.
    cfg : RewardConfig

    Returns
    -------
    reward : float
    components : dict
        Dict with the value of each reward component for logging.
    """
    components: dict[str, float] = {
        "log_return": float(log_return),
        "dd_penalty": 0.0,
        "evt_cvar_penalty": 0.0,
    }

    if cfg.variant == "V0":
        return float(log_return), components

    # V1, V2, V3, V4 all include the quadratic drawdown penalty.
    dd_penalty = cfg.lambda_dd * (new_dd ** 2)
    components["dd_penalty"] = float(dd_penalty)

    if cfg.variant in ("V3", "V4"):
        cvar_dd = cached_cvar_dd if cached_cvar_dd is not None \
                  else _evt_cvar_drawdown(drawdown_history, cfg)

    # V3 and V4: add the EVT-CVaR-DD term.
    cvar_dd = _evt_cvar_drawdown(drawdown_history, cfg)
    evt_penalty = cfg.lambda_evt * cvar_dd
    components["evt_cvar_penalty"] = float(evt_penalty)
    reward = log_return - dd_penalty - evt_penalty
    return float(reward), components

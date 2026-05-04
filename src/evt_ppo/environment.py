"""
Gymnasium environment for long-only portfolio management.

State : float32 vector (see features.build_state).
Action: float32 vector in R^N; converted to portfolio weights on the
        unit simplex via softmax (long-only, fully invested).
Reward: per-step reward as configured (see reward.compute_reward).

Episode dynamics:
    - At step t the agent observes state s_t and chooses raw action a_t.
    - Action is mapped to weights w_t = softmax(a_t).
    - Transaction cost: c * sum |w_t - w_{t-1}| is deducted from the value.
    - Portfolio is held until t+1, when returns r_{t+1} are realised:
        V_{t+1} = V_t * (1 + w_t @ R_{t+1}) - cost
      where R_{t+1} are simple returns (computed from log-returns).
    - Drawdown is updated, reward is computed.

Resetting samples a starting index uniformly within the allowed range,
giving the agent a fresh trajectory.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .drawdown import current_drawdown, maximum_drawdown, underwater_curve
from .features import StateConfig, build_state, state_dim
from .reward import RewardConfig, compute_reward


@dataclass
class EnvConfig:
    """Configuration of the portfolio environment."""
    transaction_cost: float = 0.0010      # 10 bps per unit turnover (one-way)
    initial_value: float = 1.0            # starting portfolio value (normalised)
    max_episode_length: int = 252         # one trading year
    state_cfg: StateConfig = field(default_factory=StateConfig)
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    random_starts: bool = True            # sample episode start uniformly


class PortfolioEnv(gym.Env):
    """Long-only portfolio management environment.

    Parameters
    ----------
    log_returns : np.ndarray, shape (T, N)
        Daily log-returns of the N assets over T trading days.
    cfg : EnvConfig
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        log_returns: np.ndarray,
        cfg: Optional[EnvConfig] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if log_returns.ndim != 2:
            raise ValueError("log_returns must be 2-D (time x assets).")
        self.log_returns = np.asarray(log_returns, dtype=np.float32)
        self.simple_returns = np.expm1(self.log_returns).astype(np.float32)
        self.T, self.N = self.log_returns.shape
        self.cfg = cfg or EnvConfig()

        # Action: raw real-valued logits, mapped to simplex via softmax.
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.N,), dtype=np.float32,
        )

        dim = state_dim(self.N, self.cfg.state_cfg)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._seed_value = seed

        # State variables, initialised in reset().
        self.t = 0
        self.start_t = 0
        self.steps_taken = 0
        self.value: float = self.cfg.initial_value
        self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float32)
        self.value_history: list[float] = []
        self.dd_history: list[float] = []
        self.portfolio_log_returns: deque[float] = deque(
            maxlen=self.cfg.state_cfg.evt_window,
        )

        self._cached_cvar_dd: float = 0.0
        self._cached_step: int = -10**9

    # ---------- gym API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed_value = seed

        L = self.cfg.state_cfg.window_length
        max_start = self.T - self.cfg.max_episode_length - 1
        min_start = L  # need a full window of past returns
        if max_start <= min_start:
            raise ValueError(
                f"Time series too short for episode length "
                f"{self.cfg.max_episode_length} and window {L}."
            )
        if self.cfg.random_starts:
            self.start_t = int(self._rng.integers(min_start, max_start + 1))
        else:
            self.start_t = min_start
        self.t = self.start_t
        self.steps_taken = 0

        # Reset portfolio state.
        self.value = self.cfg.initial_value
        self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float32)
        self.value_history = [self.value]
        self.dd_history = [0.0]
        self.portfolio_log_returns.clear()
        # Pre-load portfolio log-returns history with synthetic equal-weight
        # log returns, so EVT features have something to chew on at t=0.
        # We use the last `evt_window` days of equal-weight returns prior to start_t.
        warmup_start = max(0, self.start_t - self.cfg.state_cfg.evt_window)
        warmup = self.log_returns[warmup_start:self.start_t].mean(axis=1)
        for r in warmup:
            self.portfolio_log_returns.append(float(r))

        self._cached_cvar_dd = 0.0
        self._cached_step = -10**9

        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).ravel()
        if action.shape[0] != self.N:
            raise ValueError(f"Action must have {self.N} elements.")

        # Map raw action to simplex via softmax (long-only, fully invested).
        # We center first to avoid numerical issues with large logits.
        a = action - action.max()
        new_weights = np.exp(a)
        new_weights = new_weights / new_weights.sum()

        # Transaction cost based on absolute change in weights.
        turnover = np.abs(new_weights - self.weights).sum()
        cost = self.cfg.transaction_cost * turnover

        # Realise returns at the next time index.
        prev_value = self.value
        next_simple_returns = self.simple_returns[self.t + 1]
        portfolio_simple_return = float(new_weights @ next_simple_returns)
        # Apply turnover cost as a fractional reduction.
        new_value = prev_value * (1.0 + portfolio_simple_return) * (1.0 - cost)
        # Net log-return (post-cost).
        if new_value > 0 and prev_value > 0:
            net_log_return = float(np.log(new_value / prev_value))
        else:
            net_log_return = -10.0  # severe penalty if value collapses
            new_value = max(new_value, 1e-9)

        self.value = new_value
        self.weights = new_weights
        self.value_history.append(self.value)
        self.portfolio_log_returns.append(net_log_return)

        # Update drawdown.
        new_dd = current_drawdown(np.asarray(self.value_history))
        self.dd_history.append(new_dd)
        
        # Después de actualizar self.dd_history y antes de compute_reward:
        if self.cfg.reward_cfg.variant in ("V3", "V4"):
            if self.steps_taken - self._cached_step >= self.cfg.reward_cfg.evt_recompute_every:
                from .reward import _evt_cvar_drawdown
                self._cached_cvar_dd = _evt_cvar_drawdown(
                    np.asarray(self.dd_history), self.cfg.reward_cfg
                )
                self._cached_step = self.steps_taken

        # Compute reward usando el cache:
        reward, components = compute_reward(
            log_return=net_log_return,
            new_dd=new_dd,
            drawdown_history=np.asarray(self.dd_history),
            cfg=self.cfg.reward_cfg,
            cached_cvar_dd=self._cached_cvar_dd,
        )

        # Advance time.
        self.t += 1
        self.steps_taken += 1
        terminated = False
        truncated = self.steps_taken >= self.cfg.max_episode_length
        truncated = truncated or (self.t >= self.T - 1)

        info = {
            "value": self.value,
            "drawdown": new_dd,
            "weights": self.weights.copy(),
            "turnover": float(turnover),
            "log_return_net": net_log_return,
            **{f"reward/{k}": v for k, v in components.items()},
        }
        return self._observation(), float(reward), terminated, truncated, info

    # ---------- helpers ----------

    def _observation(self) -> np.ndarray:
        L = self.cfg.state_cfg.window_length
        # Past L log-returns of the N assets.
        window = self.log_returns[self.t - L:self.t]  # (L, N)
        port_returns = np.asarray(list(self.portfolio_log_returns), dtype=np.float32)
        return build_state(
            returns_window=window,
            weights=self.weights,
            value_normalised=self.value / self.cfg.initial_value,
            current_dd=self.dd_history[-1] if self.dd_history else 0.0,
            portfolio_returns_window=port_returns,
            cfg=self.cfg.state_cfg,
        )

    # ---------- evaluation accessors ----------

    @property
    def value_array(self) -> np.ndarray:
        return np.asarray(self.value_history, dtype=float)

    @property
    def episode_mdd(self) -> float:
        return maximum_drawdown(self.value_array)

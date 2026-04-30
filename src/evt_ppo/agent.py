"""
PPO agent wrapper around Stable-Baselines3.

The wrapper:
    - Builds a Monitor + DummyVecEnv from a single PortfolioEnv.
    - Configures PPO with hyperparameters tuned for portfolio tasks
      following Karzanov et al. (2025) and FinRL defaults.
    - Provides train(), evaluate(), and a one-shot deterministic rollout
      for backtesting on a test slice (no random_starts).

Notes
-----
- We use MlpPolicy because the state is a flat vector (the market block
  is already flattened over (assets, lags) in features.py).
- For very large state dimensions (e.g. SP50 with L=60: 3000 dims),
  consider switching to a CNN policy that respects the (N, L) structure;
  this is documented as future work.
"""
from __future__ import annotations

import os as _os
_os.environ["OMP_NUM_THREADS"] = "1"
_os.environ["MKL_NUM_THREADS"] = "1"

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Heavy ML imports are optional and only loaded when used. This lets the
# rest of the package (data, evt, drawdown, metrics, statistics, plots)
# work in environments without torch installed.
try:
    import torch  # noqa: F401
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False

from .environment import EnvConfig, PortfolioEnv


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
    })
    verbose: int = 0
    device: str = "cpu"


def _make_env(log_returns: np.ndarray, env_cfg: EnvConfig, seed: int) -> "DummyVecEnv":
    if not _HAS_SB3:
        raise ImportError(
            "Stable-Baselines3 / PyTorch are not installed. "
            "Install with: pip install stable-baselines3 torch"
        )

    def _factory():
        env = PortfolioEnv(log_returns, env_cfg, seed=seed)
        return Monitor(env)

    return DummyVecEnv([_factory])


def train_agent(
    train_log_returns: np.ndarray,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    seed: int,
    save_path: Optional[Path | str] = None,
) -> "PPO":
    """Train a PPO agent on the training slice."""
    if not _HAS_SB3:
        raise ImportError("PPO training requires stable-baselines3 and torch.")

    vec_env = _make_env(train_log_returns, env_cfg, seed=seed)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=ppo_cfg.learning_rate,
        n_steps=ppo_cfg.n_steps,
        batch_size=ppo_cfg.batch_size,
        n_epochs=ppo_cfg.n_epochs,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clip_range=ppo_cfg.clip_range,
        ent_coef=ppo_cfg.ent_coef,
        vf_coef=ppo_cfg.vf_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        policy_kwargs=ppo_cfg.policy_kwargs,
        verbose=ppo_cfg.verbose,
        seed=seed,
        device=ppo_cfg.device
    )
    model.learn(total_timesteps=ppo_cfg.total_timesteps, progress_bar=False)
    if save_path is not None:
        model.save(str(save_path))
    return model


def deterministic_rollout(
    model: "PPO",
    log_returns: np.ndarray,
    env_cfg: EnvConfig,
    start_index: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Run a single deterministic rollout over a test slice.

    Sets random_starts=False and starts at `start_index` (default = first
    valid index = window_length). Returns the value series, weights
    history, and per-step infos.
    """
    if not _HAS_SB3:
        raise ImportError("Rollout requires stable-baselines3 and torch.")

    cfg = EnvConfig(
        transaction_cost=env_cfg.transaction_cost,
        initial_value=env_cfg.initial_value,
        max_episode_length=len(log_returns) - env_cfg.state_cfg.window_length - 2,
        state_cfg=env_cfg.state_cfg,
        reward_cfg=env_cfg.reward_cfg,
        random_starts=False,
    )
    env = PortfolioEnv(log_returns, cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    weights_log: list[np.ndarray] = [env.weights.copy()]
    rewards: list[float] = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        weights_log.append(info["weights"])
        rewards.append(reward)
        if term or trunc:
            break
    return {
        "values": np.asarray(env.value_history, dtype=float),
        "weights_history": np.asarray(weights_log, dtype=float),
        "rewards": np.asarray(rewards, dtype=float),
        "mdd": env.episode_mdd,
    }


def random_rollout(
    log_returns: np.ndarray,
    env_cfg: EnvConfig,
    seed: int = 0,
) -> dict:
    """Sanity-check rollout with uniform random actions; useful for debugging."""
    cfg = EnvConfig(
        transaction_cost=env_cfg.transaction_cost,
        initial_value=env_cfg.initial_value,
        max_episode_length=len(log_returns) - env_cfg.state_cfg.window_length - 2,
        state_cfg=env_cfg.state_cfg,
        reward_cfg=env_cfg.reward_cfg,
        random_starts=False,
    )
    env = PortfolioEnv(log_returns, cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    weights_log = [env.weights.copy()]
    rewards = []
    while True:
        a = rng.normal(0.0, 0.5, size=env.N).astype(np.float32)
        obs, reward, term, trunc, info = env.step(a)
        weights_log.append(info["weights"])
        rewards.append(reward)
        if term or trunc:
            break
    return {
        "values": np.asarray(env.value_history, dtype=float),
        "weights_history": np.asarray(weights_log, dtype=float),
        "rewards": np.asarray(rewards, dtype=float),
        "mdd": env.episode_mdd,
    }

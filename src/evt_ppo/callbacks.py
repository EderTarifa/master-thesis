"""
Telemetry callback for SB3 PPO.

Captures internal training metrics (KL, explained variance, clip fraction,
entropy, gradient norms) per rollout and persists them as a parquet file
alongside the regular runs.parquet.

This is essential for defending convergence of V0 (vanilla PPO) to a Q1 reviewer:
without telemetry there is no way to argue that the baseline was given
a fair chance to converge.

Usage
-----
    from evt_ppo.callbacks import TelemetryCallback
    cb = TelemetryCallback(out_path="logs/telemetry/DJIA_f00_V0_s0.parquet")
    model.learn(total_timesteps=100_000, callback=cb)

The callback captures the same dictionary that SB3 logs to TensorBoard but
persists it to parquet for downstream analysis without TensorBoard.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from stable_baselines3.common.callbacks import BaseCallback


class TelemetryCallback(BaseCallback):
    """Capture PPO internals every rollout and write to parquet at end of training.

    Captured fields per rollout (after each ``train()`` call):
      - timesteps: cumulative env interactions so far
      - approx_kl: Schulman KL approximation (train/approx_kl)
      - explained_variance: 1 - Var(returns - V) / Var(returns)
      - clip_fraction: fraction of samples whose ratio was clipped
      - entropy_loss: -mean(entropy) of the policy
      - policy_gradient_loss: clipped surrogate loss component
      - value_loss: critic MSE
      - loss: total combined loss
      - std: average policy std-dev (Gaussian policy)
      - learning_rate: current LR (in case of schedule)
      - effective_step_norm: L2 norm of (params_after - params_before),
        a robust proxy for effective gradient step size that does not
        require hooks into the optimiser

    The "effective_step_norm" is computed by snapshotting the parameter
    vector before and after the rollout-update cycle, so it captures the
    net change after clipping, LR decay and momentum, which is what a
    convergence diagnostic actually cares about.
    """

    def __init__(self, out_path: str | Path, verbose: int = 0):
        super().__init__(verbose)
        self.out_path = Path(out_path)
        self.records: list[dict] = []
        self._prev_params: np.ndarray | None = None

    def _snapshot_params(self) -> np.ndarray:
        """Flatten all policy parameters into a single numpy vector."""
        return np.concatenate(
            [p.detach().cpu().numpy().ravel()
             for p in self.model.policy.parameters()]
        )

    # ---- BaseCallback hooks ----

    def _on_training_start(self) -> None:
        # Snapshot initial params so the first delta is meaningful.
        self._prev_params = self._snapshot_params()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Called after collecting a rollout AND running train() on it."""
        # SB3 populates self.model.logger.name_to_value during train().
        log = self.model.logger.name_to_value

        current = self._snapshot_params()
        if self._prev_params is not None and current.shape == self._prev_params.shape:
            step_norm = float(np.linalg.norm(current - self._prev_params))
        else:
            step_norm = float("nan")
        self._prev_params = current

        # Pull metrics with safe defaults if SB3 hasn't populated yet.
        def _get(key: str, default: float = float("nan")) -> float:
            v = log.get(key, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        self.records.append({
            "timesteps": int(self.num_timesteps),
            "approx_kl": _get("train/approx_kl"),
            "explained_variance": _get("train/explained_variance"),
            "clip_fraction": _get("train/clip_fraction"),
            "entropy_loss": _get("train/entropy_loss"),
            "policy_gradient_loss": _get("train/policy_gradient_loss"),
            "value_loss": _get("train/value_loss"),
            "loss": _get("train/loss"),
            "std": _get("train/std"),
            "learning_rate": _get("train/learning_rate"),
            "effective_step_norm": step_norm,
        })

    def _on_training_end(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.records)
        df.to_parquet(self.out_path, index=False)
        if self.verbose:
            print(f"[TelemetryCallback] wrote {len(df)} records to {self.out_path}")

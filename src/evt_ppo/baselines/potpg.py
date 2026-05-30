"""
POTPG: Peaks-Over-Threshold Policy Gradient (Davar, Godin & Garrido 2024).

Adapted from the original code at https://github.com/parisadavar/EVT-policy-gradient-RL
(single-asset option-hedging) to multi-asset portfolio allocation on the simplex.

Algorithm (Algorithm 1 of Davar et al. 2024):
  1. Initialise policy theta.
  2. For j = 1..J epochs:
     a. Collect M episodes under pi_theta -> cumulative-cost trajectories L_1..L_M.
     b. Identify the (1-q)*M episodes in the right tail of cost distribution.
     c. Fit GPD to the excesses (L_i - u) where u = q-quantile.
     d. Estimate CVaR_alpha^EVT via the closed-form formula (eq 5-6 of paper).
     e. Compute REINFORCE-style gradient where each tail trajectory contributes
        with weight (L_i - CVaR_alpha) to the score function.
     f. Adam step on theta.

The policy is a Gaussian over logits, projected to the simplex via softmax,
identical in architecture to the V4 PPO policy to make the comparison fair.

GPU
---
The policy network and rollout-aggregation tensors live on `device`. GPD
fitting (scipy.stats.genpareto.fit) stays on CPU because scipy is not
GPU-aware; the fit is done once per batch on the cost scalars (M floats),
not per-step, so the bottleneck is rollout collection, not the fit.

This module returns metrics compatible with `evt_ppo.experiment.run_one_variant`
output schema (mdd, cdar_95, sharpe, calmar, ...) so it slots into the
existing analysis pipeline as a "variant" called POTPG.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..evt import fit_gpd, gpd_cvar, GPDFit, _fit_gpd_mle


@dataclass
class POTPGConfig:
    """Configuration for the POTPG baseline.

    Defaults chosen to match the V4 PPO hyperparameters where applicable, so
    differences in performance can be attributed to the algorithm rather than
    to gratuitous hyperparameter variation.
    """
    total_timesteps: int = 100_000       # match V4 budget
    n_trajectories_per_batch: int = 32   # M in Algorithm 1
    learning_rate: float = 3e-4
    q_threshold: float = 0.90            # POT threshold quantile
    alpha_cvar: float = 0.95             # CVaR confidence level
    hidden_dim: int = 128
    n_hidden_layers: int = 2
    grad_clip: float = 0.5
    min_excesses_for_gpd: int = 5        # below this, fall back to empirical CVaR
    device: str = "cuda"                 # "cuda" or "cpu"
    verbose: int = 0


class POTPGPolicy(nn.Module):
    """Gaussian policy with softmax projection to the simplex.

    Architecture mirrors V4 PPO policy: MLP [hidden, hidden] ReLU, with a
    state-dependent mean head and a state-independent learnable log_std.
    The output is reparameterised (rsample) so the log-prob is
    differentiable through the action sample.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, n_hidden: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for _ in range(n_hidden):
            layers.extend([nn.Linear(prev, hidden_dim), nn.ReLU()])
            prev = hidden_dim
        self.shared = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # State-independent log_std (standard PPO/SAC convention).
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(state)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def sample_action(self, state: torch.Tensor, deterministic: bool = False
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample logits to feed PortfolioEnv (which does softmax internally).

        Returns (logits_raw, log_prob).
        IMPORTANT: PortfolioEnv expects Box(-5, 5) raw logits and performs
        the softmax projection itself. We therefore output logits, NOT
        softmaxed weights, to avoid double-softmaxing.

        The logits are clamped to [-5, 5] to respect the action_space
        bounds; this is done outside the gradient path (clamp is
        differentiable with zero gradient at the boundary, which is OK).
        """
        mean, std = self.forward(state)
        if deterministic:
            logits = mean
            log_prob = torch.zeros(mean.size(0), device=mean.device)
        else:
            # Reparameterised Gaussian sample.
            eps = torch.randn_like(mean)
            logits = mean + std * eps
            # log_prob is sum over action_dim of independent Gaussian logprobs.
            log_prob = (
                -0.5 * ((logits - mean) / std).pow(2)
                - std.log()
                - 0.5 * np.log(2 * np.pi)
            ).sum(dim=-1)
        # Clamp to action_space bounds (-5, 5) to avoid env warnings.
        # Gradient is preserved in the interior; zero at the boundary.
        logits_clamped = torch.clamp(logits, -5.0, 5.0)
        return logits_clamped, log_prob


class POTPGTrainer:
    """
    Train a POTPG agent on a vectorised environment.

    Public surface (compatible with run_one_variant):
      .learn(total_timesteps)
      .deterministic_rollout(test_log_returns, env_cfg, seed) -> dict[...]

    The trainer expects a callable `env_fn` returning a fresh env instance
    compatible with your PortfolioEnv (Gymnasium API: reset, step).
    """
    def __init__(self, env_fn, state_dim: int, action_dim: int,
                 cfg: POTPGConfig | None = None, seed: int = 0):
        self.cfg = cfg or POTPGConfig()
        self.env_fn = env_fn
        self.device = torch.device(self.cfg.device if torch.cuda.is_available()
                                   else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)

        self.policy = POTPGPolicy(
            state_dim, action_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_hidden=self.cfg.n_hidden_layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.cfg.learning_rate
        )

        # Telemetry, returned at end of training.
        self.history: list[dict] = []

    # ---- Rollout collection ----

    def _collect_trajectory(self, seed: int | None = None
                            ) -> tuple[torch.Tensor, float, int]:
        """Run one episode, return (sum_log_probs, terminal_loss, length).

        terminal_loss is defined as the *negative* cumulative reward, i.e.
        the loss to minimise. For portfolio drawdown reward (V4 reward
        equation), -sum(r_t) is approximately the total drawdown + cost,
        which is what we want POTPG's CVaR to act on.

        Actions are raw logits in Box(-5, 5); PortfolioEnv applies softmax
        internally before computing weights and the next return.
        """
        env = self.env_fn()
        obs, _ = env.reset(seed=seed)
        sum_log_prob = torch.zeros(1, device=self.device)
        cumulative_reward = 0.0
        steps = 0
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, log_prob = self.policy.sample_action(obs_t, deterministic=False)
            logits_np = logits.detach().cpu().numpy().squeeze(0)
            obs, reward, terminated, truncated, _ = env.step(logits_np)
            cumulative_reward += float(reward)
            sum_log_prob = sum_log_prob + log_prob
            done = bool(terminated or truncated)
            steps += 1
        env.close()
        terminal_loss = -cumulative_reward  # POTPG minimises loss
        return sum_log_prob.squeeze(0), terminal_loss, steps

    def _compute_cvar_evt_loss(self, log_probs: torch.Tensor,
                               losses_np: np.ndarray) -> tuple[torch.Tensor, float, dict]:
        """
        Compute the differentiable POTPG objective.

        Mathematically: J(theta) = E_pi[L * I{L > VaR_alpha}] / (1 - alpha),
        where L is the cumulative loss and the indicator is replaced by the
        EVT extrapolation of the GPD fit above the threshold u = q-quantile.

        The REINFORCE-style gradient estimator weights each trajectory's
        score function (log pi) by its centred loss contribution to CVaR.

        Numerical stability note
        ------------------------
        Cumulative losses across 252-step episodes can range in the
        tens-to-hundreds (V4 reward has lambda_dd=2, lambda_evt=2, so
        per-step penalties dominate log-return magnitudes). At those
        absolute scales, scipy's MLE optimiser for GPD with default
        initialisation fails to converge (gpd_shape stuck at 0, gpd_scale
        at 1e-3). We therefore:
          1. Rescale losses by their median absolute deviation before
             fitting so excesses sit in a unit-ish range.
          2. Fit GPD on the rescaled excesses.
          3. Transform the resulting CVaR back to the original scale.

        Equivalent to fitting GPD(sigma * s, xi) where s is the scaling
        factor. Shape parameter xi is scale-invariant; only sigma is
        rescaled, which we do analytically.
        """
        u = float(np.quantile(losses_np, self.cfg.q_threshold))
        excess_mask = losses_np > u
        n_exc = int(excess_mask.sum())

        diagnostics = {
            "threshold_u": u,
            "n_excesses": n_exc,
            "mean_loss": float(np.mean(losses_np)),
            "max_loss": float(np.max(losses_np)),
        }

        if n_exc < self.cfg.min_excesses_for_gpd:
            # Not enough tail samples for GPD: fall back to empirical CVaR
            # over the top (1-alpha) fraction (vanilla REINFORCE-CVaR).
            k = max(int(round((1 - self.cfg.alpha_cvar) * len(losses_np))), 1)
            top_idx = np.argsort(losses_np)[-k:]
            top_losses = torch.as_tensor(losses_np[top_idx], dtype=torch.float32,
                                         device=self.device)
            cvar_empirical = float(top_losses.mean().item())
            # REINFORCE: minimise CVaR => grad = E[L_tail * grad log pi]
            loss_tensor = (log_probs[top_idx] * (top_losses - top_losses.mean()).detach()
                           ).mean()
            diagnostics.update({
                "cvar_estimate": cvar_empirical,
                "fit_mode": "empirical",
                "gpd_shape": float("nan"),
                "gpd_scale": float("nan"),
                "gpd_converged": False,
                "scale_factor": float("nan"),
            })
            return loss_tensor, cvar_empirical, diagnostics

        # Fit GPD on rescaled excesses for numerical stability.
        # Scale factor = MAD of all losses (robust to outliers); ensures
        # rescaled excesses are O(1), where scipy's MLE is well-behaved.
        scale_factor = float(np.median(np.abs(losses_np - np.median(losses_np))))
        if scale_factor < 1e-8:
            # Degenerate: all losses identical. Fall back to empirical.
            cvar_empirical = float(np.mean(losses_np[excess_mask]))
            loss_tensor = (log_probs[excess_mask] * 0.0).mean()  # zero gradient
            diagnostics.update({
                "cvar_estimate": cvar_empirical,
                "fit_mode": "empirical",
                "gpd_shape": float("nan"),
                "gpd_scale": float("nan"),
                "gpd_converged": False,
                "scale_factor": scale_factor,
            })
            return loss_tensor, cvar_empirical, diagnostics

        excesses_scaled = (losses_np[excess_mask] - u) / scale_factor
        # Call _fit_gpd_mle directly. fit_gpd() requires n_total >= 20 (a
        # guard sensible for time-series windows of 250 days), but POTPG
        # only has M=64 trajectories so it would always trip that guard
        # and return a degenerate placeholder. _fit_gpd_mle requires only
        # n >= 5, which is the actual statistical limit for MLE.
        xi_fit, sigma_fit_scaled, converged = _fit_gpd_mle(excesses_scaled)
        # Shape xi is scale-invariant; sigma scales linearly. Transform back:
        # if Y = X/s, GPD(sigma_Y, xi) then X ~ GPD(sigma_Y * s, xi).
        sigma_original = sigma_fit_scaled * scale_factor
        full_fit = GPDFit(
            threshold=u,
            shape=xi_fit,
            scale=sigma_original,
            n_excesses=n_exc,
            n_total=len(losses_np),
            converged=converged,
        )
        cvar_evt = gpd_cvar(full_fit, self.cfg.alpha_cvar)
        if not np.isfinite(cvar_evt):
            cvar_evt = float(np.mean(losses_np[excess_mask]))

        # REINFORCE-style gradient with EVT-CVaR as advantage.
        # Each tail trajectory has weight (L_i - CVaR_alpha^EVT).
        excess_losses_t = torch.as_tensor(losses_np[excess_mask],
                                          dtype=torch.float32, device=self.device)
        cvar_t = torch.as_tensor(cvar_evt, dtype=torch.float32, device=self.device)
        advantages = (excess_losses_t - cvar_t).detach()  # gradient flows ONLY via log_probs
        excess_log_probs = log_probs[excess_mask]
        loss_tensor = (excess_log_probs * advantages).mean()

        diagnostics.update({
            "cvar_estimate": float(cvar_evt),
            "fit_mode": "gpd",
            "gpd_shape": float(full_fit.shape),
            "gpd_scale": float(full_fit.scale),
            "gpd_converged": bool(full_fit.converged),
            "scale_factor": scale_factor,
        })
        return loss_tensor, float(cvar_evt), diagnostics

    # ---- Public API ----

    def learn(self, total_timesteps: int | None = None,
              telemetry_path: str | None = None) -> "POTPGTrainer":
        """Train the policy for approximately `total_timesteps` env interactions.

        Parameters
        ----------
        total_timesteps : int, optional
            Override config budget.
        telemetry_path : str, optional
            If set, writes per-iteration diagnostics to this parquet on
            training end. Used to defend convergence to a Q1 reviewer.
            Recorded fields:
              - iteration, timesteps
              - mean_loss, max_loss, threshold_u, n_excesses
              - cvar_estimate, fit_mode (gpd|empirical)
              - gpd_shape, gpd_scale, gpd_converged
              - objective (the POTPG loss tensor value)
        """
        budget = total_timesteps if total_timesteps is not None else self.cfg.total_timesteps
        cumulative_steps = 0
        iteration = 0
        while cumulative_steps < budget:
            # Collect a batch of trajectories.
            log_probs_list: list[torch.Tensor] = []
            losses_list: list[float] = []
            iter_steps = 0
            for m in range(self.cfg.n_trajectories_per_batch):
                lp, term_loss, steps = self._collect_trajectory(
                    seed=iteration * self.cfg.n_trajectories_per_batch + m
                )
                log_probs_list.append(lp)
                losses_list.append(term_loss)
                iter_steps += steps
            cumulative_steps += iter_steps

            log_probs = torch.stack(log_probs_list)  # [M]
            losses_np = np.asarray(losses_list, dtype=np.float64)

            # Compute POTPG loss.
            loss_tensor, cvar_value, diag = self._compute_cvar_evt_loss(
                log_probs, losses_np
            )

            self.optimizer.zero_grad()
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.grad_clip
            )
            self.optimizer.step()

            self.history.append({
                "iteration": iteration,
                "timesteps": cumulative_steps,
                "objective": float(loss_tensor.item()),
                **diag,
            })
            if self.cfg.verbose and iteration % 5 == 0:
                print(f"[POTPG it={iteration:3d} ts={cumulative_steps:6d}] "
                      f"mean_loss={diag['mean_loss']:.4f} "
                      f"CVaR^EVT={cvar_value:.4f} "
                      f"n_exc={diag['n_excesses']}")
            iteration += 1

        # Persist telemetry if requested.
        if telemetry_path is not None and self.history:
            from pathlib import Path
            tp = Path(telemetry_path)
            tp.parent.mkdir(parents=True, exist_ok=True)
            import pandas as pd
            pd.DataFrame(self.history).to_parquet(tp, index=False)

        return self

    @torch.no_grad()
    def predict(self, observation: np.ndarray, deterministic: bool = True
                ) -> tuple[np.ndarray, None]:
        """SB3-compatible predict() for use in deterministic_rollout.

        Returns raw logits in Box(-5, 5); PortfolioEnv applies softmax.
        """
        obs_t = torch.as_tensor(observation, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        logits, _ = self.policy.sample_action(obs_t, deterministic=deterministic)
        return logits.cpu().numpy().squeeze(0), None
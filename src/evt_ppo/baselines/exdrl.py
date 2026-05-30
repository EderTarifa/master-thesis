"""
EX-DRL: EXtreme Distributional RL (Malekzadeh et al. 2024).

Adapted from the public reference implementation at
https://github.com/pmalekzadeh/EX-DRL (which itself wraps QR-D4PG with a
GPD tail head). The original code is built on Acme/JAX for option hedging;
this re-implementation is a self-contained PyTorch port that:

  - Replaces the Acme/JAX backend with a vanilla PyTorch off-policy
    actor-critic (TD3-style with deterministic policy + target networks).
  - Substitutes the option-hedging action space with the long-only simplex
    of portfolio allocation (softmax projection).
  - Keeps the EX-DRL key innovation intact: the critic predicts N atoms of
    the return distribution via quantile regression, and the *tail* (top
    quantiles above a threshold) is additionally constrained to match a
    fitted GPD via a soft consistency penalty in the critic loss.

EVAC vs EX-DRL: both share the "bulk quantiles + GPD tail" architecture.
The chosen reference here is EX-DRL because (a) it has a verifiable public
codebase, (b) it is also cited in our paper, and (c) the methodological
contribution that matters for the comparison (GPD-augmented distributional
critic) is identical in both.

GPU
---
This implementation runs end-to-end on GPU. Replay buffer is held on CPU
in numpy (cheap) and minibatches are moved to GPU per gradient step. The
GPD fit (the only non-differentiable step) happens on CPU on scalar
arrays of size = batch_size; negligible overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..evt import fit_gpd, gpd_cvar, gpd_var, GPDFit


@dataclass
class EXDRLConfig:
    """EX-DRL hyperparameters.

    Defaults follow the public EX-DRL repo (run_configs/agents/d4pg.cfg) where
    sensible, adapted to the budget of V4 PPO (100k env steps) for fair
    comparison.
    """
    total_timesteps: int = 100_000
    n_quantiles: int = 51                # N atoms in the distributional critic
    tail_threshold_q: float = 0.90       # quantile defining where the "tail" head kicks in
    gpd_consistency_weight: float = 0.5  # weight of GPD-tail consistency penalty
    gpd_refit_every: int = 200           # refit GPD to recent returns every K steps
    alpha_cvar_objective: float = 0.95   # CVaR level the actor minimises
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005                   # soft target update rate
    batch_size: int = 256
    buffer_size: int = 100_000
    warmup_steps: int = 1_000
    hidden_dim: int = 128
    n_hidden_layers: int = 2
    action_noise_std: float = 0.1        # exploration noise on actor output (TD3-style)
    grad_clip: float = 1.0
    device: str = "cuda"
    verbose: int = 0


class _MLP(nn.Module):
    """Generic MLP with ReLU activations, used by both actor and critic."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128,
                 n_hidden: int = 2, out_activation: Optional[nn.Module] = None):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden):
            layers.extend([nn.Linear(prev, hidden_dim), nn.ReLU()])
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        if out_activation is not None:
            layers.append(out_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EXDRLActor(nn.Module):
    """Deterministic actor outputting raw logits (Box(-5, 5)).

    IMPORTANT: PortfolioEnv expects Box(-5, 5) raw logits and performs
    softmax internally to project onto the simplex. We therefore use
    `tanh * 5.0` as the output activation so the actor naturally lives
    in the env's action space and can produce any extreme of the simplex
    via large positive/negative logits.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, n_hidden: int = 2):
        super().__init__()
        self.body = _MLP(state_dim, action_dim, hidden_dim, n_hidden)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.body(state)
        # Tanh-scaled output to stay inside Box(-5, 5) of PortfolioEnv.
        logits = 5.0 * torch.tanh(raw)
        return logits


class EXDRLDistributionalCritic(nn.Module):
    """Critic outputting N quantiles of Z(s, a) = return distribution.

    Trained via quantile Huber loss (QR-DQN style) to learn the return
    distribution. The top quantiles are additionally constrained to be
    consistent with a fitted GPD via the consistency penalty in the
    trainer's critic_loss.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 n_quantiles: int = 51, hidden_dim: int = 128, n_hidden: int = 2):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.body = _MLP(state_dim + action_dim, n_quantiles, hidden_dim, n_hidden)
        # midpoint quantiles used by the QR-Huber loss
        taus = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
        self.register_buffer("taus", taus)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.body(sa)  # [B, N]


class _ReplayBuffer:
    """Plain CPU replay buffer; batches are moved to device per sample."""
    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            float(r),
            np.asarray(s_next, dtype=np.float32),
            bool(done),
        ))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        s, a, r, s_next, done = [], [], [], [], []
        for i in idx:
            e = self.buf[i]
            s.append(e[0]); a.append(e[1]); r.append(e[2])
            s_next.append(e[3]); done.append(e[4])
        return (
            np.stack(s), np.stack(a),
            np.asarray(r, dtype=np.float32),
            np.stack(s_next),
            np.asarray(done, dtype=np.float32),
        )


def _quantile_huber_loss(q_pred: torch.Tensor, target: torch.Tensor,
                         taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Standard QR-DQN Huber loss.

    q_pred: [B, N] predicted quantiles for the current (s, a).
    target: [B, N] target quantiles (r + gamma * Z(s', a*)).
    taus:   [N]    midpoint quantile levels in (0, 1).
    """
    # td_errors[i, j, k] = target[i, k] - q_pred[i, j]
    diff = target.unsqueeze(1) - q_pred.unsqueeze(2)  # [B, N_pred, N_target]
    huber = torch.where(
        diff.abs() <= kappa,
        0.5 * diff.pow(2),
        kappa * (diff.abs() - 0.5 * kappa),
    )
    # tau-weighted asymmetric scaling: |tau - I{diff<0}| * huber
    taus_expanded = taus.view(1, -1, 1)
    weight = (taus_expanded - (diff < 0).float()).abs()
    loss = (weight * huber).sum(dim=1).mean()
    return loss


class EXDRLTrainer:
    """Off-policy actor-critic with distributional critic and GPD tail head.

    Public API mirrors what `run_one_variant` expects:
      .learn(total_timesteps)
      .predict(obs, deterministic) -> (action, None)
    """
    def __init__(self, env_fn, state_dim: int, action_dim: int,
                 cfg: EXDRLConfig | None = None, seed: int = 0):
        self.cfg = cfg or EXDRLConfig()
        self.env_fn = env_fn

        self.device = torch.device(self.cfg.device if torch.cuda.is_available()
                                   else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)

        self.actor = EXDRLActor(state_dim, action_dim,
                                self.cfg.hidden_dim, self.cfg.n_hidden_layers
                                ).to(self.device)
        self.actor_target = EXDRLActor(state_dim, action_dim,
                                       self.cfg.hidden_dim, self.cfg.n_hidden_layers
                                       ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = EXDRLDistributionalCritic(
            state_dim, action_dim, self.cfg.n_quantiles,
            self.cfg.hidden_dim, self.cfg.n_hidden_layers,
        ).to(self.device)
        self.critic_target = EXDRLDistributionalCritic(
            state_dim, action_dim, self.cfg.n_quantiles,
            self.cfg.hidden_dim, self.cfg.n_hidden_layers,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(),
                                          lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=self.cfg.critic_lr)

        self.buffer = _ReplayBuffer(self.cfg.buffer_size)
        self.history: list[dict] = []

        # GPD state: refreshed every cfg.gpd_refit_every steps.
        self._gpd_xi: float = 0.0
        self._gpd_sigma: float = 1e-3
        self._gpd_u: float = 0.0
        self._gpd_Fu: float = 0.05  # exceedance probability

    # ---- GPD fit helper ----

    def _refit_gpd(self):
        """Fit GPD to the tail of recent target-quantile values.

        Uses the empirical distribution of cumulative returns from the
        replay buffer. Operates on the *loss* side (-reward) to be
        consistent with the EVT convention used elsewhere in evt_ppo.
        """
        if len(self.buffer) < 200:
            return
        # Sample a large chunk of returns from the buffer (cheap).
        rewards = np.asarray([e[2] for e in self.buffer.buf], dtype=np.float64)
        losses = -rewards
        u = float(np.quantile(losses, self.cfg.tail_threshold_q))
        excesses = losses[losses > u] - u
        if excesses.size < 30:
            return
        fit = fit_gpd(excesses, quantile=0.0)
        self._gpd_xi = float(fit.shape)
        self._gpd_sigma = float(fit.scale)
        self._gpd_u = u
        self._gpd_Fu = float((losses > u).mean())

    # ---- Actor objective: minimise CVaR of Z(s, pi(s)) ----

    def _actor_objective(self, states: torch.Tensor) -> torch.Tensor:
        """
        Actor minimises CVaR_alpha of the predicted return distribution.

        With the distributional critic Z(s, a) outputting N quantiles, the
        CVaR_alpha is the mean of the top (1-alpha) quantiles. Because we
        want to *maximise* the CVaR of returns (= minimise CVaR of losses),
        the actor loss is -CVaR_alpha(Z).
        """
        actions = self.actor(states)
        q_values = self.critic(states, actions)  # [B, N]
        # CVaR_alpha = mean of bottom (1-alpha) quantiles for the loss view,
        # equivalently mean of top alpha quantiles for the return view.
        # The return-view CVaR is what we want to MAXIMISE.
        q_sorted, _ = torch.sort(q_values, dim=-1)
        n_tail = max(int(round((1.0 - self.cfg.alpha_cvar_objective)
                               * self.cfg.n_quantiles)), 1)
        # Worst-case quantiles in return space = lowest quantiles.
        worst_quantiles = q_sorted[:, :n_tail]
        cvar_returns = worst_quantiles.mean(dim=-1)
        return -cvar_returns.mean()  # maximise CVaR of returns

    # ---- Critic objective: QR-Huber + GPD consistency on the tail ----

    def _critic_loss(self, batch_t) -> tuple[torch.Tensor, dict]:
        s, a, r, s_next, done = batch_t
        with torch.no_grad():
            next_action = self.actor_target(s_next)
            target_quantiles = self.critic_target(s_next, next_action)
            target = r.unsqueeze(-1) + self.cfg.gamma * (1.0 - done.unsqueeze(-1)) * target_quantiles

        q_pred = self.critic(s, a)  # [B, N]
        qr_loss = _quantile_huber_loss(q_pred, target, self.critic.taus)

        # ---- GPD consistency penalty on the tail ----
        # Push the top quantiles of q_pred to match the GPD extrapolation
        # of return losses. This is the EX-DRL "tail head" innovation.
        consistency = torch.tensor(0.0, device=s.device)
        if self._gpd_Fu > 0 and abs(self._gpd_xi) < 1.0:
            # For each of the top tail quantiles, compute the expected GPD value.
            # We work in the "loss" frame: predicted return -> loss = -return.
            losses_pred, _ = torch.sort(-q_pred, dim=-1, descending=True)  # [B, N] descending
            n_tail = max(int(round((1.0 - self.cfg.tail_threshold_q)
                                   * self.cfg.n_quantiles)), 1)
            tail_losses = losses_pred[:, :n_tail]
            # Build a small target by querying the GPD VaR at the same tau levels.
            taus_tail = self.critic.taus[-n_tail:]  # the top quantile levels in (0,1)
            taus_tail_np = taus_tail.detach().cpu().numpy()
            with torch.no_grad():
                # construct an ad-hoc GPDFit object
                fit = GPDFit(
                    threshold=self._gpd_u,
                    shape=self._gpd_xi,
                    scale=self._gpd_sigma,
                    n_excesses=max(int(self._gpd_Fu * 1000), 30),
                    n_total=1000,
                    converged=True,
                )
                gpd_targets = np.asarray(
                    [gpd_var(fit, float(t)) for t in taus_tail_np],
                    dtype=np.float32,
                )
            gpd_targets_t = torch.as_tensor(gpd_targets, device=s.device).unsqueeze(0)
            consistency = (tail_losses - gpd_targets_t).pow(2).mean()

        total = qr_loss + self.cfg.gpd_consistency_weight * consistency
        diag = {
            "qr_loss": float(qr_loss.item()),
            "gpd_consistency": float(consistency.item()),
            "gpd_xi": self._gpd_xi,
            "gpd_sigma": self._gpd_sigma,
        }
        return total, diag

    # ---- Public API ----

    def learn(self, total_timesteps: int | None = None,
              telemetry_path: str | None = None) -> "EXDRLTrainer":
        """Train EX-DRL for ~total_timesteps environment steps.

        Parameters
        ----------
        total_timesteps : int, optional
            Override config budget.
        telemetry_path : str, optional
            If set, writes per-1000-step diagnostics to parquet on training end.
            Fields:
              - step, critic_loss, actor_loss
              - qr_loss, gpd_consistency, gpd_xi, gpd_sigma
        """
        budget = total_timesteps if total_timesteps is not None else self.cfg.total_timesteps
        env = self.env_fn()
        obs, _ = env.reset()
        step = 0
        episode_return = 0.0
        episode_steps = 0

        while step < budget:
            # ---- act ----
            # Action space is Box(-5, 5) raw logits; PortfolioEnv softmaxes internally.
            if step < self.cfg.warmup_steps:
                # Uniform exploration in the raw-logit action space.
                action = np.random.uniform(
                    low=-5.0, high=5.0, size=env.action_space.shape
                ).astype(np.float32)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                with torch.no_grad():
                    a_t = self.actor(obs_t).squeeze(0)
                action = a_t.cpu().numpy()
                # TD3-style Gaussian exploration noise.
                if self.cfg.action_noise_std > 0:
                    action = action + np.random.normal(
                        0.0, self.cfg.action_noise_std * 5.0,  # scale noise to logit range
                        size=action.shape
                    ).astype(np.float32)
                # Clip to action space bounds.
                action = np.clip(action, -5.0, 5.0).astype(np.float32)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            self.buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += float(reward)
            episode_steps += 1
            step += 1
            if done:
                obs, _ = env.reset()
                episode_return = 0.0
                episode_steps = 0

            # ---- learn ----
            if len(self.buffer) >= max(self.cfg.batch_size, self.cfg.warmup_steps):
                if step % self.cfg.gpd_refit_every == 0:
                    self._refit_gpd()

                s_np, a_np, r_np, sn_np, d_np = self.buffer.sample(self.cfg.batch_size)
                s = torch.as_tensor(s_np, device=self.device)
                a = torch.as_tensor(a_np, device=self.device)
                r = torch.as_tensor(r_np, device=self.device)
                sn = torch.as_tensor(sn_np, device=self.device)
                d = torch.as_tensor(d_np, device=self.device)

                # Critic update
                critic_loss, diag = self._critic_loss((s, a, r, sn, d))
                self.critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.cfg.grad_clip
                )
                self.critic_opt.step()

                # Actor update
                actor_loss = self._actor_objective(s)
                self.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.cfg.grad_clip
                )
                self.actor_opt.step()

                # Soft target updates
                with torch.no_grad():
                    for p, p_t in zip(self.actor.parameters(),
                                      self.actor_target.parameters()):
                        p_t.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
                    for p, p_t in zip(self.critic.parameters(),
                                      self.critic_target.parameters()):
                        p_t.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

                if step % 1000 == 0:
                    self.history.append({
                        "step": step,
                        "critic_loss": float(critic_loss.item()),
                        "actor_loss": float(actor_loss.item()),
                        **diag,
                    })
                    if self.cfg.verbose:
                        print(f"[EXDRL step={step}] critic={critic_loss.item():.4f} "
                              f"actor={actor_loss.item():.4f} xi={self._gpd_xi:.3f}")
        env.close()

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
        """SB3-compatible predict() for use in deterministic_rollout."""
        obs_t = torch.as_tensor(observation, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        action = self.actor(obs_t).squeeze(0).cpu().numpy()
        if not deterministic and self.cfg.action_noise_std > 0:
            action = action + np.random.normal(
                0.0, self.cfg.action_noise_std * 5.0, size=action.shape
            ).astype(np.float32)
        action = np.clip(action, -5.0, 5.0).astype(np.float32)
        return action, None

"""Train + evaluate a single (market, fold, baseline, seed) and append to a parquet.

Analogous to scripts/run_one.py but invokes one of the new baseline trainers
instead of SB3 PPO. The output schema (parquet row) matches `runs.parquet`
exactly, so downstream analysis (paired_table, aggregate_runs, the existing
plots) treats the baseline as just another "variant".

Usage
-----
    python scripts/run_one_baseline.py \\
        --market DJIA --fold 0 --baseline POTPG --seed 0 \\
        --config src/configs/baselines.yaml \\
        --out results/baselines_optimal
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Threading caps. Even though baselines use GPU for compute, the env loop
# (PortfolioEnv.step) is pure numpy on CPU and benefits from single-thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

import yaml

from evt_ppo.data import load_local, prices_to_log_returns
from evt_ppo.environment import EnvConfig, PortfolioEnv
from evt_ppo.features import StateConfig, state_dim
from evt_ppo.metrics import all_metrics
from evt_ppo.reward import RewardConfig
from evt_ppo.walkforward import WalkForwardConfig, generate_splits

# Baseline trainers
from evt_ppo.baselines.potpg import POTPGTrainer, POTPGConfig
from evt_ppo.baselines.exdrl import EXDRLTrainer, EXDRLConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant presets for baselines.
#
# The PPO V0..V4 ablation uses a quadratic drawdown penalty + optional EVT
# state features + optional EVT reward regulariser. The baselines below
# compete with V4 (the full proposal) under the same reward signal, so each
# baseline is trained against the V4 reward configuration to make the
# comparison head-to-head: same MDP, same reward, different algorithm.
# ---------------------------------------------------------------------------
BASELINE_REWARD = RewardConfig(
    variant="V4", lambda_dd=2.0, lambda_evt=2.0,  # match full_optimal V4
    evt_window=250, cvar_alpha=0.95,
)

BASELINE_STATE_INCLUDES_EVT = True  # match V4


# ---------------------------------------------------------------------------
# Evaluation routine: deterministic rollout over test slice.
# Mirrors evt_ppo.agent.deterministic_rollout but works with our custom
# baseline trainers' .predict() method.
# ---------------------------------------------------------------------------


def deterministic_rollout_baseline(model, log_returns: np.ndarray,
                                   env_cfg: EnvConfig, seed: int = 0) -> dict:
    """Run one deterministic rollout on the test slice, return values + weights."""
    cfg_eval = EnvConfig(
        transaction_cost=env_cfg.transaction_cost,
        initial_value=env_cfg.initial_value,
        max_episode_length=len(log_returns) - env_cfg.state_cfg.window_length - 2,
        state_cfg=env_cfg.state_cfg,
        reward_cfg=env_cfg.reward_cfg,
        random_starts=False,
    )
    env = PortfolioEnv(log_returns, cfg_eval, seed=seed)
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


# ---------------------------------------------------------------------------
# Baseline factory.
# ---------------------------------------------------------------------------


def build_baseline(name: str, env_fn, state_dim_: int, action_dim: int,
                   cfg_dict: dict, seed: int):
    """Construct the appropriate baseline trainer."""
    if name == "POTPG":
        cfg = POTPGConfig(**cfg_dict)
        return POTPGTrainer(env_fn=env_fn, state_dim=state_dim_,
                            action_dim=action_dim, cfg=cfg, seed=seed)
    if name == "EXDRL":
        cfg = EXDRLConfig(**cfg_dict)
        return EXDRLTrainer(env_fn=env_fn, state_dim=state_dim_,
                            action_dim=action_dim, cfg=cfg, seed=seed)
    raise ValueError(f"Unknown baseline: {name}. Available: POTPG, EXDRL.")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--baseline", required=True, choices=["POTPG", "EXDRL"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", default="src/configs/baselines.yaml")
    p.add_argument("--out", default="results/baselines_optimal")
    args = p.parse_args()

    out_file = Path(args.out) / "rows" / (
        f"{args.market}_f{args.fold:02d}_{args.baseline}_s{args.seed}.parquet"
    )
    if out_file.exists():
        log.info("SKIP: already exists -> %s", out_file)
        return

    with open(args.config) as f:
        d = yaml.safe_load(f)

    wf = WalkForwardConfig(**d["walkforward"])
    baseline_cfg = d["baselines"][args.baseline]

    # Match V4's state + reward so comparison is head-to-head.
    state_cfg = StateConfig(
        window_length=d["state_window_length"],
        evt_window=d["evt_window"],
        include_evt=BASELINE_STATE_INCLUDES_EVT,
        normalise_market=True,
        use_auto_threshold=True,
    )
    env_cfg = EnvConfig(
        transaction_cost=d["transaction_cost"],
        max_episode_length=d["max_episode_length"],
        state_cfg=state_cfg,
        reward_cfg=BASELINE_REWARD,
        random_starts=True,
    )

    # Load fold data (identical logic to run_one.py).
    prices = load_local(f"data/{args.market}.parquet")
    log_ret_df = prices_to_log_returns(prices).dropna(how="any")
    splits = generate_splits(log_ret_df.index, wf)
    if args.fold >= len(splits):
        log.info("SKIP: fold %d out of range (%d available)", args.fold, len(splits))
        return
    split = splits[args.fold]
    train_lr = log_ret_df.loc[split.train_start:split.val_end].values
    test_lr = log_ret_df.loc[split.test_start:split.test_end].values

    action_dim = train_lr.shape[1]
    obs_dim = state_dim(action_dim, state_cfg)

    log.info("Building %s on %s fold %d seed %d (state=%d, action=%d)",
             args.baseline, args.market, args.fold, args.seed, obs_dim, action_dim)

    env_fn = lambda: PortfolioEnv(train_lr, env_cfg, seed=args.seed)
    model = build_baseline(args.baseline, env_fn, obs_dim, action_dim,
                           baseline_cfg, seed=args.seed)

    # Telemetry path: one parquet per (market, fold, baseline, seed).
    telemetry_path = Path(args.out) / "telemetry" / (
        f"{args.market}_f{args.fold:02d}_{args.baseline}_s{args.seed}.parquet"
    )

    t0 = time.time()
    model.learn(
        total_timesteps=baseline_cfg["total_timesteps"],
        telemetry_path=str(telemetry_path),
    )
    train_time = time.time() - t0
    log.info("%s training done in %.1fs", args.baseline, train_time)

    # Evaluate.
    rollout = deterministic_rollout_baseline(model, test_lr, env_cfg, seed=args.seed)
    metrics = all_metrics(rollout["values"], rollout["weights_history"])

    row = {
        "market": args.market,
        "fold": args.fold,
        "test_start": split.test_start,
        "test_end": split.test_end,
        "variant": args.baseline,  # so it shows up as POTPG / EXDRL in runs.parquet
        "seed": args.seed,
        "train_time_seconds": train_time,
        **metrics,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(out_file)
    log.info("Wrote %s", out_file)


if __name__ == "__main__":
    main()

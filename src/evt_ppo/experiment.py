"""
Experiment orchestration.

Runs the full (market, fold, seed, variant) grid:

    for market in markets:
        for fold in walkforward_splits(market):
            for seed in seeds:
                for variant in variants:
                    train PPO on fold.train
                    evaluate on fold.test
                    record metrics
        run benchmarks for that fold

Outputs are persisted incrementally to results/<run_name>/:
    - runs.parquet : long-format table with one row per evaluation.
    - benchmarks.parquet : one row per (market, fold, benchmark).
    - splits/<market>.csv : the walk-forward schedule per market.
    - configs/<run_name>.yaml : the exact configuration used.

Crucially, results are flushed to disk after each variant so a crash
doesn't lose hours of compute.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .agent import PPOConfig, deterministic_rollout, train_agent
from .benchmarks import run_benchmarks
from .data import load_local, prices_to_log_returns
from .environment import EnvConfig
from .features import StateConfig
from .metrics import all_metrics
from .reward import RewardConfig
from .walkforward import WalkForwardConfig, generate_splits


logger = logging.getLogger(__name__)

'''
FIRST EXPERIMENT VARIANT PRESETS
VARIANT_PRESETS: dict[str, dict] = {
    "V0": {"include_evt": False, "lambda_dd": 0.0, "lambda_evt": 0.0,
           "reward_variant": "V0"},
    "V1": {"include_evt": False, "lambda_dd": 1.0, "lambda_evt": 0.0,
           "reward_variant": "V1"},
    "V2": {"include_evt": True,  "lambda_dd": 1.0, "lambda_evt": 0.0,
           "reward_variant": "V2"},
    "V3": {"include_evt": False, "lambda_dd": 1.0, "lambda_evt": 0.5,
           "reward_variant": "V3"},
    "V4": {"include_evt": True,  "lambda_dd": 1.0, "lambda_evt": 0.5,
           "reward_variant": "V4"},
}
'''

# SECOND EXPERIMENT VARIANT PRESETS (SENSITIVITY ANALYSIS)
VARIANT_PRESETS = {
    "V0": {"include_evt": False, "lambda_dd": 0.0, "lambda_evt": 0.0,
           "reward_variant": "V0"},
    "V1": {"include_evt": False, "lambda_dd": 2.0, "lambda_evt": 0.0,
           "reward_variant": "V1"},
    "V2": {"include_evt": True,  "lambda_dd": 0.0, "lambda_evt": 2.0,
           "reward_variant": "V2"},
    "V3": {"include_evt": False, "lambda_dd": 2.0, "lambda_evt": 2.0,
           "reward_variant": "V3"},
    "V4": {"include_evt": True,  "lambda_dd": 2.0, "lambda_evt": 2.0,
           "reward_variant": "V4"},
}

@dataclass
class ExperimentConfig:
    run_name: str = "smoke"
    markets: tuple[str, ...] = ("DJIA",)        # data files in data/
    variants: tuple[str, ...] = ("V0", "V1", "V2", "V3", "V4")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    state_window_length: int = 60
    evt_window: int = 250
    transaction_cost: float = 0.0010
    max_episode_length: int = 252
    ppo: PPOConfig = field(default_factory=PPOConfig)
    output_dir: str = "results"
    n_eval_rollouts: int = 5             # for stability of the test metric
    save_per_fold_outputs: bool = True


def _build_configs(variant_name: str, exp_cfg: ExperimentConfig) -> tuple[EnvConfig, str]:
    """Build env config for the variant, returning the readable variant name."""
    preset = VARIANT_PRESETS[variant_name]
    state_cfg = StateConfig(
        window_length=exp_cfg.state_window_length,
        evt_window=exp_cfg.evt_window,
        include_evt=preset["include_evt"],
        normalise_market=True,
        use_auto_threshold=True,
    )
    reward_cfg = RewardConfig(
        variant=preset["reward_variant"],
        lambda_dd=preset["lambda_dd"],
        lambda_evt=preset["lambda_evt"],
        evt_window=exp_cfg.evt_window,
    )
    env_cfg = EnvConfig(
        transaction_cost=exp_cfg.transaction_cost,
        max_episode_length=exp_cfg.max_episode_length,
        state_cfg=state_cfg,
        reward_cfg=reward_cfg,
        random_starts=True,
    )
    return env_cfg, variant_name


def _eval_one_rollout(model, log_returns_test: np.ndarray, env_cfg: EnvConfig,
                      seed: int) -> dict:
    """Evaluate the model deterministically on the test slice."""
    out = deterministic_rollout(
        model, log_returns_test, env_cfg, seed=seed,
    )
    metrics = all_metrics(out["values"], out["weights_history"])
    return {
        "values": out["values"],
        "weights_history": out["weights_history"],
        **metrics,
    }


def run_one_variant(
    train_log_returns: np.ndarray,
    test_log_returns: np.ndarray,
    variant: str,
    seed: int,
    exp_cfg: ExperimentConfig,
) -> dict:
    """Train + evaluate one (variant, seed) on one fold."""
    env_cfg, _ = _build_configs(variant, exp_cfg)
    t0 = time.time()
    model = train_agent(
        train_log_returns=train_log_returns,
        env_cfg=env_cfg,
        ppo_cfg=exp_cfg.ppo,
        seed=seed,
    )
    train_time = time.time() - t0
    eval_result = _eval_one_rollout(model, test_log_returns, env_cfg, seed=seed)
    eval_result["train_time_seconds"] = train_time
    return eval_result


def run_experiment(exp_cfg: ExperimentConfig, data_dir: str = "data") -> Path:
    """Run the full experiment grid and persist to disk.

    Expects parquet files at data/<market>.parquet with columns = tickers
    and DatetimeIndex.
    """
    out_dir = Path(exp_cfg.output_dir) / exp_cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist config for reproducibility.
    cfg_dict = asdict(exp_cfg)
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))

    runs_path = out_dir / "runs.parquet"
    bench_path = out_dir / "benchmarks.parquet"
    runs_rows: list[dict] = []
    bench_rows: list[dict] = []

    data_dir_path = Path(data_dir)
    for market in exp_cfg.markets:
        prices = load_local(data_dir_path / f"{market}.parquet")
        log_returns_df = prices_to_log_returns(prices)
        log_returns_df = log_returns_df.dropna(how="any")
        index = log_returns_df.index
        splits = generate_splits(index, exp_cfg.walkforward)
        if not splits:
            logger.warning("No splits generated for %s; skipping.", market)
            continue

        # Save schedule for this market.
        from .walkforward import describe_splits
        describe_splits(splits).to_csv(out_dir / f"splits_{market}.csv", index=False)

        for split in splits:
            train_lr = log_returns_df.loc[split.train_start:split.val_end].values
            test_lr = log_returns_df.loc[split.test_start:split.test_end].values
            if test_lr.shape[0] < exp_cfg.state_window_length + 5:
                logger.warning("Fold %d for %s too short; skipping.",
                               split.fold_id, market)
                continue

            # ---- Benchmarks (same for all variants/seeds) ----
            bench_results = run_benchmarks(
                train_log_returns=train_lr,
                test_log_returns=test_lr,
                rebalance_freq=21,
                transaction_cost=exp_cfg.transaction_cost,
            )
            for name, br in bench_results.items():
                m = all_metrics(br.values, br.weights_history)
                bench_rows.append({
                    "market": market, "fold": split.fold_id,
                    "test_start": split.test_start, "test_end": split.test_end,
                    "benchmark": name, **m,
                })

            # ---- RL variants ----
            for variant in exp_cfg.variants:
                for seed in exp_cfg.seeds:
                    logger.info("Running market=%s fold=%d variant=%s seed=%d",
                                market, split.fold_id, variant, seed)
                    try:
                        result = run_one_variant(
                            train_log_returns=train_lr,
                            test_log_returns=test_lr,
                            variant=variant,
                            seed=seed,
                            exp_cfg=exp_cfg,
                        )
                    except Exception as e:
                        logger.exception("Run failed: %s", e)
                        continue

                    row = {
                        "market": market, "fold": split.fold_id,
                        "test_start": split.test_start, "test_end": split.test_end,
                        "variant": variant, "seed": seed,
                        "train_time_seconds": result["train_time_seconds"],
                    }
                    # Strip the heavy arrays; metrics only.
                    for k, v in result.items():
                        if k in ("values", "weights_history"):
                            continue
                        row[k] = v
                    runs_rows.append(row)

                    # Optional: save per-fold equity curves for V0..V4.
                    if exp_cfg.save_per_fold_outputs:
                        eq_dir = out_dir / "per_fold" / market / f"fold{split.fold_id:02d}"
                        eq_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            eq_dir / f"{variant}_seed{seed}.npz",
                            values=result["values"],
                            weights_history=result["weights_history"],
                        )

                # Flush after each variant to prevent loss on crash.
                pd.DataFrame(runs_rows).to_parquet(runs_path)
                pd.DataFrame(bench_rows).to_parquet(bench_path)

    # Final flush.
    pd.DataFrame(runs_rows).to_parquet(runs_path)
    pd.DataFrame(bench_rows).to_parquet(bench_path)
    logger.info("Experiment finished. %d runs saved to %s", len(runs_rows), out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Analysis helpers (used by scripts/04_analyze_results.py)
# ---------------------------------------------------------------------------


def aggregate_runs(runs_df: pd.DataFrame, group_by: tuple[str, ...] = ("variant",)
                   ) -> pd.DataFrame:
    """Aggregate run-level metrics by `group_by`.

    Returns mean and std for the main metrics, with `mdd` as the headline.
    """
    metric_cols = [
        "mdd", "cdar_95", "dd_q95", "calmar", "sharpe", "sortino",
        "cagr", "vol_annualised", "cvar_95", "cvar_99", "turnover_mean",
    ]
    metric_cols = [c for c in metric_cols if c in runs_df.columns]
    agg = runs_df.groupby(list(group_by))[metric_cols].agg(["mean", "std", "count"])
    return agg


def paired_table(runs_df: pd.DataFrame, baseline: str = "V1", treatment: str = "V4"
                 ) -> pd.DataFrame:
    """Build the paired (variant, seed, fold, market) table for hypothesis tests."""
    keep = runs_df[runs_df["variant"].isin([baseline, treatment])]
    pivot = keep.pivot_table(
        index=["market", "fold", "seed"],
        columns="variant",
        values="mdd",
    ).dropna(how="any").reset_index()
    pivot["diff"] = pivot[treatment] - pivot[baseline]
    return pivot

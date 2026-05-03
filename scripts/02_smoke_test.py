"""
Smoke test of the full pipeline.

Uses tiny PPO timesteps and a single (market, fold, seed, variant) so
that you can run end-to-end on a laptop in ~5-10 minutes and verify
that data loading, feature construction, training, evaluation,
benchmarks, plotting, and statistics all work together.

DO NOT use the resulting numbers as evidence — they're for plumbing only.

Usage:
    python scripts/01_download_data.py --synthetic   # if not already done
    python scripts/02_smoke_test.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
torch.set_num_threads(1)

import numpy as np
import pandas as pd

from evt_ppo.agent import PPOConfig
from evt_ppo.experiment import (ExperimentConfig, run_experiment,
                                 paired_table, aggregate_runs)
from evt_ppo.statistics import full_comparison
from evt_ppo.walkforward import WalkForwardConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    cfg = ExperimentConfig(
        run_name="smoke",
        markets=("DJIA",),                 # one market only
        variants=("V1", "V4"),             # the headline contrast
        seeds=(0, 1),                      # two seeds
        walkforward=WalkForwardConfig(
            train_years=2, val_years=1, test_years=1, step_years=2,
        ),
        state_window_length=20,
        evt_window=120,
        max_episode_length=120,
        ppo=PPOConfig(
            total_timesteps=20_000,        # tiny for smoke
            n_steps=512,
            batch_size=32,
            verbose=0,
        ),
        output_dir="results",
        save_per_fold_outputs=True,
    )
    out_dir = run_experiment(cfg, data_dir="data")

    runs = pd.read_parquet(out_dir / "runs.parquet")
    print(f"\nRuns table: {runs.shape}")
    print(runs[["market", "fold", "variant", "seed", "mdd", "calmar",
                "cagr", "sharpe"]].to_string())

    print("\nAggregated MDD by variant:")
    agg = aggregate_runs(runs, group_by=("variant",))
    print(agg["mdd"])

    paired = paired_table(runs, baseline="V1", treatment="V4")
    print("\nPaired (V4 - V1) MDD differences:")
    print(paired.to_string())

    if len(paired) >= 5:
        result = full_comparison(
            paired["V1"].values, paired["V4"].values,
            alpha=0.05, n_bootstrap=2000, seed=0,
        )
        print("\nHypothesis test V1 vs V4 (one-sided H1: MDD_V4 < MDD_V1):")
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print("\nNot enough paired runs for hypothesis test "
              "(this is expected in smoke test).")


if __name__ == "__main__":
    main()

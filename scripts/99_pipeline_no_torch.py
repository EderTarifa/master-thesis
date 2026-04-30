"""
End-to-end pipeline smoke test that bypasses PPO training.

This stand-in uses a random policy in place of the trained PPO agent so
we can verify the orchestration, metrics, statistics, and plotting code
on a machine without torch installed. The numeric results are
meaningless — this is purely a plumbing test.

Run with:
    python scripts/01_download_data.py --synthetic
    python scripts/99_pipeline_no_torch.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from evt_ppo import plots as P
from evt_ppo.benchmarks import run_benchmarks
from evt_ppo.data import load_local, prices_to_log_returns
from evt_ppo.environment import EnvConfig, PortfolioEnv
from evt_ppo.experiment import VARIANT_PRESETS, paired_table, aggregate_runs
from evt_ppo.features import StateConfig
from evt_ppo.metrics import all_metrics
from evt_ppo.reward import RewardConfig
from evt_ppo.statistics import full_comparison
from evt_ppo.walkforward import WalkForwardConfig, generate_splits

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def random_policy_rollout(env: PortfolioEnv, seed: int) -> dict:
    """Deterministic rollout with a 'concentrate-then-relax' heuristic policy.

    Different seeds produce different policies, just like different PPO
    agents would.
    """
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    # Bias each seed toward a specific subset of assets to differentiate.
    bias = rng.normal(0, 1, size=env.N).astype(np.float32)
    weights_log = [env.weights.copy()]
    while True:
        # Action = bias + small noise, mapped to simplex by the env.
        a = (bias + rng.normal(0, 0.2, size=env.N)).astype(np.float32)
        obs, reward, term, trunc, info = env.step(a)
        weights_log.append(info["weights"])
        if term or trunc:
            break
    return {
        "values": np.asarray(env.value_history, dtype=float),
        "weights_history": np.asarray(weights_log, dtype=float),
    }


def main() -> None:
    market = "DJIA"
    prices = load_local(Path("data") / f"{market}.parquet")
    log_ret = prices_to_log_returns(prices).dropna(how="any")
    print(f"Loaded {market}: {prices.shape}")

    splits = generate_splits(
        log_ret.index,
        WalkForwardConfig(train_years=2, val_years=1, test_years=1, step_years=2),
    )
    print(f"Generated {len(splits)} folds")

    variants = ("V1", "V4")
    seeds = (0, 1, 2)
    all_rows: list[dict] = []
    bench_rows: list[dict] = []

    for split in splits:
        train_lr = log_ret.loc[split.train_start:split.val_end].values
        test_lr = log_ret.loc[split.test_start:split.test_end].values
        if test_lr.shape[0] < 30:
            continue

        bench_results = run_benchmarks(
            train_lr, test_lr,
            rebalance_freq=21, transaction_cost=0.0010,
        )
        for name, br in bench_results.items():
            m = all_metrics(br.values, br.weights_history)
            bench_rows.append({"market": market, "fold": split.fold_id,
                                "benchmark": name, **m})

        for variant in variants:
            preset = VARIANT_PRESETS[variant]
            state_cfg = StateConfig(
                window_length=20, evt_window=120,
                include_evt=preset["include_evt"],
            )
            reward_cfg = RewardConfig(
                variant=preset["reward_variant"],
                lambda_dd=preset["lambda_dd"],
                lambda_evt=preset["lambda_evt"],
                evt_window=120,
            )
            env_cfg = EnvConfig(
                transaction_cost=0.001,
                max_episode_length=test_lr.shape[0] - 25,
                state_cfg=state_cfg,
                reward_cfg=reward_cfg,
                random_starts=False,
            )
            for seed in seeds:
                env = PortfolioEnv(test_lr, env_cfg, seed=seed)
                roll = random_policy_rollout(env, seed=seed)
                m = all_metrics(roll["values"], roll["weights_history"])
                all_rows.append({
                    "market": market, "fold": split.fold_id,
                    "variant": variant, "seed": seed, **m,
                })

    runs = pd.DataFrame(all_rows)
    bench = pd.DataFrame(bench_rows)
    print(f"\n=== Runs ({len(runs)}) ===")
    print(runs[["market", "fold", "variant", "seed", "mdd", "calmar",
                "cagr", "sharpe"]].to_string())

    print("\n=== Benchmarks ===")
    print(bench.groupby("benchmark")[["mdd", "calmar", "sharpe"]].mean())

    print("\n=== Aggregated by variant ===")
    print(aggregate_runs(runs, ("variant",))["mdd"])

    paired = paired_table(runs, "V1", "V4")
    print(f"\n=== Paired V1 vs V4 ({len(paired)} rows) ===")
    if len(paired) >= 5:
        result = full_comparison(
            paired["V1"].values, paired["V4"].values,
            alpha=0.05, n_bootstrap=2000, seed=0,
        )
        print("\nHypothesis test V1 vs V4:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    plots_dir = Path("results/no_torch_smoke/plots")
    P.plot_mdd_boxplots(
        runs, title="Smoke test — MDD by variant (random policy stand-in)",
        save_path=plots_dir / "mdd_boxplot_global",
    )
    if len(paired) > 0:
        P.plot_paired_diffs(
            paired["diff"].values,
            title="Smoke test — paired difference V4 - V1",
            save_path=plots_dir / "paired_diff",
        )

    # One equity plot per fold, comparing variants.
    for fold_id in runs["fold"].unique():
        sub = runs[runs["fold"] == fold_id]
        # Re-run rollouts to produce equity curves for plotting.
        split = next(s for s in splits if s.fold_id == fold_id)
        test_lr = log_ret.loc[split.test_start:split.test_end].values
        series = {}
        for variant in variants:
            preset = VARIANT_PRESETS[variant]
            state_cfg = StateConfig(
                window_length=20, evt_window=120,
                include_evt=preset["include_evt"],
            )
            reward_cfg = RewardConfig(
                variant=preset["reward_variant"],
                lambda_dd=preset["lambda_dd"], lambda_evt=preset["lambda_evt"],
                evt_window=120,
            )
            env_cfg = EnvConfig(
                transaction_cost=0.001,
                max_episode_length=test_lr.shape[0] - 25,
                state_cfg=state_cfg, reward_cfg=reward_cfg, random_starts=False,
            )
            env = PortfolioEnv(test_lr, env_cfg, seed=0)
            roll = random_policy_rollout(env, seed=0)
            series[variant] = roll["values"]
        if series:
            P.plot_value_and_drawdown(
                series,
                title=f"Smoke test — {market} fold {fold_id} (random policy)",
                save_path=plots_dir / f"equity_fold{fold_id:02d}",
            )

    print(f"\nPlots saved to: {plots_dir.absolute()}")


if __name__ == "__main__":
    main()

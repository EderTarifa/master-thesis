"""Train + evaluate a single (market, fold, variant, seed) and append to a parquet."""
import argparse, sys, logging, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Limit each subprocess to 1 BLAS thread so they don't fight each other.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)

import pandas as pd
from evt_ppo.agent import PPOConfig, deterministic_rollout
from evt_ppo.data import load_local, prices_to_log_returns
from evt_ppo.experiment import VARIANT_PRESETS, run_one_variant, ExperimentConfig, _build_configs
from evt_ppo.walkforward import WalkForwardConfig, generate_splits
from evt_ppo.metrics import all_metrics
from evt_ppo.callbacks import TelemetryCallback  # NEW


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_one_variant_with_telemetry(train_lr, test_lr, variant, seed, exp_cfg,
                                   telemetry_path=None):
    """Like experiment.run_one_variant but threads a TelemetryCallback through."""
    import time
    from evt_ppo.agent import _make_env, _HAS_SB3
    from stable_baselines3 import PPO

    env_cfg, _ = _build_configs(variant, exp_cfg)
    vec_env = _make_env(train_lr, env_cfg, seed=seed)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=exp_cfg.ppo.learning_rate,
        n_steps=exp_cfg.ppo.n_steps,
        batch_size=exp_cfg.ppo.batch_size,
        n_epochs=exp_cfg.ppo.n_epochs,
        gamma=exp_cfg.ppo.gamma,
        gae_lambda=exp_cfg.ppo.gae_lambda,
        clip_range=exp_cfg.ppo.clip_range,
        ent_coef=exp_cfg.ppo.ent_coef,
        vf_coef=exp_cfg.ppo.vf_coef,
        max_grad_norm=exp_cfg.ppo.max_grad_norm,
        policy_kwargs=exp_cfg.ppo.policy_kwargs,
        verbose=exp_cfg.ppo.verbose,
        seed=seed,
        device=exp_cfg.ppo.device,
    )
    cb = TelemetryCallback(out_path=telemetry_path) if telemetry_path else None

    t0 = time.time()
    model.learn(total_timesteps=exp_cfg.ppo.total_timesteps,
                callback=cb, progress_bar=False)
    train_time = time.time() - t0

    out = deterministic_rollout(model, test_lr, env_cfg, seed=seed)
    metrics = all_metrics(out["values"], out["weights_history"])
    return {
        "train_time_seconds": train_time,
        "values": out["values"],
        "weights_history": out["weights_history"],
        **metrics,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", default="src/configs/full_optimal.yaml")
    p.add_argument(
        "--out", default="results",
        help="Results ROOT. Output is written to "
             "<out>/rebalance_frec/every_<X>/rows/... where X = rebalance_every.",
    )
    args = p.parse_args()

    # Load config first: the rebalancing frequency drives the output folder,
    # so a sweep over rebalance_every self-organises into separate directories.
    import yaml
    with open(args.config) as f:
        d = yaml.safe_load(f)

    rebalance_every = int(d.get("rebalance_every", 21))
    out_dir = Path(args.out) / "rebalance_frec" / f"every_{rebalance_every}"
    out_file = out_dir / "rows" / (
        f"{args.market}_f{args.fold:02d}_{args.variant}_s{args.seed}.parquet"
    )
    if out_file.exists():
        logging.info(f"SKIP: already exists -> {out_file}")
        return

    wf = WalkForwardConfig(**d["walkforward"])
    ppo_d = d["ppo"]
    ppo_d.setdefault("device", "cpu")
    ppo_d["policy_kwargs"]["net_arch"] = dict(ppo_d["policy_kwargs"]["net_arch"])

    exp = ExperimentConfig(
        markets=tuple(d["markets"]), variants=tuple(d["variants"]),
        seeds=tuple(d["seeds"]), walkforward=wf, ppo=PPOConfig(**ppo_d),
        state_window_length=d["state_window_length"], evt_window=d["evt_window"],
        transaction_cost=d["transaction_cost"],
        max_episode_length=d["max_episode_length"],
        rebalance_every=rebalance_every,
        output_dir=d["output_dir"], save_per_fold_outputs=d["save_per_fold_outputs"],
    )

    prices = load_local(f"data/{args.market}.parquet")
    log_ret = prices_to_log_returns(prices).dropna(how="any")
    splits = generate_splits(log_ret.index, exp.walkforward)
    if args.fold >= len(splits):
        logging.info(f"SKIP: market={args.market} fold={args.fold} "
                    f"out of range ({len(splits)} folds available)")
        return
    split = splits[args.fold]

    train = log_ret.loc[split.train_start:split.val_end].values
    test = log_ret.loc[split.test_start:split.test_end].values

    # Telemetry path
    telemetry_path = None
    if args.telemetry:
        telemetry_path = Path(args.out) / "telemetry" / (
            f"{args.market}_f{args.fold:02d}_{args.variant}_s{args.seed}.parquet"
        )

    res = run_one_variant_with_telemetry(
        train, test, args.variant, args.seed, exp,
        telemetry_path=telemetry_path,
    )
    
    row = {
        "market": args.market, "fold": args.fold,
        "variant": args.variant, "seed": args.seed,
        "rebalance_every": rebalance_every,   # record the frequency in the row
        "train_time_seconds": res["train_time_seconds"]
    }
    for k, v in res.items():
        if k not in ("values", "weights_history"):
            row[k] = v

    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(out_file)

if __name__ == "__main__":
    main()
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
from evt_ppo.agent import PPOConfig
from evt_ppo.data import load_local, prices_to_log_returns
from evt_ppo.experiment import VARIANT_PRESETS, run_one_variant, ExperimentConfig
from evt_ppo.walkforward import WalkForwardConfig, generate_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", default="src/configs/full_optimal.yaml")
    p.add_argument("--out", default="results/full_optimal")
    args = p.parse_args()

    out_file = Path(args.out) / "rows" / f"{args.market}_f{args.fold:02d}_{args.variant}_s{args.seed}.parquet"
    if out_file.exists():
        logging.info(f"SKIP: already exists -> {out_file}")
        return

    import yaml
    with open(args.config) as f:
        d = yaml.safe_load(f)

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
        output_dir=d["output_dir"], save_per_fold_outputs=d["save_per_fold_outputs"],
    )

    prices = load_local(f"data/{args.market}.parquet")
    log_ret = prices_to_log_returns(prices).dropna(how="any")
    splits = generate_splits(log_ret.index, exp.walkforward)
    split = splits[args.fold]

    train = log_ret.loc[split.train_start:split.val_end].values
    test = log_ret.loc[split.test_start:split.test_end].values

    res = run_one_variant(train, test, args.variant, args.seed, exp)
    row = {
        "market": args.market, "fold": args.fold,
        "variant": args.variant, "seed": args.seed,
        "train_time_seconds": res["train_time_seconds"]
    }
    for k, v in res.items():
        if k not in ("values", "weights_history"):
            row[k] = v

    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(out_file)

if __name__ == "__main__":
    main()
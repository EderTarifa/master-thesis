"""
run_one con override de hiperparámetros para experimentos de sensibilidad.

Permite sobreescribir lambda_dd, lambda_evt y evt_window desde la línea
de comandos sin tener que crear un YAML por combinación.

Uso:
    python scripts/run_one_sens.py \
        --market DJIA --fold 0 --variant V4 --seed 0 \
        --lambda-dd 1.0 --lambda-evt 0.5 --evt-window 250 \
        --total-timesteps 100000 \
        --config src/configs/intermediate_B.yaml \
        --out results/sens_lambda

El parquet se nombra con todos los hiperparámetros para evitar colisiones.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Forzar 1-thread de PyTorch (igual que en run_one.py)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# MAC OPTIMIZATION
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")    # Apple Accelerate
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch
torch.set_num_threads(1)

import pandas as pd
import yaml

from evt_ppo.agent import PPOConfig
from evt_ppo.data import load_local, prices_to_log_returns
from evt_ppo.experiment import VARIANT_PRESETS, ExperimentConfig, run_one_variant
from evt_ppo.walkforward import WalkForwardConfig, generate_splits

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lambda-dd", type=float, default=None)
    p.add_argument("--lambda-evt", type=float, default=None)
    p.add_argument("--evt-window", type=int, default=None)
    p.add_argument("--evt-recompute-every", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=None)
    args = p.parse_args()

    with open(args.config) as f:
        d = yaml.safe_load(f)

    wf = WalkForwardConfig(**d["walkforward"])
    ppo_d = d["ppo"].copy()
    ppo_d.setdefault("device", "cpu")
    ppo_d["policy_kwargs"]["net_arch"] = dict(ppo_d["policy_kwargs"]["net_arch"])
    if args.total_timesteps is not None:
        ppo_d["total_timesteps"] = args.total_timesteps

    exp = ExperimentConfig(
        run_name=d.get("run_name", "sens"),
        markets=tuple(d["markets"]),
        variants=tuple(d["variants"]),
        seeds=tuple(d["seeds"]),
        walkforward=wf,
        ppo=PPOConfig(**ppo_d),
        state_window_length=d["state_window_length"],
        evt_window=args.evt_window if args.evt_window else d["evt_window"],
        transaction_cost=d["transaction_cost"],
        max_episode_length=d["max_episode_length"],
        output_dir=d["output_dir"],
        save_per_fold_outputs=d.get("save_per_fold_outputs", False),
    )

    # Override de lambda y evt_recompute_every via monkey-patch del preset
    preset = VARIANT_PRESETS[args.variant].copy()
    if args.lambda_dd is not None:
        preset["lambda_dd"] = args.lambda_dd
    if args.lambda_evt is not None:
        preset["lambda_evt"] = args.lambda_evt
    VARIANT_PRESETS[args.variant] = preset

    # Cargar datos
    prices = load_local(f"data/{args.market}.parquet")
    log_ret = prices_to_log_returns(prices).dropna(how="any")
    splits = generate_splits(log_ret.index, exp.walkforward)
    if args.fold >= len(splits):
        raise ValueError(f"Fold {args.fold} >= {len(splits)} splits")
    split = splits[args.fold]
    train = log_ret.loc[split.train_start:split.val_end].values
    test = log_ret.loc[split.test_start:split.test_end].values

    # Ejecutar
    res = run_one_variant(train, test, args.variant, args.seed, exp)

    # Construir nombre de archivo con todos los hiperparámetros
    parts = [
        args.market,
        f"f{args.fold:02d}",
        args.variant,
        f"s{args.seed}",
    ]
    if args.lambda_dd is not None:
        parts.append(f"l1_{args.lambda_dd:.2f}")
    if args.lambda_evt is not None:
        parts.append(f"l2_{args.lambda_evt:.2f}")
    if args.evt_window is not None:
        parts.append(f"W{args.evt_window}")
    if args.evt_recompute_every is not None:
        parts.append(f"K{args.evt_recompute_every}")
    if args.total_timesteps is not None:
        parts.append(f"T{args.total_timesteps}")
    fname = "_".join(parts) + ".parquet"

    row = {
        "market": args.market, "fold": args.fold,
        "variant": args.variant, "seed": args.seed,
        "lambda_dd": args.lambda_dd if args.lambda_dd is not None else preset["lambda_dd"],
        "lambda_evt": args.lambda_evt if args.lambda_evt is not None else preset["lambda_evt"],
        "evt_window": args.evt_window if args.evt_window else exp.evt_window,
        "total_timesteps": args.total_timesteps if args.total_timesteps else exp.ppo.total_timesteps,
        "train_time_seconds": res["train_time_seconds"],
    }
    for k, v in res.items():
        if k not in ("values", "weights_history"):
            row[k] = v

    out = Path(args.out) / "rows"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(out / fname)


if __name__ == "__main__":
    main()
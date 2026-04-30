"""
Run the full experiment defined in a YAML config file.

Usage:
    python scripts/03_run_full_experiment.py --config src/configs/full.yaml

This will iterate over (market, fold, seed, variant) and persist
results incrementally. With the default config (3 markets x 13 folds x
5 seeds x 5 variants x 200k timesteps) expect runtimes on the order of
days on a single machine without GPU, hours with GPU. Plan accordingly.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
torch.set_num_threads(1)

import yaml

from evt_ppo.agent import PPOConfig
from evt_ppo.experiment import ExperimentConfig, run_experiment
from evt_ppo.walkforward import WalkForwardConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def load_config(path: Path) -> ExperimentConfig:
    with open(path) as f:
        d = yaml.safe_load(f)

    wf = WalkForwardConfig(**d.pop("walkforward"))
    ppo_d = d.pop("ppo")
    ppo_d.setdefault("policy_kwargs", {"net_arch": {"pi": [128, 128], "vf": [128, 128]}})
    if "policy_kwargs" in ppo_d and "net_arch" in ppo_d["policy_kwargs"]:
        # YAML deserialises dict but stable-baselines3 expects net_arch as a dict-with-pi-vf.
        ppo_d["policy_kwargs"]["net_arch"] = dict(ppo_d["policy_kwargs"]["net_arch"])
    ppo = PPOConfig(**ppo_d)

    return ExperimentConfig(
        markets=tuple(d.pop("markets")),
        variants=tuple(d.pop("variants")),
        seeds=tuple(d.pop("seeds")),
        walkforward=wf,
        ppo=ppo,
        **d,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = run_experiment(cfg, data_dir=args.data_dir)
    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()

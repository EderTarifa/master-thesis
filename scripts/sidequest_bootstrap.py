"""
Side-quest C: Robustez del bootstrap y análisis de potencia estadística.

C.1: Sensibilidad del IC bootstrap al tamaño de bloque.
C.2: Curva de potencia del test pareado.

Funciona sobre los parquets parciales del experimento principal,
así que se puede ejecutar mientras el experimento sigue corriendo.

Uso:
    python scripts/sidequest_bootstrap.py \
        --rows-dir results/full/rows \
        --baseline V1 --treatment V4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from evt_ppo.statistics import block_bootstrap_diff_ci, paired_one_sided_t


def c1_block_size_robustness(diffs: np.ndarray, out_dir: Path) -> pd.DataFrame:
    """Sensibilidad del IC bootstrap al tamaño de bloque."""
    rows = []
    for block_size in [1, 2, 3, 5, 10, 20]:
        if len(diffs) < block_size:
            continue
        mean_d, lo, hi = block_bootstrap_diff_ci(
            np.zeros(len(diffs)),  # base ficticio (cancela)
            diffs,
            block_size=block_size,
            n_resamples=5000,
            alpha=0.05,
            seed=0,
        )
        rows.append({
            "block_size": block_size,
            "mean_diff": float(mean_d),
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "ci_width": float(hi - lo),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "bootstrap_blocksize_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df["block_size"], df["mean_diff"],
                 yerr=[df["mean_diff"] - df["ci_lower"],
                        df["ci_upper"] - df["mean_diff"]],
                 fmt='o-', capsize=5, lw=2)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Block size")
    ax.set_ylabel("Mean difference (V4 - V1) MDD")
    ax.set_title(f"Block bootstrap CI sensitivity (n = {len(diffs)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "blocksize_sensitivity.png", dpi=150)
    plt.close(fig)

    print(f"[C.1] Block-size sensitivity: {len(rows)} sizes tested")
    print(df.round(5).to_string(index=False))
    return df


def c2_power_analysis(observed_std: float, n_obs: int,
                       out_dir: Path) -> pd.DataFrame:
    """Curva de potencia del test t pareado.

    Para distintos tamaños de efecto y tamaños muestrales, simula la
    probabilidad de rechazar H0.
    """
    effect_sizes = np.linspace(0.0, 0.05, 21)  # MDD reduction from 0 to 5pp
    sample_sizes = [50, 100, 195, 300, 500]
    n_simulations = 2000
    rng = np.random.default_rng(42)

    rows = []
    for n in sample_sizes:
        for eff in effect_sizes:
            rejections = 0
            for _ in range(n_simulations):
                # Simulate paired diffs ~ N(-eff, observed_std)
                diffs = rng.normal(-eff, observed_std, size=n)
                # One-sided t-test, H1: mean < 0
                t = stats.ttest_1samp(diffs, 0, alternative="less")
                if t.pvalue < 0.05:
                    rejections += 1
            rows.append({
                "n": n,
                "effect_size": eff,
                "power": rejections / n_simulations,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "power_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    for n in sample_sizes:
        sub = df[df["n"] == n]
        ax.plot(sub["effect_size"], sub["power"],
                 marker="o", label=f"n = {n}", lw=2)
    ax.axhline(0.80, color="red", ls="--", lw=1, label="power = 0.80")
    ax.set_xlabel("True effect size (MDD reduction in absolute units)")
    ax.set_ylabel("Statistical power (P[reject H0 | H1 true])")
    ax.set_title(f"Power curves (paired t-test, alpha=0.05)\n"
                  f"Simulated SD = {observed_std:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "power_curves.png", dpi=150)
    plt.close(fig)

    print(f"[C.2] Power analysis done")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows-dir", default="results/full/rows")
    p.add_argument("--baseline", default="V1")
    p.add_argument("--treatment", default="V4")
    p.add_argument("--out-dir", default="results/sidequest_bootstrap")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar parquets parciales
    files = sorted(Path(args.rows_dir).glob("*.parquet"))
    if not files:
        print(f"No parquets in {args.rows_dir}.")
        # Aun asi podemos hacer C.2 con un std hipotetico
        print("[C.2] Running power analysis with hypothetical SD = 0.05")
        c2_power_analysis(0.05, n_obs=195, out_dir=out_dir)
        return

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} parquets")

    # Filtrar y parear
    sub = df[df["variant"].isin([args.baseline, args.treatment])]
    paired = sub.pivot_table(
        index=["market", "fold", "seed"],
        columns="variant", values="mdd",
    ).dropna(how="any")

    if args.baseline not in paired.columns or args.treatment not in paired.columns:
        print(f"Insufficient data for {args.baseline} vs {args.treatment}")
        c2_power_analysis(0.05, n_obs=195, out_dir=out_dir)
        return

    diffs = (paired[args.treatment] - paired[args.baseline]).values
    print(f"\nPaired observations: n = {len(diffs)}")
    print(f"  mean diff = {diffs.mean():+.5f}")
    print(f"  std diff  = {diffs.std(ddof=1):.5f}")

    # C.1
    c1_block_size_robustness(diffs, out_dir)

    # C.2 con std observado
    c2_power_analysis(diffs.std(ddof=1), n_obs=len(diffs), out_dir=out_dir)

    print(f"\nOutputs in {out_dir}/")


if __name__ == "__main__":
    main()
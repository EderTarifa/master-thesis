"""
Análisis condicional del efecto V4 vs V1 por régimen y por shape (xi) de cola.

PREGUNTA CIENTÍFICA: ¿La mejora de V4 sobre V1 es uniforme entre regímenes,
o se concentra en folds de cola pesada (xi alto) y crisis?

Si el efecto es heterogéneo y se concentra en colas pesadas, validas
empíricamente la motivación teórica del TFM: EVT debe aportar precisamente
en regímenes donde la cola es informativa.

Lee:
  - results/full_optimal/rows/*.parquet  (resultados RL)
  - results/sidequest_evt/regime_classification.csv  (clasificación EVT)

Produce:
  - tables/conditional_diff_by_regime.csv: diff V4-V1 por régimen
  - tables/conditional_diff_by_market_regime.csv: idem por mercado x régimen
  - tables/diff_by_xi_bins.csv: diff V4-V1 binned por shape parameter
  - tables/regression_diff_on_xi.csv: regresión OLS diff ~ xi + kurtosis
  - figs/diff_vs_xi_scatter.png: scatter con línea de tendencia
  - figs/diff_by_regime_boxplot.png: boxplot por régimen
  - figs/heatmap_diff_market_regime.png: heatmap mercado x régimen

Uso:
    python scripts/conditional_analysis.py \
        --rows-dir results/full_optimal/rows \
        --regimes-csv results/sidequest_evt/regime_classification.csv \
        --out-dir results/conditional_analysis
"""
from __future__ import annotations

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


def load_data(rows_dir: Path, regimes_csv: Path) -> pd.DataFrame:
    """Cargar runs y unir con clasificación de regímenes."""
    files = sorted(rows_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquets in {rows_dir}")
    runs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(runs)} runs")

    regimes = pd.read_csv(regimes_csv)
    # Renombrar para evitar choques
    regimes = regimes[["market", "fold", "regime", "evt_xi",
                        "evt_cvar99", "kurtosis", "mdd_1N"]]
    regimes.columns = ["market", "fold", "regime", "fold_xi",
                        "fold_cvar99", "fold_kurtosis", "fold_mdd_1N"]

    merged = runs.merge(regimes, on=["market", "fold"], how="left")
    n_unmerged = merged["regime"].isna().sum()
    if n_unmerged > 0:
        print(f"Warning: {n_unmerged} rows without regime classification")
        merged = merged.dropna(subset=["regime"])
    print(f"Merged: {len(merged)} rows with regime info")
    return merged


def make_paired(df: pd.DataFrame, baseline: str, treatment: str) -> pd.DataFrame:
    """Crear tabla pareada por (market, fold, seed) entre dos variantes."""
    sub = df[df["variant"].isin([baseline, treatment])]
    pivot = sub.pivot_table(
        index=["market", "fold", "seed", "regime", "fold_xi",
                "fold_cvar99", "fold_kurtosis", "fold_mdd_1N"],
        columns="variant", values="mdd",
    ).dropna(how="any").reset_index()
    pivot["diff"] = pivot[treatment] - pivot[baseline]
    return pivot


def diff_by_regime(paired: pd.DataFrame, out_dir: Path,
                    baseline: str, treatment: str) -> pd.DataFrame:
    """Análisis del diff V4-V1 por régimen."""
    rows = []
    for regime in ["benign", "moderate", "crisis"]:
        sub = paired[paired["regime"] == regime]
        if len(sub) < 5:
            continue
        diff = sub["diff"].values
        # Test t pareado: H0: diff >= 0, H1: diff < 0
        t_res = stats.ttest_1samp(diff, 0, alternative="less")
        w_res = stats.wilcoxon(diff, alternative="less") if len(diff) >= 5 else None
        rows.append({
            "regime": regime,
            "n": len(diff),
            f"mean_{baseline}": float(sub[baseline].mean()),
            f"mean_{treatment}": float(sub[treatment].mean()),
            "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)),
            f"pct_{treatment}_better": float((diff < 0).mean() * 100),
            "t_pvalue": float(t_res.pvalue),
            "wilcoxon_pvalue": float(w_res.pvalue) if w_res else np.nan,
            "ci_lower_95": float(np.quantile(diff, 0.025)),
            "ci_upper_95": float(np.quantile(diff, 0.975)),
        })
    df = pd.DataFrame(rows).round(5)
    df.to_csv(out_dir / "tables" / "conditional_diff_by_regime.csv", index=False)
    print(f"\n=== Diff {treatment}-{baseline} by regime ===")
    print(df.to_string(index=False))
    return df


def diff_by_market_regime(paired: pd.DataFrame, out_dir: Path,
                           baseline: str, treatment: str) -> pd.DataFrame:
    """Cruce mercado x régimen."""
    rows = []
    for market in paired["market"].unique():
        for regime in ["benign", "moderate", "crisis"]:
            sub = paired[(paired["market"] == market)
                         & (paired["regime"] == regime)]
            if len(sub) < 3:
                continue
            diff = sub["diff"].values
            rows.append({
                "market": market,
                "regime": regime,
                "n": len(diff),
                "mean_diff": float(diff.mean()),
                "median_diff": float(np.median(diff)),
                f"pct_{treatment}_better": float((diff < 0).mean() * 100),
            })
    df = pd.DataFrame(rows).round(5)
    df.to_csv(out_dir / "tables" / "conditional_diff_by_market_regime.csv",
               index=False)
    print(f"\n=== Diff {treatment}-{baseline} by market x regime ===")
    print(df.to_string(index=False))
    return df


def diff_by_xi_bins(paired: pd.DataFrame, out_dir: Path,
                     baseline: str, treatment: str) -> pd.DataFrame:
    """Binear por shape parameter para ver tendencia continua."""
    paired = paired.copy()
    paired["xi_bin"] = pd.cut(
        paired["fold_xi"],
        bins=[-np.inf, -0.2, 0.0, 0.1, np.inf],
        labels=["very_bounded", "bounded", "near_zero", "heavy"],
    )
    rows = []
    for bin_label in ["very_bounded", "bounded", "near_zero", "heavy"]:
        sub = paired[paired["xi_bin"] == bin_label]
        if len(sub) < 3:
            continue
        diff = sub["diff"].values
        rows.append({
            "xi_bin": str(bin_label),
            "n": len(diff),
            "xi_mean": float(sub["fold_xi"].mean()),
            "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)),
            f"pct_{treatment}_better": float((diff < 0).mean() * 100),
        })
    df = pd.DataFrame(rows).round(5)
    df.to_csv(out_dir / "tables" / "diff_by_xi_bins.csv", index=False)
    print(f"\n=== Diff {treatment}-{baseline} by xi bin ===")
    print(df.to_string(index=False))
    return df


def regression_on_xi(paired: pd.DataFrame, out_dir: Path,
                      treatment: str) -> pd.DataFrame:
    """Regresión OLS: diff ~ fold_xi + fold_kurtosis."""
    from numpy.linalg import lstsq

    y = paired["diff"].values
    X1 = np.column_stack([np.ones(len(paired)), paired["fold_xi"].values])
    X2 = np.column_stack([np.ones(len(paired)),
                            paired["fold_xi"].values,
                            paired["fold_kurtosis"].values])

    coef1, *_ = lstsq(X1, y, rcond=None)
    coef2, *_ = lstsq(X2, y, rcond=None)

    # R^2
    yhat1 = X1 @ coef1
    yhat2 = X2 @ coef2
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_1 = 1 - np.sum((y - yhat1) ** 2) / ss_tot
    r2_2 = 1 - np.sum((y - yhat2) ** 2) / ss_tot

    # SE de los coeficientes (asumiendo homocedasticidad)
    n, k1 = X1.shape
    sigma2_1 = np.sum((y - yhat1) ** 2) / (n - k1)
    cov1 = sigma2_1 * np.linalg.inv(X1.T @ X1)
    se1 = np.sqrt(np.diag(cov1))
    t1 = coef1 / se1
    p1 = 2 * (1 - stats.t.cdf(np.abs(t1), df=n - k1))

    n, k2 = X2.shape
    sigma2_2 = np.sum((y - yhat2) ** 2) / (n - k2)
    cov2 = sigma2_2 * np.linalg.inv(X2.T @ X2)
    se2 = np.sqrt(np.diag(cov2))
    t2 = coef2 / se2
    p2 = 2 * (1 - stats.t.cdf(np.abs(t2), df=n - k2))

    rows = [
        {"model": "diff ~ xi", "term": "intercept",
         "coef": float(coef1[0]), "se": float(se1[0]),
         "t": float(t1[0]), "p": float(p1[0]), "r2": float(r2_1)},
        {"model": "diff ~ xi", "term": "fold_xi",
         "coef": float(coef1[1]), "se": float(se1[1]),
         "t": float(t1[1]), "p": float(p1[1]), "r2": float(r2_1)},
        {"model": "diff ~ xi + kurt", "term": "intercept",
         "coef": float(coef2[0]), "se": float(se2[0]),
         "t": float(t2[0]), "p": float(p2[0]), "r2": float(r2_2)},
        {"model": "diff ~ xi + kurt", "term": "fold_xi",
         "coef": float(coef2[1]), "se": float(se2[1]),
         "t": float(t2[1]), "p": float(p2[1]), "r2": float(r2_2)},
        {"model": "diff ~ xi + kurt", "term": "fold_kurtosis",
         "coef": float(coef2[2]), "se": float(se2[2]),
         "t": float(t2[2]), "p": float(p2[2]), "r2": float(r2_2)},
    ]
    df = pd.DataFrame(rows).round(5)
    df.to_csv(out_dir / "tables" / "regression_diff_on_xi.csv", index=False)
    print(f"\n=== OLS regression: diff_{treatment} ~ xi (+kurt) ===")
    print(df.to_string(index=False))
    return df


def plot_scatter_diff_xi(paired: pd.DataFrame, out_dir: Path,
                          baseline: str, treatment: str) -> None:
    """Scatter diff vs xi con línea OLS."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"benign": "#2ca02c", "moderate": "#ff7f0e", "crisis": "#d62728"}
    for regime, color in colors.items():
        sub = paired[paired["regime"] == regime]
        if len(sub) > 0:
            ax.scatter(sub["fold_xi"], sub["diff"], alpha=0.5,
                        color=color, label=f"{regime} (n={len(sub)})", s=40)

    # Línea OLS
    z = np.polyfit(paired["fold_xi"].values, paired["diff"].values, 1)
    xs = np.linspace(paired["fold_xi"].min(), paired["fold_xi"].max(), 100)
    ax.plot(xs, np.polyval(z, xs), "k--", lw=2,
             label=f"OLS: slope = {z[0]:+.4f}")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="gray", lw=0.4, ls=":")
    ax.set_xlabel(r"EVT shape parameter $\hat{\xi}$ (test fold)")
    ax.set_ylabel(rf"MDD($\mathrm{{{treatment}}}$) - MDD($\mathrm{{{baseline}}}$)")
    ax.set_title(f"Effect of EVT depends on tail shape?\n"
                  f"Negative diff = {treatment} better than {baseline}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "diff_vs_xi_scatter.png", dpi=150)
    plt.close(fig)


def plot_boxplot_by_regime(paired: pd.DataFrame, out_dir: Path,
                            baseline: str, treatment: str) -> None:
    """Boxplot del diff por régimen."""
    fig, ax = plt.subplots(figsize=(8, 5))
    regimes = ["benign", "moderate", "crisis"]
    data = [paired[paired["regime"] == r]["diff"].values for r in regimes]
    labels = [f"{r}\n(n={len(d)})" for r, d in zip(regimes, data)]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    colors = {"benign": "#a8e6a3", "moderate": "#ffcb91", "crisis": "#f8b4b4"}
    for patch, regime in zip(bp["boxes"], regimes):
        patch.set_facecolor(colors[regime])
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_ylabel(rf"MDD({treatment}) - MDD({baseline})")
    ax.set_title(f"Effect heterogeneity: {treatment} vs {baseline} by regime\n"
                  f"Below 0 = {treatment} reduces drawdown")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "diff_by_regime_boxplot.png", dpi=150)
    plt.close(fig)


def plot_heatmap_market_regime(market_regime: pd.DataFrame,
                                 out_dir: Path,
                                 treatment: str, baseline: str) -> None:
    """Heatmap mercado x régimen del diff medio."""
    pivot = market_regime.pivot_table(
        index="market", columns="regime", values="mean_diff",
    )
    pivot = pivot[["benign", "moderate", "crisis"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                    vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Regime")
    ax.set_title(f"Mean diff MDD({treatment} - {baseline}) by market x regime\n"
                  "Green = treatment better, Red = baseline better")
    plt.colorbar(im, ax=ax, label="Mean diff")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.4f}", ha="center", va="center",
                         color="white" if abs(v) > 0.025 else "black",
                         fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "heatmap_diff_market_regime.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows-dir", default="results/full_optimal/rows")
    p.add_argument("--regimes-csv",
                    default="results/sidequest_evt/regime_classification.csv")
    p.add_argument("--out-dir", default="results/conditional_analysis")
    p.add_argument("--baseline", default="V1")
    p.add_argument("--treatment", default="V4")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.rows_dir), Path(args.regimes_csv))

    paired = make_paired(df, args.baseline, args.treatment)
    print(f"\nPaired observations: {len(paired)}")

    diff_by_regime(paired, out_dir, args.baseline, args.treatment)
    market_reg = diff_by_market_regime(paired, out_dir,
                                          args.baseline, args.treatment)
    diff_by_xi_bins(paired, out_dir, args.baseline, args.treatment)
    regression_on_xi(paired, out_dir, args.treatment)

    plot_scatter_diff_xi(paired, out_dir, args.baseline, args.treatment)
    plot_boxplot_by_regime(paired, out_dir, args.baseline, args.treatment)
    plot_heatmap_market_regime(market_reg, out_dir,
                                 args.treatment, args.baseline)

    # Repetir para V0 vs V3 (otro contraste interesante)
    if {"V0", "V3"}.issubset(df["variant"].unique()):
        print("\n" + "=" * 60)
        print("BONUS: V0 vs V3 (efecto del regularizador EVT puro)")
        print("=" * 60)
        paired_v3 = make_paired(df, "V0", "V3")
        diff_by_regime(paired_v3, out_dir, "V0", "V3")

    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
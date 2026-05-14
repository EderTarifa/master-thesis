"""
Analiza el experimento de generalización sectorial.

Mantiene una distinción clara entre:
  - Mercados ORIGINALES de equities (DJIA, SP50, IBEX, HSI) que constituyen
    el experimento principal de la memoria.
  - Mercados de EXTENSIÓN cross-asset (BRD_CMDY, CRYPTO, BOND_US) que
    forman el experimento de generalización trans-clase.

Genera tablas y gráficas equivalentes a las del capítulo 4, agrupadas por
clase de activo, además de un análisis específico de tail thickness vs
magnitud del efecto V4-V1 (validación out-of-sample del hallazgo de la
sección 5.3 sobre el papel de la kurtosis como moderador).

Uso:
    python scripts/05_analyze_sectoral_generalization.py \\
        --results-dir results/full_optimal/

Outputs (bajo <results-dir>/analysis_sectoral/):
  tables/
    agg_by_asset_class.csv
    agg_by_market_within_class.csv
    paired_V1_V4_by_class.csv
    paired_V1_V4_by_market.csv
    hypothesis_tests_extension.csv
    tail_stats_by_market.csv
    ols_kurtosis_vs_effect.txt
    master_table_all_markets.csv          <- tabla maestra para LaTeX
  plots/
    forest_plot_V1_V4_by_class.{png,svg}
    mdd_boxplot_by_asset_class.{png,svg}
    paired_diff_by_class.{png,svg}
    scatter_kurtosis_vs_effect.{png,svg}
    equity_<MARKET>_fold<XX>.{png,svg}    (uno por fold de cada extensión)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from evt_ppo import plots as P
from evt_ppo.experiment import paired_table
from evt_ppo.statistics import full_comparison

# ---------------------------------------------------------------------------
# Clasificación de mercados por clase de activo
# ---------------------------------------------------------------------------

ASSET_CLASS: dict[str, str] = {
    # Originales
    "DJIA":     "Equity (US large cap)",
    "SP50":     "Equity (US broad)",
    "IBEX":     "Equity (Europe)",
    "HSI":      "Equity (Asia)",
    # Extensiones
    "BRD_CMDY": "Commodity",
    "CRYPTO":   "Cryptocurrency",
    "BOND_US":  "Fixed Income",
    "FX_MIX":   "Foreign Exchange",
}

ORIGINAL_MARKETS = {"DJIA", "SP50", "IBEX", "HSI"}
EXTENSION_MARKETS = {"BRD_CMDY", "CRYPTO", "BOND_US", "FX_MIX"}


# ---------------------------------------------------------------------------
# Funciones de tablas
# ---------------------------------------------------------------------------

def add_asset_class_column(df: pd.DataFrame) -> pd.DataFrame:
    """Anade columnas 'asset_class' y 'is_extension' al runs.parquet."""
    df = df.copy()
    df["asset_class"] = df["market"].map(ASSET_CLASS).fillna("Unknown")
    df["is_extension"] = df["market"].isin(EXTENSION_MARKETS)
    return df


def aggregate_by_asset_class(runs: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Tabla principal: media de MDD/CDaR/Sharpe/Calmar por clase x variante."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [c for c in
                   ("mdd", "cdar_95", "sharpe", "sortino", "calmar",
                    "cagr", "vol_annualised", "turnover_mean")
                   if c in runs.columns]

    agg = (runs.groupby(["asset_class", "variant"])[metric_cols]
                .agg(["mean", "std", "count"])
                .round(4))
    agg.to_csv(out_dir / "agg_by_asset_class.csv")
    return agg


def aggregate_by_market(runs: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Tabla por mercado individual dentro de cada clase."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [c for c in
                   ("mdd", "cdar_95", "sharpe", "sortino", "calmar",
                    "cagr", "vol_annualised", "turnover_mean")
                   if c in runs.columns]

    # Ordenar por clase para que la tabla se lea natural
    runs_ord = runs.copy()
    class_order = ["Equity (US large cap)", "Equity (US broad)",
                   "Equity (Europe)", "Equity (Asia)",
                   "Commodity", "Cryptocurrency", "Fixed Income",
                   "Foreign Exchange"]
    runs_ord["asset_class"] = pd.Categorical(
        runs_ord["asset_class"], categories=class_order, ordered=True
    )
    agg = (runs_ord.groupby(["asset_class", "market", "variant"], observed=True)
                    [metric_cols].agg(["mean", "std", "count"]).round(4))
    agg.to_csv(out_dir / "agg_by_market_within_class.csv")
    return agg


def paired_tests_by_market(runs: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Test pareado V1 vs V4 separado por mercado."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for market in sorted(runs["market"].unique()):
        sub = runs[runs["market"] == market]
        if not {"V1", "V4"}.issubset(sub["variant"].unique()):
            continue
        paired = paired_table(sub, baseline="V1", treatment="V4")
        if len(paired) < 5:
            continue
        result = full_comparison(
            paired["V1"].values, paired["V4"].values,
            alpha=0.05, n_bootstrap=5000, seed=0,
        )
        rows.append({
            "market": market,
            "asset_class": ASSET_CLASS.get(market, "Unknown"),
            "n_pairs": len(paired),
            **result,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "paired_V1_V4_by_market.csv", index=False)
    return df


def paired_tests_by_class(runs: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Test pareado V1 vs V4 agrupando por clase de activo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in sorted(runs["asset_class"].unique()):
        sub = runs[runs["asset_class"] == cls]
        if not {"V1", "V4"}.issubset(sub["variant"].unique()):
            continue
        paired = paired_table(sub, baseline="V1", treatment="V4")
        if len(paired) < 5:
            continue
        result = full_comparison(
            paired["V1"].values, paired["V4"].values,
            alpha=0.05, n_bootstrap=5000, seed=0,
        )
        rows.append({
            "asset_class": cls,
            "markets_in_class": ", ".join(sorted(sub["market"].unique())),
            "n_pairs": len(paired),
            **result,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "paired_V1_V4_by_class.csv", index=False)
    return df


def extension_hypothesis_tests(runs: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Tests V0-V1, V1-V2, V1-V3, V1-V4, V2-V4, V3-V4 SOLO en extensiones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extension = runs[runs["is_extension"]].copy()
    rows = []
    contrasts = [("V0", "V1"), ("V1", "V2"), ("V1", "V3"), ("V1", "V4"),
                 ("V2", "V4"), ("V3", "V4")]
    for baseline, treatment in contrasts:
            for market in sorted(extension["market"].unique()):
                sub = extension[extension["market"] == market]
                if not {baseline, treatment}.issubset(sub["variant"].unique()):
                    continue
                paired = paired_table(sub, baseline=baseline, treatment=treatment)
                if len(paired) < 5:
                    continue
                result = full_comparison(
                    paired[baseline].values, paired[treatment].values,
                    alpha=0.05, n_bootstrap=5000, seed=0,
                )
                rows.append({
                    "market": market,
                    "asset_class": ASSET_CLASS.get(market, "Unknown"),
                    "baseline": baseline,
                    "treatment": treatment,
                    **result,
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hypothesis_tests_extension.csv", index=False)
    return df


def master_table_for_latex(runs: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Tabla maestra wide-format: filas = mercado, columnas = MDD por variante.

    Es la tabla que vas a meter directamente en LaTeX como Tabla 4.X de la
    sección de generalización trans-clase (equivalente cross-asset a la
    Tabla 4.6 actual sobre mercados de equity).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pivot = (runs.pivot_table(index=["asset_class", "market"],
                               columns="variant", values="mdd",
                               aggfunc="mean")
                  .round(4))

    # Calcular reduccion porcentual V0 -> V4 para columna extra
    if {"V0", "V4"}.issubset(pivot.columns):
        pivot["red_pct_V0V4"] = ((pivot["V0"] - pivot["V4"]) /
                                  pivot["V0"] * 100).round(2)
    if {"V1", "V4"}.issubset(pivot.columns):
        pivot["red_pct_V1V4"] = ((pivot["V1"] - pivot["V4"]) /
                                  pivot["V1"] * 100).round(2)

    pivot.to_csv(out_dir / "master_table_all_markets.csv")
    return pivot


# ---------------------------------------------------------------------------
# Análisis de tail thickness vs efecto
# ---------------------------------------------------------------------------

def compute_tail_stats_per_market(data_dir: Path, out_dir: Path
                                   ) -> pd.DataFrame:
    """Calcula kurtosis y shape parameter ξ del POT-GPD por mercado."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from evt_ppo.data import load_local, prices_to_log_returns
    from evt_ppo.evt import fit_gpd

    rows = []
    for market in ASSET_CLASS:
        path = data_dir / f"{market}.parquet"
        if not path.exists():
            continue
        prices = load_local(path)
        log_ret = prices_to_log_returns(prices).dropna(how="any")
        # Indice de mercado: equiponderado de log-returns como proxy
        market_ret = log_ret.mean(axis=1).values
        losses = -market_ret  # POT sobre perdidas (cola izquierda de retornos)
        kurt = float(stats.kurtosis(market_ret, fisher=False))  # 3 = gaussian
        skew = float(stats.skew(market_ret))

        # xi via POT con threshold al 95% (usando el fit_gpd del proyecto)
        try:
            gpd = fit_gpd(losses, quantile=0.95)
            shape = gpd.shape if gpd.converged else np.nan
            scale = gpd.scale if gpd.converged else np.nan
        except Exception:
            shape, scale = np.nan, np.nan

        rows.append({
            "market": market,
            "asset_class": ASSET_CLASS[market],
            "n_obs": len(market_ret),
            "kurtosis": round(kurt, 3),
            "skewness": round(skew, 3),
            "xi_pot_q95": round(shape, 4) if np.isfinite(shape) else None,
            "scale_pot_q95": round(scale, 4) if np.isfinite(scale) else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tail_stats_by_market.csv", index=False)
    return df


def regress_kurtosis_vs_effect(runs: pd.DataFrame,
                                tail_stats: pd.DataFrame,
                                out_dir: Path) -> str:
    """Regresion: efecto V4-V1 (mean por mercado) sobre kurtosis y ξ̂.

    Replica el espiritu del analisis de la seccion 5.3 (regresion OLS de
    ΔMDD sobre parametros de cola) pero a nivel mercado, no fold.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Calcular efecto medio V1-V4 por mercado (positivo = V4 mejor)
    effects = []
    for market in runs["market"].unique():
        sub = runs[runs["market"] == market]
        paired = paired_table(sub, baseline="V1", treatment="V4")
        if len(paired) >= 5:
            effects.append({
                "market": market,
                "delta_mdd_mean_pp": (paired["V4"] - paired["V1"]).mean() * 100,
                "delta_mdd_median_pp": (paired["V4"] - paired["V1"]).median() * 100,
                "n": len(paired),
            })
    eff_df = pd.DataFrame(effects)
    merged = eff_df.merge(tail_stats, on="market", how="inner")
    merged.to_csv(out_dir / "effects_with_tail_stats.csv", index=False)

    if len(merged) < 4:
        msg = f"Insuficientes mercados ({len(merged)}) para OLS"
        (out_dir / "ols_kurtosis_vs_effect.txt").write_text(msg)
        return msg

    # OLS simple: delta_mdd = a + b1 * kurtosis + b2 * xi
    import statsmodels.api as sm
    X = merged[["kurtosis", "xi_pot_q95"]].copy()
    X = sm.add_constant(X)
    y = merged["delta_mdd_mean_pp"]
    valid = X.notna().all(axis=1) & y.notna()
    if valid.sum() < 4:
        msg = "Insuficientes filas validas para OLS"
        (out_dir / "ols_kurtosis_vs_effect.txt").write_text(msg)
        return msg

    model = sm.OLS(y[valid], X[valid]).fit()
    summary = str(model.summary())
    (out_dir / "ols_kurtosis_vs_effect.txt").write_text(summary)
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_forest_v1_v4(by_class_df: pd.DataFrame, out_path: Path) -> None:
    """Forest plot del efecto V1-V4 por clase de activo, con CI 95%."""
    import matplotlib.pyplot as plt

    if by_class_df.empty:
        return

    # Ordenar por magnitud del efecto (mean_diff)
    df = by_class_df.copy()
    if "mean_diff" not in df.columns:
        return  # full_comparison no devolvio campo esperado
    df = df.sort_values("mean_diff")

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(df))))
    y_pos = np.arange(len(df))
    means = df["mean_diff"].values * 100  # a puntos porcentuales
    # full_comparison devuelve bootstrap_ci_lower y bootstrap_ci_upper
    if "bootstrap_ci_lower" in df.columns and "bootstrap_ci_upper" in df.columns:
        ci_low = df["bootstrap_ci_lower"].values * 100
        ci_high = df["bootstrap_ci_upper"].values * 100
        err_low = means - ci_low
        err_high = ci_high - means
        ax.errorbar(means, y_pos, xerr=[err_low, err_high],
                    fmt="o", capsize=4, color="C0", elinewidth=1.5)
    else:
        ax.plot(means, y_pos, "o", color="C0")

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["asset_class"].values)
    ax.set_xlabel("∆MDD = MDD(V4) − MDD(V1) (pp)")
    ax.set_title("Forest plot V4 − V1 por clase de activo (IC bootstrap 95%)")
    plt.tight_layout()
    fig.savefig(f"{out_path}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out_path}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_mdd_boxplot_by_class(runs: pd.DataFrame, out_path: Path) -> None:
    """Boxplot de MDD por clase de activo y variante (panel facetado)."""
    import matplotlib.pyplot as plt

    classes = sorted(runs["asset_class"].unique())
    variants = sorted(runs["variant"].unique())
    n_cls = len(classes)

    fig, axes = plt.subplots(1, n_cls, figsize=(4 * n_cls, 5),
                              sharey=False)
    if n_cls == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        sub = runs[runs["asset_class"] == cls]
        data = [sub[sub["variant"] == v]["mdd"].values for v in variants]
        ax.boxplot(data, labels=variants, showfliers=True)
        ax.set_title(cls, fontsize=10)
        ax.set_xlabel("Variante")
        ax.set_ylabel("MDD")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Distribución de MDD por clase de activo y variante",
                  y=1.02)
    plt.tight_layout()
    fig.savefig(f"{out_path}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out_path}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_paired_diff_by_class(runs: pd.DataFrame, out_path: Path) -> None:
    """Histograma overlapping de diferencias V4-V1 por clase."""
    import matplotlib.pyplot as plt

    classes = sorted(runs["asset_class"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    for cls in classes:
        sub = runs[runs["asset_class"] == cls]
        paired = paired_table(sub, baseline="V1", treatment="V4")
        if len(paired) < 5:
            continue
        diffs = (paired["V4"] - paired["V1"]).values * 100
        ax.hist(diffs, bins=20, alpha=0.45, label=f"{cls} (n={len(diffs)})",
                density=True)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("∆MDD = MDD(V4) − MDD(V1) (pp)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de diferencias pareadas V4 − V1 por clase")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_path}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out_path}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_kurtosis_vs_effect(runs: pd.DataFrame,
                              tail_stats: pd.DataFrame,
                              out_path: Path) -> None:
    """Scatter kurtosis vs efecto V1-V4 por mercado, coloreado por clase."""
    import matplotlib.pyplot as plt

    effects = []
    for market in runs["market"].unique():
        sub = runs[runs["market"] == market]
        paired = paired_table(sub, baseline="V1", treatment="V4")
        if len(paired) >= 5:
            effects.append({
                "market": market,
                "asset_class": ASSET_CLASS.get(market, "Unknown"),
                "delta_mdd_pp": (paired["V4"] - paired["V1"]).mean() * 100,
            })
    eff_df = pd.DataFrame(effects).merge(tail_stats, on="market", how="inner")
    if eff_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    classes = eff_df["asset_class_x"].unique() if "asset_class_x" in eff_df else \
              eff_df["asset_class"].unique()
    cls_col = "asset_class_x" if "asset_class_x" in eff_df.columns else "asset_class"

    for cls in classes:
        m = eff_df[eff_df[cls_col] == cls]
        ax.scatter(m["kurtosis"], m["delta_mdd_pp"], s=120,
                    label=cls, alpha=0.8, edgecolor="black")
        for _, row in m.iterrows():
            ax.annotate(row["market"],
                        (row["kurtosis"], row["delta_mdd_pp"]),
                        textcoords="offset points", xytext=(7, 4),
                        fontsize=9)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Kurtosis del retorno equiponderado del mercado")
    ax.set_ylabel("∆MDD medio = MDD(V4) − MDD(V1) (pp)")
    ax.set_title("Magnitud del efecto V4 frente a tail thickness por mercado\n"
                  "(validacion out-of-sample del hallazgo de la seccion 5.3)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_path}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out_path}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_per_fold_equity_extensions(results_dir: Path, runs: pd.DataFrame,
                                      out_dir: Path) -> None:
    """Equity curves por fold, solo para mercados de extension."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold_dir = results_dir / "per_fold"
    if not per_fold_dir.exists():
        return

    for market in EXTENSION_MARKETS:
        if market not in runs["market"].unique():
            continue
        market_dir = per_fold_dir / market
        if not market_dir.exists():
            continue
        for fold_dir in sorted(market_dir.glob("fold*")):
            series = {}
            for v in sorted(runs["variant"].unique()):
                f = fold_dir / f"{v}_seed0.npz"
                if f.exists():
                    arr = np.load(f)
                    series[v] = arr["values"]
            if series:
                P.plot_value_and_drawdown(
                    series,
                    title=f"{market} — {fold_dir.name} — equity & drawdown",
                    save_path=out_dir / f"equity_{market}_{fold_dir.name}",
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True,
                    help="Directorio con runs.parquet (ej: results/full_optimal/)")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="Directorio con los parquet de cada mercado")
    p.add_argument("--skip-equity", action="store_true",
                    help="Si se pasa, omite los plots/tablas de mercados originales "
                         "(util cuando solo quieres analizar extensiones)")
    args = p.parse_args()

    rd = args.results_dir
    runs_path = rd / "runs.parquet"
    if not runs_path.exists():
        raise SystemExit(f"No runs.parquet en {rd}")
    runs = pd.read_parquet(runs_path)
    runs = add_asset_class_column(runs)

    if args.skip_equity:
        runs = runs[runs["is_extension"]].copy()

    analysis_dir = rd / "analysis_sectoral"
    tables_dir = analysis_dir / "tables"
    plots_dir = analysis_dir / "plots"

    print(f"Cargados {len(runs)} runs ({runs['market'].nunique()} mercados, "
          f"{runs['asset_class'].nunique()} clases de activo)")
    print(f"  Por mercado: {runs['market'].value_counts().to_dict()}")

    # --- Tablas ---
    print("\n[1/7] Tabla agregada por clase de activo...")
    aggregate_by_asset_class(runs, tables_dir)

    print("[2/7] Tabla agregada por mercado dentro de cada clase...")
    aggregate_by_market(runs, tables_dir)

    print("[3/7] Tests pareados V1-V4 por mercado...")
    by_market = paired_tests_by_market(runs, tables_dir)
    print(by_market.to_string(index=False) if not by_market.empty else "  (vacio)")

    print("\n[4/7] Tests pareados V1-V4 por clase...")
    by_class = paired_tests_by_class(runs, tables_dir)
    print(by_class.to_string(index=False) if not by_class.empty else "  (vacio)")

    print("\n[5/7] Hipotesis tests V0-V1, V1-V2, ... en extensiones...")
    extension_hypothesis_tests(runs, tables_dir)

    print("[6/7] Tabla maestra para LaTeX...")
    master_table_for_latex(runs, tables_dir)

    print("[7/7] Estadisticos de cola por mercado y OLS kurtosis vs efecto...")
    try:
        tail_stats = compute_tail_stats_per_market(args.data_dir, tables_dir)
        ols_summary = regress_kurtosis_vs_effect(runs, tail_stats, tables_dir)
        print("  --- OLS ΔMDD ~ kurtosis + ξ̂ (por mercado) ---")
        # Imprimir solo las primeras lineas del summary
        for line in ols_summary.split("\n")[:20]:
            print(f"  {line}")
    except Exception as e:
        print(f"  AVISO: no se pudieron calcular tail stats: {e}")
        tail_stats = pd.DataFrame()

    # --- Plots ---
    print("\n[plots] Forest plot V1-V4 por clase...")
    plot_forest_v1_v4(by_class, plots_dir / "forest_plot_V1_V4_by_class")

    print("[plots] Boxplot MDD por clase x variante...")
    plot_mdd_boxplot_by_class(runs, plots_dir / "mdd_boxplot_by_asset_class")

    print("[plots] Histograma overlap diferencias V4-V1 por clase...")
    plot_paired_diff_by_class(runs, plots_dir / "paired_diff_by_class")

    if not tail_stats.empty:
        print("[plots] Scatter kurtosis vs efecto...")
        plot_kurtosis_vs_effect(runs, tail_stats,
                                  plots_dir / "scatter_kurtosis_vs_effect")

    print("[plots] Equity curves por fold de cada extension...")
    plot_per_fold_equity_extensions(rd, runs, plots_dir)

    print(f"\nTodos los outputs en: {analysis_dir}")


if __name__ == "__main__":
    main()
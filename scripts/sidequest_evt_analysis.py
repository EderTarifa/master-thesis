"""
Análisis EVT puro (sin RL).

Ejecuta los tres sub-experimentos del Grupo B en una sola pasada:

  B.1: Backtesting clásico de modelos de cola (VaR/CVaR empírico,
       Normal, GARCH-EVT) sobre el portfolio 1/N.
  B.2: Análisis de regímenes EVT por (mercado, fold). Permite clasificar
       los 13 folds en regímenes de cola distintos y justificar que EVT
       tiene material informativo en folds de crisis.
  B.3: Comparación POT (GPD) vs Block Maxima (GEV) para validar la
       elección metodológica de POT.

Uso:
    python scripts/sidequest_evt_analysis.py

Outputs en results/sidequest_evt/:
  - tail_backtest_<MARKET>.csv         (B.1)
  - regime_classification.csv          (B.2)
  - pot_vs_blockmaxima.csv             (B.3)
  - figs/*.png

Coste: ~30 min total. No requiere torch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from evt_ppo import data as D
from evt_ppo import evt as E
from evt_ppo import drawdown as DD
from evt_ppo.walkforward import WalkForwardConfig, generate_splits


# ---------------------------------------------------------------------------
# B.1: Tail backtesting (Kupiec + Christoffersen)
# ---------------------------------------------------------------------------


def kupiec_lrt(violations: np.ndarray, alpha: float) -> dict:
    """Test de Kupiec (cobertura incondicional).

    H0: la tasa observada de violaciones es alpha.
    """
    n = len(violations)
    x = int(violations.sum())
    p_hat = x / n if n > 0 else 0.0
    if p_hat in (0.0, 1.0) or alpha in (0.0, 1.0):
        return {"violations": x, "n": n, "rate": p_hat,
                "expected_rate": alpha, "stat": np.nan, "pvalue": np.nan}
    # Likelihood ratio
    lr = -2 * (
        x * np.log(alpha) + (n - x) * np.log(1 - alpha)
        - x * np.log(p_hat) - (n - x) * np.log(1 - p_hat)
    )
    pvalue = 1 - stats.chi2.cdf(lr, df=1)
    return {"violations": x, "n": n, "rate": p_hat,
            "expected_rate": alpha, "stat": float(lr),
            "pvalue": float(pvalue)}


def christoffersen_independence(violations: np.ndarray) -> dict:
    """Test de independencia de Christoffersen.

    H0: las violaciones son independientes.
    """
    n = len(violations)
    if n < 2:
        return {"stat": np.nan, "pvalue": np.nan}
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        a, b = int(violations[i-1]), int(violations[i])
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi  = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    if pi in (0.0, 1.0) or pi0 in (0.0, 1.0) or pi1 in (0.0, 1.0):
        return {"stat": np.nan, "pvalue": np.nan}
    lr = -2 * (
        (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        - n00 * np.log(1 - pi0) - n01 * np.log(pi0)
        - n10 * np.log(1 - pi1) - n11 * np.log(pi1)
    )
    pvalue = 1 - stats.chi2.cdf(lr, df=1)
    return {"stat": float(lr), "pvalue": float(pvalue)}


def normal_var_cvar(returns: np.ndarray, alpha: float) -> tuple[float, float]:
    """VaR/CVaR paramétrico Normal."""
    mu, sd = float(returns.mean()), float(returns.std(ddof=1))
    z_alpha = stats.norm.ppf(alpha)
    var = -(mu + sd * z_alpha)  # convertir a pérdida (positiva)
    pdf_z = stats.norm.pdf(z_alpha)
    cvar = -(mu - sd * pdf_z / (1 - alpha))
    return float(var), float(cvar)


def empirical_var_cvar(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """VaR/CVaR empírico no paramétrico."""
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size > 0 else var
    return var, cvar


def evt_var_cvar(losses: np.ndarray, alpha: float,
                  threshold_quantile: float = 0.90) -> tuple[float, float]:
    """VaR/CVaR mediante POT-GPD."""
    fit = E.fit_gpd(losses, quantile=threshold_quantile)
    return E.gpd_var(fit, alpha), E.gpd_cvar(fit, alpha)


def b1_tail_backtest(market: str, prices: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Backtest de modelos de cola sobre el portfolio 1/N rolling."""
    log_ret = np.log(prices / prices.shift(1)).dropna(how="any")
    portfolio_log = log_ret.mean(axis=1).values
    losses = -portfolio_log

    rows = []
    window = 500            # ventana de calibración rolling
    horizon = len(losses) - window

    for alpha in (0.95, 0.99):
        violations_emp = np.zeros(horizon, dtype=int)
        violations_norm = np.zeros(horizon, dtype=int)
        violations_evt = np.zeros(horizon, dtype=int)

        for t in range(horizon):
            insample = losses[t:t+window]
            actual_loss = losses[t+window]
            var_e, _ = empirical_var_cvar(insample, alpha)
            var_n, _ = normal_var_cvar(insample, alpha)
            var_v, _ = evt_var_cvar(insample, alpha)
            violations_emp[t] = int(actual_loss > var_e)
            violations_norm[t] = int(actual_loss > var_n)
            violations_evt[t] = int(actual_loss > var_v)

        for name, viol in (("Empirical", violations_emp),
                            ("Normal", violations_norm),
                            ("EVT_POT", violations_evt)):
            kup = kupiec_lrt(viol, 1 - alpha)
            chr_ = christoffersen_independence(viol)
            rows.append({
                "market": market,
                "model": name,
                "alpha": alpha,
                **{f"kupiec_{k}": v for k, v in kup.items()},
                "chr_stat": chr_["stat"],
                "chr_pvalue": chr_["pvalue"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"tail_backtest_{market}.csv", index=False)
    print(f"[B.1] {market}: backtest done, {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# B.2: Regime classification per fold
# ---------------------------------------------------------------------------


def b2_regime_classification(prices_dict: dict[str, pd.DataFrame],
                              wf: WalkForwardConfig,
                              out_dir: Path) -> pd.DataFrame:
    """Clasifica cada (market, fold) por su régimen de cola."""
    rows = []
    for market, prices in prices_dict.items():
        log_ret = np.log(prices / prices.shift(1)).dropna(how="any")
        portfolio_log = log_ret.mean(axis=1)
        portfolio_val = 100.0 * np.exp(portfolio_log.cumsum())
        portfolio_val = pd.concat([
            pd.Series([100.0], index=[portfolio_log.index[0] - pd.Timedelta(days=1)]),
            portfolio_val,
        ])
        splits = generate_splits(log_ret.index, wf)

        for split in splits:
            # Datos del periodo TEST
            test_log = portfolio_log.loc[split.test_start:split.test_end].values
            test_val = portfolio_val.loc[split.test_start:split.test_end].values
            if len(test_log) < 100:
                continue
            # Pérdidas
            losses_test = -test_log
            mdd_test = DD.maximum_drawdown(test_val)

            # Estadísticos EVT sobre el periodo de test
            threshold, fit = E.select_threshold_auto(losses_test)
            xi = fit.shape
            sigma = fit.scale
            cvar99 = E.gpd_cvar(fit, 0.99)

            # Estadísticos descriptivos
            skew = float(stats.skew(test_log))
            kurt = float(stats.kurtosis(test_log))
            vol = float(test_log.std() * np.sqrt(252))

            # Clasificación de régimen por MDD del 1/N
            if mdd_test < 0.10:
                regime = "benign"
            elif mdd_test < 0.20:
                regime = "moderate"
            else:
                regime = "crisis"

            rows.append({
                "market": market, "fold": split.fold_id,
                "test_year": split.test_start.year,
                "regime": regime,
                "mdd_1N": float(mdd_test),
                "vol_annualised": vol,
                "skewness": skew,
                "kurtosis": kurt,
                "evt_xi": float(xi),
                "evt_sigma": float(sigma),
                "evt_cvar99": float(cvar99),
                "evt_threshold": float(threshold),
                "evt_n_excesses": fit.n_excesses,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "regime_classification.csv", index=False)
    print(f"[B.2] regime classification done, {len(df)} folds")

    # Plot: heatmap xi por (market, year)
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot = df.pivot_table(values="evt_xi", index="market",
                            columns="test_year")
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                    vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label="xi (shape)")
    ax.set_title("EVT shape parameter (xi) by market and test year\n"
                  "Red = heavy tail, Blue = bounded tail")
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "regime_xi_heatmap.png", dpi=150)
    plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# B.3: POT vs Block Maxima
# ---------------------------------------------------------------------------


def b3_pot_vs_blockmaxima(prices_dict: dict[str, pd.DataFrame],
                           wf: WalkForwardConfig,
                           out_dir: Path) -> pd.DataFrame:
    """Compara empíricamente POT (GPD) vs Block Maxima (GEV)."""
    rows = []
    for market, prices in prices_dict.items():
        log_ret = np.log(prices / prices.shift(1)).dropna(how="any")
        portfolio_log = log_ret.mean(axis=1).values
        losses = -portfolio_log

        splits = generate_splits(log_ret.index, wf)
        for split in splits:
            train_idx = log_ret.index.get_indexer(
                log_ret.loc[split.train_start:split.val_end].index
            )
            test_idx = log_ret.index.get_indexer(
                log_ret.loc[split.test_start:split.test_end].index
            )
            train_losses = losses[train_idx]
            test_losses = losses[test_idx]
            if len(train_losses) < 200 or len(test_losses) < 50:
                continue

            # Ajustes
            _, fit_pot = E.select_threshold_auto(train_losses)
            fit_gev = E.fit_gev_block_maxima(train_losses, block_size=5)

            # Predicción out-of-sample en cuantiles altos
            for alpha in (0.95, 0.99):
                # POT
                var_pot = E.gpd_var(fit_pot, alpha)
                # GEV: VaR aproximado (cuantil de la GEV con frecuencia anual)
                # GEV está en escala de máximos por bloque; la conversión a cuantil
                # de la distribución original usa la relación n_blocks * (1-alpha)
                # de eventos en un año; para fines comparativos calculamos
                # la cuantil empírica en el bloque maximo
                if abs(fit_gev.shape) < 1e-8:
                    var_gev = (fit_gev.location
                                - fit_gev.scale * np.log(-np.log(alpha)))
                else:
                    var_gev = (fit_gev.location
                                + fit_gev.scale / fit_gev.shape
                                * ((-np.log(alpha))**(-fit_gev.shape) - 1))

                # Violaciones empíricas
                vio_pot = float((test_losses > var_pot).mean())
                vio_gev = float((test_losses > var_gev).mean())

                rows.append({
                    "market": market, "fold": split.fold_id, "alpha": alpha,
                    "pot_xi": float(fit_pot.shape),
                    "pot_sigma": float(fit_pot.scale),
                    "pot_var": float(var_pot),
                    "pot_violation_rate": vio_pot,
                    "pot_expected_rate": 1 - alpha,
                    "gev_shape": float(fit_gev.shape),
                    "gev_location": float(fit_gev.location),
                    "gev_scale": float(fit_gev.scale),
                    "gev_var": float(var_gev),
                    "gev_violation_rate": vio_gev,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pot_vs_blockmaxima.csv", index=False)
    print(f"[B.3] POT vs Block Maxima: {len(df)} rows")

    # Plot: violation rates
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, alpha in zip(axes, (0.95, 0.99)):
        sub = df[df["alpha"] == alpha]
        ax.scatter(sub["pot_violation_rate"], sub["gev_violation_rate"],
                    alpha=0.6)
        ax.axhline(1 - alpha, color="red", ls="--", lw=1,
                    label=f"expected = {1-alpha}")
        ax.axvline(1 - alpha, color="red", ls="--", lw=1)
        ax.plot([0, 0.2], [0, 0.2], "k:", alpha=0.3)
        ax.set_xlabel("POT (GPD) violation rate")
        ax.set_ylabel("Block Maxima (GEV) violation rate")
        ax.set_title(f"alpha = {alpha}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "pot_vs_gev_violations.png", dpi=150)
    plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path("results/sidequest_evt")
    figs_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Side-quest EVT analysis (Group B)")
    print("=" * 70)

    # Cargar datos
    prices_dict = {}
    for m in ("DJIA", "SP50", "IBEX"):
        path = Path(f"data/{m}.parquet")
        if not path.exists():
            print(f"[!] {m} data missing, skipping")
            continue
        prices_dict[m] = D.load_local(path)
        print(f"Loaded {m}: {prices_dict[m].shape}")

    if not prices_dict:
        raise SystemExit("No data found. Run `make data` first.")

    wf = WalkForwardConfig(train_years=4, val_years=1,
                            test_years=1, step_years=1)

    # B.1
    print("\n--- B.1: Tail model backtesting ---")
    b1_results = []
    for market, prices in prices_dict.items():
        b1_results.append(b1_tail_backtest(market, prices, out_dir))
    b1_full = pd.concat(b1_results, ignore_index=True)
    b1_full.to_csv(out_dir / "tail_backtest_all.csv", index=False)

    # B.2
    print("\n--- B.2: Regime classification per fold ---")
    b2 = b2_regime_classification(prices_dict, wf, out_dir)
    print("Regime distribution:")
    print(b2["regime"].value_counts())

    # B.3
    print("\n--- B.3: POT vs Block Maxima ---")
    b3 = b3_pot_vs_blockmaxima(prices_dict, wf, out_dir)

    print("\n" + "=" * 70)
    print(f"All side-quest EVT outputs saved to: {out_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
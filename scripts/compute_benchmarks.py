"""
Cómputo de benchmarks clásicos sobre la cuadrícula walk-forward del TFM.

Calcula cuatro estrategias clásicas y reporta las mismas métricas que el
agente DRL para comparación directa:

  1. Equiponderado 1/N rebalanceado mensualmente.
  2. Mínima varianza (long-only) con rebalanceo mensual.
  3. Markowitz mean-variance con gamma=5, long-only, rebalanceo mensual.
  4. Buy-and-hold equiponderado (sin rebalanceo).

Para cada combinación (market, fold) calcula MDD, CAGR, vol anualizada,
Sharpe, Sortino, Calmar, CVaR_95, CVaR_99, dd_q95, CDaR_95, turnover medio.

Uso:
    python scripts/compute_benchmarks.py
        --markets DJIA SP50 IBEX
        --out results/benchmarks
        --tcost 0.001

Outputs en results/benchmarks/:
  - rows/<MARKET>_f<FOLD>_<STRATEGY>.parquet  (uno por configuración)
  - runs.parquet                               (concatenado)
  - analysis/tables/agg_by_strategy.csv        (medias y std por estrategia)
  - analysis/tables/agg_by_market_strategy.csv (medias por mercado x estrategia)
  - analysis/tables/v4_vs_benchmarks.csv       (contraste V4 contra cada benchmark)
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from evt_ppo.data import load_local, prices_to_log_returns
from evt_ppo.walkforward import WalkForwardConfig, generate_splits

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estrategias
# ---------------------------------------------------------------------------


def equal_weight(N: int) -> np.ndarray:
    """Pesos iguales en N activos."""
    return np.ones(N) / N


def min_variance(returns_train: np.ndarray) -> np.ndarray:
    """Cartera de mínima varianza long-only.

    returns_train: matriz (T, N) de log-returns sobre la ventana de train.
    Resuelve min w'Sigma w sujeto a w_i >= 0, sum(w) = 1.
    """
    cov = np.cov(returns_train, rowvar=False)
    n = cov.shape[0]
    w0 = np.ones(n) / n

    def obj(w):
        return float(w @ cov @ w)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bnds = tuple((0.0, 1.0) for _ in range(n))

    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    if not res.success:
        log.warning(f"min_variance failed: {res.message}; using 1/N")
        return w0
    return np.asarray(res.x)


def mean_variance(returns_train: np.ndarray, gamma: float = 5.0) -> np.ndarray:
    """Cartera mean-variance long-only.

    Resuelve max w'mu - gamma/2 w'Sigma w sujeto a w_i >= 0, sum(w) = 1.
    Anualiza la media para que gamma sea comparable a la práctica.
    """
    mu = returns_train.mean(axis=0) * 252.0
    cov = np.cov(returns_train, rowvar=False) * 252.0
    n = cov.shape[0]
    w0 = np.ones(n) / n

    def neg_utility(w):
        return float(-w @ mu + 0.5 * gamma * w @ cov @ w)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bnds = tuple((0.0, 1.0) for _ in range(n))

    res = minimize(neg_utility, w0, method="SLSQP",
                    bounds=bnds, constraints=cons,
                    options={"maxiter": 500, "ftol": 1e-9})
    if not res.success:
        log.warning(f"mean_variance failed: {res.message}; using 1/N")
        return w0
    return np.asarray(res.x)


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------


def simulate_strategy(
    log_ret_train: np.ndarray,
    log_ret_test: np.ndarray,
    strategy: str,
    rebalance_every: int = 21,
    transaction_cost: float = 0.001,
    initial_capital: float = 100.0,
) -> dict:
    """Simula una estrategia clásica sobre el periodo de test.

    Parameters
    ----------
    log_ret_train : (T_train, N) array de log-returns para calibrar la estrategia.
    log_ret_test : (T_test, N) array para evaluar.
    strategy : 'equal_weight', 'min_variance', 'mean_variance', 'buy_and_hold_equal'.
    rebalance_every : pasos entre rebalanceos (default 21 = mensual).
    transaction_cost : coste proporcional al turnover.
    initial_capital : capital inicial.

    Returns
    -------
    dict con métricas: mdd, cagr, vol_annualised, sharpe, sortino, calmar,
    cvar_95, cvar_99, var_95, dd_q95, cdar_95, turnover_mean.
    """
    T_test, N = log_ret_test.shape
    simple_ret_test = np.exp(log_ret_test) - 1.0

    # Calcular pesos objetivo según estrategia
    def compute_target(window_ret: np.ndarray) -> np.ndarray:
        if strategy == "equal_weight":
            return equal_weight(N)
        elif strategy == "min_variance":
            return min_variance(window_ret)
        elif strategy == "mean_variance":
            return mean_variance(window_ret, gamma=5.0)
        elif strategy == "buy_and_hold_equal":
            return equal_weight(N)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # Inicialización
    if strategy == "buy_and_hold_equal":
        w_target = equal_weight(N)
    else:
        w_target = compute_target(log_ret_train)

    w = w_target.copy()  # pesos efectivos actuales

    values = [initial_capital]
    weights_history = [w.copy()]
    turnover_total = 0.0
    n_rebalances = 0

    for t in range(T_test):
        # Rebalanceo si toca y la estrategia no es buy-and-hold
        if (strategy != "buy_and_hold_equal"
            and t > 0 and t % rebalance_every == 0):
            # Recalcular target con datos hasta este punto
            window = np.vstack([log_ret_train, log_ret_test[:t]])
            # Tomar últimos T_train pasos (rolling)
            T_train_eff = log_ret_train.shape[0]
            window = window[-T_train_eff:]
            try:
                w_target = compute_target(window)
            except Exception:
                pass  # mantener pesos previos si falla
            turnover = float(np.sum(np.abs(w_target - w)))
            turnover_total += turnover
            n_rebalances += 1
            # Aplicar coste de transacción
            cost = transaction_cost * turnover
            current_value = values[-1] * (1 - cost)
            values[-1] = current_value
            w = w_target.copy()

        # Aplicar retorno del paso
        portfolio_return = float(w @ simple_ret_test[t])
        new_value = values[-1] * (1 + portfolio_return)
        values.append(new_value)

        # Actualizar pesos efectivos por drift
        if not np.isclose(1 + portfolio_return, 0):
            w_new = w * (1 + simple_ret_test[t]) / (1 + portfolio_return)
            w = w_new
        weights_history.append(w.copy())

    values = np.asarray(values)
    weights_history = np.asarray(weights_history)

    # Métricas
    return _compute_metrics(values, weights_history, log_ret_test,
                             turnover_total, n_rebalances, T_test)


def _compute_metrics(values, weights, log_ret_test,
                      turnover_total, n_rebalances, T_test) -> dict:
    """Computa todas las métricas de evaluación."""
    # MDD
    running_max = np.maximum.accumulate(values)
    drawdowns = (running_max - values) / running_max
    mdd = float(drawdowns.max())

    # Returns
    log_returns = np.diff(np.log(values))
    if len(log_returns) == 0:
        return {"mdd": np.nan, "cagr": np.nan}

    days_per_year = 252
    n_years = T_test / days_per_year
    total_return = values[-1] / values[0] - 1
    cagr = float((values[-1] / values[0]) ** (1 / max(n_years, 1e-6)) - 1)
    vol = float(log_returns.std() * np.sqrt(days_per_year))
    mean_log = float(log_returns.mean() * days_per_year)
    sharpe = mean_log / vol if vol > 0 else 0.0

    # Sortino
    downside = log_returns[log_returns < 0]
    downside_std = float(downside.std() * np.sqrt(days_per_year)) if len(downside) > 0 else 0.0
    sortino = mean_log / downside_std if downside_std > 0 else 0.0

    # Calmar
    calmar = cagr / mdd if mdd > 0 else 0.0

    # CVaR de retornos (simples)
    simple_returns = np.exp(log_returns) - 1
    losses = -simple_returns
    losses_sorted = np.sort(losses)
    var_95 = float(np.quantile(losses, 0.95))
    cvar_95 = float(losses[losses >= var_95].mean()) if (losses >= var_95).any() else var_95
    cvar_99 = float(losses[losses >= np.quantile(losses, 0.99)].mean()) \
        if (losses >= np.quantile(losses, 0.99)).any() else float(np.quantile(losses, 0.99))

    # Drawdown stats
    dd_q95 = float(np.quantile(drawdowns, 0.95))
    dd_above_q95 = drawdowns[drawdowns >= dd_q95]
    cdar_95 = float(dd_above_q95.mean()) if len(dd_above_q95) > 0 else dd_q95

    # Turnover medio (por rebalanceo, no por paso)
    turnover_mean = (turnover_total / max(n_rebalances, 1)) if n_rebalances > 0 else 0.0

    return {
        "mdd": mdd,
        "cagr": cagr,
        "vol_annualised": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "total_return": float(total_return),
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "var_95": var_95,
        "dd_q95": dd_q95,
        "cdar_95": cdar_95,
        "turnover_mean": turnover_mean,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_all_benchmarks(markets, out_dir, tcost):
    """Itera sobre (market, fold, strategy) y guarda parquets individuales."""
    rows_dir = out_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    strategies = ["equal_weight", "min_variance", "mean_variance",
                   "buy_and_hold_equal"]
    wf = WalkForwardConfig(train_years=4, val_years=1,
                            test_years=1, step_years=1)

    rows = []
    for market in markets:
        path = Path(f"data/{market}.parquet")
        if not path.exists():
            log.warning(f"{market} data not found, skipping")
            continue
        prices = load_local(path)
        log_ret = prices_to_log_returns(prices).dropna(how="any")
        splits = generate_splits(log_ret.index, wf)
        log.info(f"{market}: {len(splits)} folds, {prices.shape[1]} assets")

        for split in splits:
            train = log_ret.loc[split.train_start:split.val_end].values
            test = log_ret.loc[split.test_start:split.test_end].values
            if len(train) < 100 or len(test) < 50:
                continue
            for strategy in strategies:
                marker = rows_dir / f"{market}_f{split.fold_id:02d}_{strategy}.parquet"
                if marker.exists():
                    continue

                metrics = simulate_strategy(
                    train, test, strategy,
                    rebalance_every=21,
                    transaction_cost=tcost,
                )
                row = {
                    "market": market,
                    "fold": split.fold_id,
                    "strategy": strategy,
                    "test_start": split.test_start,
                    "test_end": split.test_end,
                    **metrics,
                }
                pd.DataFrame([row]).to_parquet(marker)
                rows.append(row)
                log.info(f"  {market} f{split.fold_id} {strategy}: "
                          f"MDD={metrics['mdd']:.4f}, "
                          f"Calmar={metrics['calmar']:.3f}")

    # Concatenar y guardar
    files = sorted(rows_dir.glob("*.parquet"))
    if files:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df.to_parquet(out_dir / "runs.parquet")
        log.info(f"Total {len(df)} benchmark runs -> runs.parquet")
        return df
    return pd.DataFrame()


def analyze(df_bench: pd.DataFrame, results_dir: Path,
             include_v4: bool = True) -> None:
    """Genera tablas de análisis comparando con V4."""
    out_dir = results_dir / "analysis" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Agregado por estrategia
    agg_by_strat = df_bench.groupby("strategy").agg(
        mdd_mean=("mdd", "mean"),
        mdd_std=("mdd", "std"),
        cagr_mean=("cagr", "mean"),
        sharpe_mean=("sharpe", "mean"),
        sortino_mean=("sortino", "mean"),
        calmar_mean=("calmar", "mean"),
        cdar_95_mean=("cdar_95", "mean"),
        turnover_mean=("turnover_mean", "mean"),
        n=("mdd", "count"),
    ).round(4).reset_index()
    agg_by_strat.to_csv(out_dir / "agg_by_strategy.csv", index=False)
    print("\n=== Benchmarks aggregated by strategy ===")
    print(agg_by_strat.to_string(index=False))

    # Por mercado x estrategia
    agg_market = df_bench.groupby(["market", "strategy"]).agg(
        mdd_mean=("mdd", "mean"),
        cagr_mean=("cagr", "mean"),
        calmar_mean=("calmar", "mean"),
        n=("mdd", "count"),
    ).round(4).reset_index()
    agg_market.to_csv(out_dir / "agg_by_market_strategy.csv", index=False)

    # Si incluimos V4 del experimento full_optimal
    if include_v4:
        v4_path = Path("results/full_optimal/rows")
        if v4_path.exists():
            v4_files = sorted(v4_path.glob("*_V4_*.parquet"))
            if v4_files:
                df_v4 = pd.concat([pd.read_parquet(f) for f in v4_files],
                                    ignore_index=True)
                v4_metrics = df_v4.agg({
                    "mdd": "mean",
                    "cagr": "mean",
                    "sharpe": "mean",
                    "sortino": "mean",
                    "calmar": "mean",
                    "cdar_95": "mean",
                    "turnover_mean": "mean",
                }).round(4).to_dict()
                v4_metrics["strategy"] = "V4 (DRL+EVT)"
                v4_metrics["n"] = len(df_v4)

                # Tabla comparativa
                comp = agg_by_strat.copy()
                # Renombrar columnas para alinear
                comp_renamed = comp.rename(columns={
                    "mdd_mean": "mdd",
                    "cagr_mean": "cagr",
                    "sharpe_mean": "sharpe",
                    "sortino_mean": "sortino",
                    "calmar_mean": "calmar",
                    "cdar_95_mean": "cdar_95",
                })

                v4_row = pd.DataFrame([v4_metrics])
                comp_full = pd.concat(
                    [comp_renamed[["strategy", "mdd", "cagr", "sharpe",
                                     "sortino", "calmar", "cdar_95",
                                     "turnover_mean", "n"]],
                     v4_row[["strategy", "mdd", "cagr", "sharpe",
                              "sortino", "calmar", "cdar_95",
                              "turnover_mean", "n"]]],
                    ignore_index=True,
                )
                comp_full.to_csv(out_dir / "v4_vs_benchmarks.csv", index=False)

                print("\n=== V4 vs Benchmarks comparison ===")
                print(comp_full.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--markets", nargs="+",
                    default=["DJIA", "SP50", "IBEX"])
    p.add_argument("--out", default="results/benchmarks")
    p.add_argument("--tcost", type=float, default=0.001)
    args = p.parse_args()

    out_dir = Path(args.out)
    df = run_all_benchmarks(args.markets, out_dir, args.tcost)
    if not df.empty:
        analyze(df, out_dir, include_v4=True)
    print(f"\nAll outputs under: {out_dir}/")


if __name__ == "__main__":
    main()
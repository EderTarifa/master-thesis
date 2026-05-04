"""
Market data download and cleaning.

Data source: Yahoo Finance via the `yfinance` package.

Universes (component lists are fixed snapshots; in production research you
would use point-in-time membership to avoid survivorship bias):

- DJIA: 30 components (US large cap, blue chips).
- SP50: 50 most-liquid S&P 500 components (subset to keep dimensionality
  manageable; full S&P 500 is 500 names).
- IBEX: ~35 components of the IBEX 35 index (Spain).

Usage
-----
    from evt_ppo.data import download_universe, load_local
    df = download_universe("DJIA", start="2008-01-01", end="2025-12-31")
    df.to_parquet("data/djia.parquet")

The frame returned has a DatetimeIndex and columns = tickers, with
adjusted close prices. Downstream code converts to log-returns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Universe definitions
# ---------------------------------------------------------------------------

DJIA_TICKERS: tuple[str, ...] = (
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT", "XOM",
)

# 50 large, liquid S&P 500 components; sectorially diversified.
SP50_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "UNH", "JNJ",
    "V", "XOM", "WMT", "JPM", "MA", "PG", "HD", "CVX", "LLY", "ABBV",
    "BAC", "PFE", "AVGO", "KO", "PEP", "TMO", "COST", "MRK", "DIS", "ABT",
    "CSCO", "DHR", "ACN", "MCD", "VZ", "ADBE", "WFC", "NEE", "PM", "TXN",
    "BMY", "CRM", "RTX", "T", "QCOM", "UPS", "HON", "LIN", "ORCL", "INTC",
)

# IBEX 35 components (representative snapshot; suffix .MC for Madrid Stock Exchange).
IBEX_TICKERS: tuple[str, ...] = (
    "ACS.MC", "ACX.MC", "AENA.MC", "AMS.MC", "ANA.MC", "BBVA.MC", "BKT.MC",
    "CABK.MC", "CLNX.MC", "COL.MC", "ELE.MC", "ENG.MC", "FDR.MC", "FER.MC",
    "GRF.MC", "IAG.MC", "IBE.MC", "IDR.MC", "ITX.MC", "LOG.MC", "MAP.MC",
    "MEL.MC", "MRL.MC", "MTS.MC", "NTGY.MC", "RED.MC", "REP.MC", "ROVI.MC",
    "SAB.MC", "SAN.MC", "SCYR.MC", "SLR.MC", "TEF.MC", "UNI.MC", "VIS.MC",
)


UNIVERSES: dict[str, tuple[str, ...]] = {
    "DJIA": DJIA_TICKERS,
    "SP50": SP50_TICKERS,
    "IBEX": IBEX_TICKERS,
}


# ---------------------------------------------------------------------------
# Download and cleaning
# ---------------------------------------------------------------------------


def download_universe(
    universe: str,
    start: str = "2008-01-01",
    end: str = "2025-12-31",
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """Download adjusted close prices for a named universe.

    Requires the `yfinance` package and an internet connection.
    Returns a DataFrame indexed by trading day, with one column per ticker.
    Tickers that fail to download are silently dropped from the output.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required for live downloads. "
            "Install with `pip install yfinance` or use the synthetic data generator."
        ) from e

    if universe not in UNIVERSES:
        raise KeyError(f"Unknown universe '{universe}'. Choices: {list(UNIVERSES)}")
    tickers = list(UNIVERSES[universe])

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=auto_adjust,
        progress=progress, group_by="ticker", threads=True,
    )

    # When auto_adjust=True and group_by='ticker', columns are MultiIndex
    # (ticker, field). We extract 'Close' for each ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame({
            t: raw[t]["Close"] for t in tickers
            if t in raw.columns.get_level_values(0)
        })
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    return clean_prices(prices)


def clean_prices(prices: pd.DataFrame, max_missing_frac: float = 0.05) -> pd.DataFrame:
    """Drop tickers with too many missing values, forward-fill the rest.

    Parameters
    ----------
    max_missing_frac : float
        Maximum allowed fraction of missing days for a ticker to be kept.
    """
    prices = prices.sort_index()
    missing_frac = prices.isna().mean()
    keep = missing_frac[missing_frac <= max_missing_frac].index
    cleaned = prices[keep].ffill().bfill()
    cleaned = cleaned.dropna(how="any")
    return cleaned


def prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert prices to daily log-returns. Drops the first row."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_local(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Synthetic data generator (for offline testing of the full pipeline)
# ---------------------------------------------------------------------------


def synthetic_market(
    n_assets: int = 30,
    n_days: int = 4500,
    start: str = "2008-01-01",
    seed: int = 0,
    crisis_periods: Optional[Sequence[tuple[int, int, float]]] = None,
) -> pd.DataFrame:
    """Generate a synthetic price panel with realistic features:

    - Heavy tails (Student-t with df=5 innovations).
    - Cross-asset correlation via a low-rank factor structure.
    - Volatility clustering via a simple GARCH(1,1) on each asset.
    - Optional crisis periods (drawdown-inducing regime shifts).

    Returns a DataFrame with DatetimeIndex of business days and columns
    'A0', 'A1', ..., 'A{n_assets-1}'.

    This generator is intended for pipeline smoke tests when an internet
    connection is not available. **Real experiments must use Yahoo data.**
    """
    rng = np.random.default_rng(seed)
    if crisis_periods is None:
        # 3 simulated crises with mean-shift down for ~30 days each.
        crisis_periods = [
            (300, 360, -0.005),
            (1800, 1900, -0.008),
            (3500, 3600, -0.006),
        ]

    # Factor structure: 3 latent factors loaded by all assets.
    n_factors = 3
    loadings = rng.normal(0.0, 0.4, size=(n_assets, n_factors))
    base_drift = rng.uniform(0.0001, 0.0005, size=n_assets)

    # GARCH(1,1) per asset, in standardised innovation space.
    # Targeting unconditional vol around 0.012 (~19% annualised).
    target_var = 0.012**2
    omega = target_var * (1 - 0.05 - 0.92)
    alpha = 0.05
    beta = 0.92
    sigma2 = np.full(n_assets, target_var)

    returns = np.zeros((n_days, n_assets))
    # Student-t standardisation factor so var(z) = 1 with df=5.
    df = 5.0
    t_scale = np.sqrt((df - 2.0) / df)

    for t in range(n_days):
        # Common factor shocks (Student-t, df=5, standardised, scaled to ~1% vol).
        f_shocks = (rng.standard_t(df=df, size=n_factors) * t_scale) * 0.008
        # Idiosyncratic standardised shocks.
        z = rng.standard_t(df=df, size=n_assets) * t_scale
        idio = z * np.sqrt(sigma2)
        ret = base_drift + loadings @ f_shocks + idio
        # Cap returns to prevent numerical instability under fat tails.
        ret = np.clip(ret, -0.20, 0.20)

        # Apply crisis shifts (additional negative drift).
        for (t0, t1, mu_shift) in crisis_periods:
            if t0 <= t <= t1:
                ret = ret + mu_shift

        returns[t] = ret
        # Update variance using the idiosyncratic shock only
        # (factor risk is shared, not asset-specific).
        sigma2 = omega + alpha * idio**2 + beta * sigma2
        sigma2 = np.clip(sigma2, 1e-8, 0.01)  # cap to avoid runaway

    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    dates = pd.bdate_range(start=start, periods=n_days)
    return pd.DataFrame(prices, index=dates,
                        columns=[f"A{i}" for i in range(n_assets)])

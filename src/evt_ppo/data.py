"""
Market data download and cleaning.

Data source: Yahoo Finance via the `yfinance` package.

Universes (component lists are fixed snapshots; in production research you
would use point-in-time membership to avoid survivorship bias):

- DJIA: 30 components (US large cap, blue chips).
- SP50: 50 most-liquid S&P 500 components (subset to keep dimensionality
  manageable; full S&P 500 is 500 names).
- IBEX: ~35 components of the IBEX 35 index (Spain).
- BRD_CMDY: universo broad-commodity vía ETFs líquidos cubriendo cuatro 
    sub-sectores (energía, metales preciosos, metales industriales, agricultura)
    y agregados.
- CRYPTO: 14 cryptocurrencies con histórico fiable desde 2017-01-01.
    Excluimos stablecoins (USDT, USDC) por construcción y altcoins post-2018
    (SOL, ADA mainnet 2020, AVAX, MATIC, etc.)
- BOND_US: universo de fixed-income vía ETFs cubriendo curva de Treasuries,
    crédito IG/HY, TIPS, agregados y deuda emergente.
- FX_MIX: universo de divisas con mezcla de G10 majors, crosses, emergentes y asiáticas.

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

HSI_TICKERS: tuple[str, ...] = (
    "0001.HK", "0002.HK", "0003.HK", "0005.HK", "0006.HK",
    "0012.HK", "0016.HK", "0017.HK", "0019.HK",
    "0023.HK", "0066.HK", "0083.HK", "0101.HK", "0151.HK",
    "0175.HK", "0267.HK", "0291.HK", "0293.HK", "0386.HK",
    "0388.HK", "0688.HK", "0700.HK", "0762.HK", "0857.HK",
    "0883.HK", "0939.HK", "0941.HK", "0992.HK", "1038.HK",
    "1088.HK", "1113.HK", "1398.HK", "2318.HK", "2388.HK",
    "2628.HK", "3328.HK", "3988.HK",
)

BRD_CMDY_TICKERS: tuple[str, ...] = (
    # Energía
    "USO",    # WTI crude oil
    "BNO",    # Brent crude oil
    "UNG",    # Henry Hub natural gas
    "UGA",    # gasoline RBOB
    "UHN",    # heating oil / ULSD
    "USL",    # 12-month WTI (mitigates contango)
    "DBO",    # oil oversampled (rolling)
    # Metales preciosos
    "GLD",    # gold (SPDR)
    "SLV",    # silver
    "PPLT",   # platinum
    "PALL",   # palladium
    # Metales industriales
    "CPER",   # copper
    "DBB",    # base metals broad
    # Agricultura — soft and grains
    "CORN",   # corn
    "WEAT",   # wheat
    "SOYB",   # soybeans
    "CANE",   # sugar
    "DBA",    # agriculture broad
    # Broad / diversificados (sirven de "market proxies")
    "DBC",    # Invesco DB Commodity Index Tracking
    "GSG",    # iShares S&P GSCI commodity
    "COMT",   # iShares GSCI commodity dynamic roll
    "BCI",    # Aberdeen Standard Bloomberg
    "PDBC",   # Invesco Optimum Yield diversified (no K-1)
    "FTGC",   # First Trust Global Tactical
    "DBE",    # Invesco DB Energy diversified
)

CRYPTO_TICKERS: tuple[str, ...] = (
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
    "XRP-USD",   # Ripple
    "LTC-USD",   # Litecoin
    "BCH-USD",   # Bitcoin Cash
    "DOGE-USD",  # Dogecoin
    "XLM-USD",   # Stellar
    "XMR-USD",   # Monero
    "ETC-USD",   # Ethereum Classic
    "DASH-USD",  # Dash
    "ZEC-USD",   # Zcash
    "NEO-USD",   # NEO
    "TRX-USD",   # Tron (lanzado 2017)
    "EOS-USD",   # EOS (lanzado 2017)
)

BOND_US_TICKERS: tuple[str, ...] = (
    # Treasury — curva completa
    "SHV",    # 0-1 año
    "SHY",    # 1-3 años
    "IEI",    # 3-7 años
    "IEF",    # 7-10 años
    "TLH",    # 10-20 años
    "TLT",    # 20+ años
    "VGSH",   # Vanguard short-term Treasury
    "VGIT",   # Vanguard intermediate-term Treasury
    "VGLT",   # Vanguard long-term Treasury
    # Crédito investment grade
    "LQD",    # iShares iBoxx IG corporate
    "VCSH",   # short-term IG corporate
    "VCIT",   # intermediate-term IG
    "VCLT",   # long-term IG
    "MBB",    # mortgage-backed securities
    # Crédito high yield
    "HYG",    # iShares iBoxx HY
    "JNK",    # SPDR Bloomberg HY
    "SHYG",   # short-duration HY
    # TIPS (inflation-protected)
    "TIP",    # iShares broad TIPS
    "STIP",   # short TIPS
    "SCHP",   # Schwab broad TIPS
    # Agregados
    "AGG",    # iShares core US aggregate
    "BND",    # Vanguard total bond market
    "GVI",    # intermediate gov/credit
    # Sovereign global / EM
    "BWX",    # SPDR Barclays international Treasury (ex-US)
    "EMB",    # iShares JPM EM bonds (USD-denominated)
    "EMLC",   # VanEck JPM EM local currency bonds
    "PCY",    # Invesco emerging markets sovereign
)

FX_MIX_TICKERS: tuple[str, ...] = (
    # --- G10 majors (10) ---
    "EURUSD=X",   # Euro
    "GBPUSD=X",   # Libra esterlina
    "USDJPY=X",   # Yen japones
    "USDCHF=X",   # Franco suizo (refugio)
    "USDCAD=X",   # Dolar canadiense
    "AUDUSD=X",   # Dolar australiano
    "NZDUSD=X",   # Dolar neozelandes
    "USDSEK=X",   # Corona sueca
    "USDNOK=X",   # Corona noruega
    "USDDKK=X",   # Corona danesa
    # --- Major crosses (5) ---
    "EURJPY=X",
    "EURGBP=X",
    "GBPJPY=X",
    "AUDJPY=X",
    #"EURCHF=X",  # peg roto 2015, no es comparable al resto del periodo
    # --- Emerging markets crisis-prone (8) ---
    "USDMXN=X",   # Peso mexicano
    "USDBRL=X",   # Real brasileno
    "USDZAR=X",   # Rand sudafricano
    "USDTRY=X",   # Lira turca (cola muy pesada)
    "USDPLN=X",   # Zloty polaco
    "USDHUF=X",   # Forinto hungaro
    "USDINR=X",   # Rupia india
    "USDPHP=X",   # Peso filipino
    # --- Asia liquid (4) ---
    #"USDCNY=X",   # Yuan onshore (managed float), managed float, no es FX libre

    "USDSGD=X",   # Dolar singapur
    "USDKRW=X",   # Won surcoreano
    #"USDTWD=X",   # Dolar taiwanes, # errores de datos masivos en Yahoo (2011, 2014)
    # --- Opcional (descomentar si interesa) ---
    # "USDRUB=X",   # Rublo ruso (split estructural feb 2022)
    # "USDIDR=X",   # Rupia indonesia
    # "USDTHB=X",   # Baht tailandes
    # "USDCLP=X",   # Peso chileno
)

UNIVERSES: dict[str, tuple[str, ...]] = {
    # Cross-asset (generalización sectorial)
    "DJIA": DJIA_TICKERS,
    "IBEX": IBEX_TICKERS,
    "SP50": SP50_TICKERS,
    "HSI": HSI_TICKERS,
    "BRD_CMDY": BRD_CMDY_TICKERS,
    "CRYPTO":   CRYPTO_TICKERS,
    "BOND_US":  BOND_US_TICKERS,
    "FX_MIX":   FX_MIX_TICKERS,
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


def clean_prices(prices: pd.DataFrame, max_missing_frac: float = 0.05) -> pd.DataFrame: # minimum 95% data completeness per ticker!!
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

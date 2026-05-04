"""
Download market data from Yahoo Finance and save as parquet.

Usage:
    python scripts/01_download_data.py
    python scripts/01_download_data.py --markets DJIA SP50
    python scripts/01_download_data.py --start 2010-01-01 --end 2024-12-31
    python scripts/01_download_data.py --synthetic   # use offline generator

If yfinance fails or you have no internet, use --synthetic to generate
realistic-looking data for pipeline smoke-testing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evt_ppo import data as D


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--markets", nargs="+", default=["DJIA", "SP50", "IBEX"],
                   choices=list(D.UNIVERSES.keys()))
    p.add_argument("--start", default="2008-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--synthetic", action="store_true",
                   help="Generate synthetic data instead of downloading.")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for m in args.markets:
        path = out / f"{m}.parquet"
        if args.synthetic:
            print(f"[{m}] generating synthetic panel ...")
            n = len(D.UNIVERSES[m])
            df = D.synthetic_market(
                n_assets=n, n_days=4500, start=args.start, seed=hash(m) % 1000,
            )
            # Rename columns to look like the real tickers, for consistency.
            df.columns = list(D.UNIVERSES[m])[:n]
        else:
            print(f"[{m}] downloading from Yahoo Finance ({args.start} to {args.end})...")
            df = D.download_universe(m, start=args.start, end=args.end)
        D.save_dataset(df, path)
        print(f"[{m}] saved {df.shape[0]} rows x {df.shape[1]} tickers -> {path}")


if __name__ == "__main__":
    main()

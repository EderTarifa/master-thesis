"""
Rebuild results/full_optimal/runs.parquet from the per-run parquets in rows/,
and diagnose the seed 0-4 vs 5-9 discrepancy before overwriting anything.

This does NOT overwrite runs.parquet unless --write is passed. It first prints
a full diagnostic so we understand what is in rows/ vs the current aggregate.

Usage
-----
    # Diagnose only (safe, writes nothing):
    python scripts/rebuild_runs.py --rows-dir results/full_optimal/rows

    # Diagnose AND write the rebuilt parquet:
    python scripts/rebuild_runs.py --rows-dir results/full_optimal/rows --write
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROW_RE = re.compile(
    r"(?P<market>[A-Z0-9_]+)_f(?P<fold>\d{2})_(?P<variant>[A-Z0-9_]+)_s(?P<seed>\d+)\.parquet$"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", default="results/full_optimal/rows")
    ap.add_argument("--out", default="results/full_optimal/runs.parquet")
    ap.add_argument("--write", action="store_true",
                    help="If set, overwrite the aggregate parquet. Default: diagnose only.")
    args = ap.parse_args()

    rows_dir = Path(args.rows_dir)
    files = sorted(glob.glob(str(rows_dir / "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files in {rows_dir}")

    print(f"Found {len(files)} per-run parquets in {rows_dir}\n")

    # Load every row, tracking the source filename and parsed metadata.
    records = []
    bad_files = []
    for f in files:
        name = Path(f).name
        m = ROW_RE.search(name)
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            bad_files.append((name, f"READ ERROR: {e}"))
            continue
        if len(df) != 1:
            bad_files.append((name, f"expected 1 row, got {len(df)}"))
        # Attach filename-derived metadata for cross-checking against columns.
        if m:
            df = df.copy()
            df["_fname_market"] = m["market"]
            df["_fname_fold"] = int(m["fold"])
            df["_fname_variant"] = m["variant"]
            df["_fname_seed"] = int(m["seed"])
        df["_source_file"] = name
        records.append(df)

    if bad_files:
        print("=== FILES WITH ISSUES ===")
        for n, msg in bad_files[:20]:
            print(f"  {n}: {msg}")
        print()

    full = pd.concat(records, ignore_index=True)
    print(f"Total rows assembled: {len(full)}\n")

    # --- Cross-check: do the in-file columns match the filename metadata? ---
    print("=== CONSISTENCY: filename vs in-file columns ===")
    for col in ["market", "variant", "seed", "fold"]:
        fcol = f"_fname_{col}"
        if col in full.columns and fcol in full.columns:
            mismatch = (full[col].astype(str) != full[fcol].astype(str)).sum()
            print(f"  {col}: {mismatch} mismatches between column and filename")
    print()

    # --- The key diagnostic: V4 by seed ---
    v4 = full[full["variant"] == "V4"]
    print("=== V4 metrics by seed (from rebuilt rows) ===")
    metric_cols = [c for c in ["mdd", "sharpe", "calmar", "cagr", "total_return",
                               "turnover_mean", "train_time_seconds"]
                   if c in v4.columns]
    print(v4.groupby("seed")[metric_cols].mean().round(4).to_string())
    print()

    # --- Count runs per (variant, seed) to detect duplicates or missing ---
    print("=== Run counts per (variant, seed) ===")
    counts = full.groupby(["variant", "seed"]).size().unstack(fill_value=0)
    print(counts.to_string())
    print()

    # --- Detect duplicate (market, fold, variant, seed) keys ---
    key = ["market", "fold", "variant", "seed"]
    if all(k in full.columns for k in key):
        dup = full.duplicated(subset=key, keep=False)
        n_dup = int(dup.sum())
        print(f"=== Duplicate (market,fold,variant,seed) keys: {n_dup} ===")
        if n_dup > 0:
            print("  These rows appear more than once - likely the cause of")
            print("  contamination if two runs wrote the same cell differently.")
            dups = full[dup].sort_values(key)
            print(dups[key + ["_source_file"] + metric_cols[:3]].head(20).to_string(index=False))
        print()

    # --- Compare seed 0-4 vs 5-9 to see if it's bimodal ---
    if "seed" in v4.columns and len(v4) > 0:
        v4 = v4.copy()
        v4["grp"] = np.where(v4["seed"] < 5, "0-4", "5-9")
        print("=== V4: seed group 0-4 vs 5-9 ===")
        print(v4.groupby("grp")[metric_cols].mean().round(4).to_string())
        print()
        # Per market, to see if one market drives it
        print("=== V4 by (market, seed group) ===")
        print(v4.groupby(["market", "grp"])[["mdd", "sharpe", "calmar"]].mean().round(4).to_string())
        print()

    # --- Drop helper columns before writing ---
    helper_cols = [c for c in full.columns if c.startswith("_")]
    clean = full.drop(columns=helper_cols)

    if args.write:
        out = Path(args.out)
        # Back up the existing aggregate first.
        if out.exists():
            backup = out.with_suffix(".parquet.bak")
            out.rename(backup)
            print(f"Backed up existing aggregate to {backup}")
        clean.to_parquet(out, index=False)
        print(f"Wrote rebuilt aggregate: {out} ({len(clean)} rows)")
    else:
        print("DIAGNOSE-ONLY mode. Re-run with --write to overwrite the aggregate.")


if __name__ == "__main__":
    main()
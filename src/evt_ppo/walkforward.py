"""
Walk-forward split generator.

Produces a sequence of (train, val, test) slices over a multi-year price
series. Each slice is non-overlapping in test, but train windows do
overlap across folds — this is intentional for parameter stability and
is the standard practice in financial time-series cross-validation.

Default schedule (years):
    - train: 4
    - val:   1
    - test:  1
    - step:  1 (slide forward one year per fold)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class WalkForwardSplit:
    """One fold of the walk-forward schedule."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def slice_df(self, df: pd.DataFrame, which: str) -> pd.DataFrame:
        if which == "train":
            return df.loc[self.train_start:self.train_end]
        if which == "val":
            return df.loc[self.val_start:self.val_end]
        if which == "test":
            return df.loc[self.test_start:self.test_end]
        raise ValueError(f"which must be train/val/test, got {which}")


@dataclass
class WalkForwardConfig:
    train_years: int = 4
    val_years: int = 1
    test_years: int = 1
    step_years: int = 1


def generate_splits(
    index: pd.DatetimeIndex,
    cfg: WalkForwardConfig | None = None,
) -> list[WalkForwardSplit]:
    """Generate walk-forward folds over a date index."""
    cfg = cfg or WalkForwardConfig()
    splits: list[WalkForwardSplit] = []
    start = index.min()
    end = index.max()
    fold = 0
    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(years=cfg.train_years) - pd.Timedelta(days=1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.DateOffset(years=cfg.val_years) - pd.Timedelta(days=1)
        test_start = val_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=cfg.test_years) - pd.Timedelta(days=1)

        if test_end > end:
            break

        splits.append(WalkForwardSplit(
            fold_id=fold,
            train_start=train_start, train_end=train_end,
            val_start=val_start, val_end=val_end,
            test_start=test_start, test_end=test_end,
        ))
        fold += 1
        cursor = cursor + pd.DateOffset(years=cfg.step_years)

    return splits


def describe_splits(splits: list[WalkForwardSplit]) -> pd.DataFrame:
    """Return a DataFrame summarising the schedule for logging."""
    rows = []
    for s in splits:
        rows.append({
            "fold": s.fold_id,
            "train_start": s.train_start.date(),
            "train_end": s.train_end.date(),
            "val_start": s.val_start.date(),
            "val_end": s.val_end.date(),
            "test_start": s.test_start.date(),
            "test_end": s.test_end.date(),
        })
    return pd.DataFrame(rows)

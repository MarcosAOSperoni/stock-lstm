from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_folds(
    dates: pd.DatetimeIndex,
    start,
    holdout_start,
    train_years: int,
    test_months: int,
    purge: int,
) -> list[Fold]:
    """Expanding-window folds over trading `dates` in [start, holdout_start).

    `purge` trading days are dropped before each test window (and at the end of the last
    one) so no training label, and no test label, depends on prices past that boundary.
    For labels built with lag L and horizon H, purge must be at least L + H.
    """
    start, holdout_start = pd.Timestamp(start), pd.Timestamp(holdout_start)
    dates = pd.DatetimeIndex(np.sort(dates.unique()))
    dates = dates[(dates >= start) & (dates < holdout_start)]
    folds: list[Fold] = []
    test_start = start + pd.DateOffset(years=train_years)
    while test_start < holdout_start:
        test_end = min(test_start + pd.DateOffset(months=test_months), holdout_start)
        ts = int(dates.searchsorted(test_start))
        te = int(dates.searchsorted(test_end)) - 1
        if test_end >= holdout_start:
            te -= purge
        tr_end = ts - purge - 1
        if tr_end > 0 and te >= ts:
            folds.append(Fold(dates[0], dates[tr_end], dates[ts], dates[te]))
        test_start = test_end
    return folds

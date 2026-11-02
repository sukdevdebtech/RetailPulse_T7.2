"""Feature engineering for the weekly demand series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_weekly(daily: pd.DataFrame, column: str = "total_revenue") -> pd.Series:
    """Weekly (Mon-Sun, labelled by week-end) totals. Incomplete edge weeks are dropped,
    otherwise a partial final week looks like a sales collapse and poisons the forecast."""
    idx = daily.index
    start = idx.min()
    if start.weekday() != 0:                     # begin on the first full Monday
        start = start + pd.Timedelta(days=7 - start.weekday())
    end = idx.max()
    if end.weekday() != 6:                       # end on the last full Sunday
        end = end - pd.Timedelta(days=end.weekday() + 1)
    sub = daily.loc[start:end, column]
    weekly = sub.resample("W-SUN").sum()
    weekly.name = column
    return weekly


def fourier_terms(dates: pd.DatetimeIndex, order: int = 3) -> np.ndarray:
    """Yearly seasonality encoded as sin/cos pairs (smooth, no 53-way dummy needed)."""
    doy = np.asarray(dates.dayofyear, dtype=float)
    cols = []
    for k in range(1, order + 1):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    return np.column_stack(cols)


def lag_features(history: np.ndarray, lags: tuple[int, ...] = (1, 2, 3, 4, 8)) -> np.ndarray:
    """Feature vector for predicting the step after `history`."""
    h = np.asarray(history, dtype=float)
    vals = [h[-lag] for lag in lags]
    vals.append(h[-4:].mean())
    vals.append(h[-8:].mean() if len(h) >= 8 else h.mean())
    return np.asarray(vals)

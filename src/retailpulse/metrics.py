"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(_arr(y) - _arr(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((_arr(y) - _arr(yhat)) ** 2)))


def mape(y, yhat) -> float:
    """Mean absolute percentage error in %, ignoring zero actuals."""
    y, yhat = _arr(y), _arr(yhat)
    mask = y != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def smape(y, yhat) -> float:
    y, yhat = _arr(y), _arr(yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    mask = denom != 0
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / denom[mask]) * 100)


def wape(y, yhat) -> float:
    """Weighted absolute percentage error (%); robust when some actuals are tiny."""
    y, yhat = _arr(y), _arr(yhat)
    total = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / total * 100) if total else float("nan")


def summary(y, yhat) -> dict:
    return {
        "MAPE": round(mape(y, yhat), 2),
        "sMAPE": round(smape(y, yhat), 2),
        "WAPE": round(wape(y, yhat), 2),
        "MAE": round(mae(y, yhat), 2),
        "RMSE": round(rmse(y, yhat), 2),
    }

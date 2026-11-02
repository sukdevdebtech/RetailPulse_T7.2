"""Weekly demand forecasting: baselines + ML lag models + CV-weighted ensemble.

Everything here depends only on numpy / pandas / scikit-learn so it runs anywhere.
Prophet is supported as an *optional* extra model when the package is installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, metrics
from .features import fourier_terms, lag_features

# --------------------------------------------------------------------------- models


class BaseForecaster:
    name = "base"

    def fit(self, y: pd.Series) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError


class NaiveLast(BaseForecaster):
    name = "Naive (last week)"

    def fit(self, y):
        self._last = float(y.iloc[-1])
        return self

    def predict(self, horizon):
        return np.repeat(self._last, horizon)


class MovingAverage(BaseForecaster):
    name = "Moving average (4w)"

    def __init__(self, window: int = 4):
        self.window = window

    def fit(self, y):
        self._level = float(y.iloc[-self.window:].mean())
        return self

    def predict(self, horizon):
        return np.repeat(self._level, horizon)


class SeasonalNaive(BaseForecaster):
    """Same week last year, scaled by recent year-over-year momentum (falls back to level)."""

    name = "Seasonal naive (52w)"

    def __init__(self, season: int = 52):
        self.season = season

    def fit(self, y):
        self._y = y.astype(float).values
        return self

    def predict(self, horizon):
        y, s = self._y, self.season
        if len(y) < s + 8:
            return np.repeat(y[-4:].mean(), horizon)
        growth = y[-8:].sum() / max(y[-8 - s:-s].sum(), 1.0)
        growth = float(np.clip(growth, 0.6, 1.6))
        base = np.array([y[len(y) - s + i] for i in range(horizon)])
        return base * growth


class LagRegressor(BaseForecaster):
    """Recursive multi-step model on lags + rolling means + yearly Fourier terms (log target)."""

    def __init__(self, estimator, name: str, fourier_order: int = 3):
        self.estimator = estimator
        self.name = name
        self.fourier_order = fourier_order

    def _row(self, hist: np.ndarray, date: pd.Timestamp) -> np.ndarray:
        f = fourier_terms(pd.DatetimeIndex([date]), self.fourier_order)[0]
        return np.concatenate([np.log1p(lag_features(hist)), f])

    def fit(self, y):
        vals = y.astype(float).values
        dates = y.index
        X, t = [], []
        for i in range(8, len(vals)):
            X.append(self._row(vals[:i], dates[i]))
            t.append(np.log1p(vals[i]))
        self.estimator.fit(np.asarray(X), np.asarray(t))
        self._hist = vals.copy()
        self._last_date = dates[-1]
        return self

    def predict(self, horizon):
        hist = self._hist.copy()
        out = []
        date = self._last_date
        for _ in range(horizon):
            date = date + pd.Timedelta(days=7)
            pred = float(np.expm1(self.estimator.predict(self._row(hist, date)[None, :])[0]))
            pred = max(pred, 0.0)
            out.append(pred)
            hist = np.append(hist, pred)
        return np.asarray(out)


class ProphetModel(BaseForecaster):  # optional
    name = "Prophet"

    def fit(self, y):
        from prophet import Prophet  # imported lazily: optional dependency

        df = pd.DataFrame({"ds": y.index, "y": y.values})
        self._m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        self._m.fit(df)
        self._last = y.index[-1]
        return self

    def predict(self, horizon):
        fut = pd.DataFrame({"ds": pd.date_range(self._last + pd.Timedelta(days=7), periods=horizon, freq="W-SUN")})
        return np.clip(self._m.predict(fut)["yhat"].values, 0, None)


def default_models(include_prophet: bool = False) -> dict[str, BaseForecaster]:
    rs = config.RANDOM_STATE
    models: dict[str, BaseForecaster] = {
        "naive": NaiveLast(),
        "moving_avg": MovingAverage(4),
        "seasonal_naive": SeasonalNaive(52),
        "ridge": LagRegressor(make_pipeline(StandardScaler(), Ridge(alpha=3.0)), "Ridge lag model"),
        "gbm": LagRegressor(
            GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                      subsample=0.8, random_state=rs),
            "Gradient boosting lag model"),
    }
    if include_prophet:
        try:
            import prophet  # noqa: F401
            models["prophet"] = ProphetModel()
        except ImportError:
            pass
    return models


def _fresh(model: BaseForecaster) -> BaseForecaster:
    from copy import deepcopy

    return deepcopy(model)


# ------------------------------------------------------------------ validation


@dataclass
class CVResult:
    """Out-of-sample predictions per model for every walk-forward fold."""

    frame: pd.DataFrame                     # columns: fold, step, date, actual, <model keys...>
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)


def walk_forward_cv(models: dict[str, BaseForecaster], y: pd.Series, initial: int | None = None,
                    horizon: int | None = None, step: int | None = None) -> CVResult:
    initial = initial or config.CV_INITIAL_WEEKS
    horizon = horizon or config.FORECAST_HORIZON_WEEKS
    step = step or config.CV_STEP_WEEKS
    if len(y) < initial + horizon:
        raise ValueError(f"Need at least {initial + horizon} weeks of data, got {len(y)}")
    rows = []
    fold = 0
    for end in range(initial, len(y) - horizon + 1, step):
        fold += 1
        train, test = y.iloc[:end], y.iloc[end:end + horizon]
        preds = {k: _fresh(m).fit(train).predict(horizon) for k, m in models.items()}
        for s in range(horizon):
            row = {"fold": fold, "step": s + 1, "date": test.index[s], "actual": float(test.iloc[s])}
            row.update({k: float(p[s]) for k, p in preds.items()})
            rows.append(row)
    frame = pd.DataFrame(rows)
    scores = pd.DataFrame({
        k: metrics.summary(frame["actual"], frame[k]) for k in models
    }).T.sort_values("WAPE")
    scores.index.name = "model"
    return CVResult(frame=frame, scores=scores)


def ensemble_weights(scores: pd.DataFrame, top_k: int = 3) -> dict[str, float]:
    """Inverse-squared-error weights over the top_k models (by WAPE)."""
    best = scores.head(top_k)
    inv = 1.0 / np.square(best["WAPE"].clip(lower=0.1))
    w = inv / inv.sum()
    return {k: float(v) for k, v in w.items()}


def blend(frame_or_preds: pd.DataFrame | dict, weights: dict[str, float]):
    return sum(frame_or_preds[k] * w for k, w in weights.items())


# ---------------------------------------------------------------- final product


@dataclass
class ForecastReport:
    forecast: pd.DataFrame          # date, forecast, lower, upper, per-model columns
    cv_scores: pd.DataFrame         # per-model CV metrics (+ Ensemble row)
    cv_frame: pd.DataFrame
    weights: dict[str, float]
    coverage: float                 # empirical coverage of the prediction interval in CV


def fit_and_forecast(y: pd.Series, horizon: int | None = None, interval: float = 0.8,
                     include_prophet: bool = False) -> ForecastReport:
    horizon = horizon or config.FORECAST_HORIZON_WEEKS
    models = default_models(include_prophet)
    cv = walk_forward_cv(models, y)
    weights = ensemble_weights(cv.scores)

    cvf = cv.frame.copy()
    cvf["ensemble"] = blend(cvf, weights)
    scores = cv.scores.copy()
    scores.loc["ensemble"] = metrics.summary(cvf["actual"], cvf["ensemble"])
    scores = scores.sort_values("WAPE")

    # prediction interval from relative CV errors of the ensemble
    rel = (cvf["actual"] / cvf["ensemble"].clip(lower=1.0)) - 1.0
    lo_q, hi_q = (1 - interval) / 2, 1 - (1 - interval) / 2
    lo, hi = float(rel.quantile(lo_q)), float(rel.quantile(hi_q))
    lo, hi = min(lo, -0.01), max(hi, 0.01)      # an interval must always bracket the point forecast
    coverage = float(((rel >= lo) & (rel <= hi)).mean())

    preds = {k: _fresh(models[k]).fit(y).predict(horizon) for k in weights}
    ens = blend(preds, weights)
    dates = pd.date_range(y.index[-1] + pd.Timedelta(days=7), periods=horizon, freq="W-SUN")
    fc = pd.DataFrame({
        "date": dates,
        "forecast": ens,
        "lower": np.clip(ens * (1 + lo), 0, None),
        "upper": ens * (1 + hi),
    })
    for k, p in preds.items():
        fc[k] = p
    return ForecastReport(forecast=fc, cv_scores=scores, cv_frame=cvf, weights=weights,
                          coverage=coverage)

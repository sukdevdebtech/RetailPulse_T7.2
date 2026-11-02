import numpy as np
import pandas as pd
import pytest

from retailpulse import data, features, forecasting, metrics


def _series(n=90):
    idx = pd.date_range("2010-01-03", periods=n, freq="W-SUN")
    t = np.arange(n)
    return pd.Series(1000 + 5 * t + 200 * np.sin(2 * np.pi * t / 52), index=idx)


def test_metrics_basic():
    assert metrics.mae([1, 2], [2, 2]) == 0.5
    assert metrics.mape([100, 0], [110, 5]) == pytest.approx(10.0)   # zero actual ignored
    assert metrics.wape([10, 10], [12, 8]) == pytest.approx(20.0)


def test_to_weekly_drops_partial_weeks():
    d = data.load_daily()
    w = features.to_weekly(d)
    assert w.index.min().weekday() == 6 and w.index.max().weekday() == 6
    assert w.index.max() <= d.index.max()
    assert w.iloc[-1] > 0.3 * w.iloc[-5:-1].mean()      # last week is a full week, not a stub


def test_walk_forward_has_no_lookahead():
    y = _series()
    cv = forecasting.walk_forward_cv({"naive": forecasting.NaiveLast()}, y)
    f = cv.frame
    first = f[f["fold"] == 1]
    # the naive forecast for fold 1 must equal the last TRAIN value (week 52), never a test value
    assert first["naive"].iloc[0] == pytest.approx(y.iloc[51])


def test_cv_needs_enough_data():
    with pytest.raises(ValueError):
        forecasting.walk_forward_cv({"naive": forecasting.NaiveLast()}, _series(20))


def test_ensemble_weights_sum_to_one():
    scores = pd.DataFrame({"WAPE": [10.0, 20.0, 40.0, 80.0]}, index=list("abcd"))
    w = forecasting.ensemble_weights(scores, top_k=3)
    assert set(w) == {"a", "b", "c"} and sum(w.values()) == pytest.approx(1.0)
    assert w["a"] > w["b"] > w["c"]


def test_fit_and_forecast_shapes_and_bounds():
    rep = forecasting.fit_and_forecast(_series(), horizon=4)
    fc = rep.forecast
    assert len(fc) == 4
    assert (fc["lower"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["upper"]).all()
    assert (fc["forecast"] >= 0).all()
    assert rep.cv_scores.loc["ensemble", "WAPE"] < 15           # clean synthetic series is easy
    assert 0.5 <= rep.coverage <= 1.0

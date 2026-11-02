import numpy as np
import pandas as pd
import pytest

from retailpulse import churn, data, drift, inventory, segmentation


def test_score_rfm_ranges_and_labels():
    s = segmentation.score_rfm(data.load_rfm())
    for c in ("r_score", "f_score", "m_score"):
        assert s[c].between(1, 5).all()
    best = s.sort_values("monetary").iloc[-1]
    assert best["m_score"] == 5
    assert segmentation.label_segment(5, 5, 5) == "Champions"
    assert segmentation.label_segment(1, 1, 1) == "Lost"


def test_kmeans_picks_valid_k():
    rfm = data.load_rfm().sample(1500, random_state=0)
    out, info = segmentation.kmeans_segments(rfm)
    assert 3 <= info["k"] <= 6 and -1 < info["silhouette"] < 1
    assert out["cluster"].nunique() == info["k"]


def test_churn_does_not_use_recency():
    assert "recency" not in " ".join(churn.FEATURES)
    res = churn.train(data.load_rfm())
    assert 0.6 < res.metrics["auc_holdout"] < 0.99       # AUC of 1.0 would signal label leakage
    assert res.scored["churn_probability"].between(0, 1).all()
    risky = churn.at_risk_active(res.scored, 10)
    assert (risky["lapsed"] == 0).all()
    assert risky["revenue_at_risk"].is_monotonic_decreasing


def test_eoq_matches_formula():
    q = inventory.eoq(annual_demand=1000, ordering_cost=50, holding_cost_per_unit=2)
    assert q == pytest.approx(np.sqrt(2 * 1000 * 50 / 2))
    with pytest.raises(ValueError):
        inventory.eoq(0, 50, 2)


def test_safety_stock_grows_with_service_level():
    a = inventory.safety_stock(10, 4, 0.90)
    b = inventory.safety_stock(10, 4, 0.99)
    assert b > a > 0
    with pytest.raises(ValueError):
        inventory.z_value(1.0)


def test_simulation_never_negative_and_conserves_units():
    idx = pd.date_range("2024-01-01", periods=120)
    demand = pd.Series(np.random.default_rng(1).poisson(20, 120).astype(float), index=idx)
    sim = inventory.simulate(demand, reorder_pt=150, order_qty=300, lead_time_days=5)
    assert (sim["on_hand"] >= 0).all()
    assert (sim["sold"] + sim["shortage"]).sum() == pytest.approx(demand.sum())
    assert inventory.sim_metrics(sim)["fill_rate_pct"] > 90


def test_simulation_does_not_reorder_while_order_in_transit():
    idx = pd.date_range("2024-01-01", periods=10)
    demand = pd.Series([50.0] * 10, index=idx)
    sim = inventory.simulate(demand, reorder_pt=100, order_qty=500, lead_time_days=6, initial_stock=120)
    assert (sim["ordered"] > 0).sum() <= 2


def test_psi_identical_is_zero_and_shift_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert drift.psi(a, a) < 0.01
    assert drift.psi(a, rng.normal(2, 1, 5000)) > 0.25
    assert drift.status_from_psi(0.05) == "Stable"
    assert drift.status_from_psi(0.3) == "Significant drift"


def test_performance_drift_flag():
    assert drift.performance_drift(10, 20)["retrain_recommended"] is True
    assert drift.performance_drift(10, 11)["retrain_recommended"] is False

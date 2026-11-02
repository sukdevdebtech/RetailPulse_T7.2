import json

import pytest

from retailpulse.service import AnalyticsService


def test_pipeline_writes_all_artifacts(artifacts):
    expected = {"summary.json", "customer_scores.csv", "forecast.csv", "cv_scores.csv",
                "churn_model.joblib", "drift_report.csv", "inventory_simulation.csv",
                "at_risk_customers.csv", "run_history.jsonl"}
    assert expected <= {p.name for p in artifacts.iterdir()}
    s = json.loads((artifacts / "summary.json").read_text())
    assert s["rows"]["customers"] > 5000 and s["forecast"]["ensemble"]["WAPE"] < 60


def test_service_queries(artifacts):
    svc = AnalyticsService(artifacts, auto_train=False)
    assert svc.health()["status"] == "ok"
    assert len(svc.forecast(3)) == 3
    with pytest.raises(ValueError):
        svc.forecast(0)
    c = svc.customer(12347)
    assert c["segment"] and 0 <= c["churn_probability"] <= 1
    with pytest.raises(KeyError):
        svc.customer(-1)


def test_service_churn_and_inventory(artifacts):
    svc = AnalyticsService(artifacts, auto_train=False)
    low = svc.predict_churn(frequency=50, monetary=50000)["churn_probability"]
    high = svc.predict_churn(frequency=1, monetary=40)["churn_probability"]
    assert high > low
    a = svc.inventory_plan(service_level=0.90)["plan"]["safety_stock"]
    b = svc.inventory_plan(service_level=0.99)["plan"]["safety_stock"]
    assert b > a
    with pytest.raises(ValueError):
        svc.predict_churn(0, 10)

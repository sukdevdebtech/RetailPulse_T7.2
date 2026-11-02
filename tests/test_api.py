"""API tests run only when fastapi + httpx are installed (they are in CI)."""
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(artifacts):
    from api import main
    from retailpulse.service import AnalyticsService

    main.get_service.cache_clear()
    main.app.dependency_overrides[main.get_service] = lambda: AnalyticsService(artifacts, auto_train=False)
    return TestClient(main.app)


def test_health_and_forecast(client):
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/forecast", params={"horizon": 4}).json()
    assert len(body["forecast"]) == 4
    assert client.get("/forecast", params={"horizon": 9}).status_code == 422


def test_customer_and_churn(client):
    assert client.get("/customers/12347").status_code == 200
    assert client.get("/customers/-5").status_code == 404
    ok = client.post("/churn/predict", json={"frequency": 2, "monetary": 100})
    assert 0 <= ok.json()["churn_probability"] <= 1
    assert client.post("/churn/predict", json={"frequency": 0, "monetary": 100}).status_code == 422


def test_inventory_and_monitoring(client):
    r = client.post("/inventory/plan", json={"service_level": 0.99, "lead_time_days": 10})
    assert r.json()["plan"]["safety_stock"] > 0
    assert "features" in client.get("/monitoring/drift").json()

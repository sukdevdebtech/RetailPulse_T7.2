"""RetailPulse REST API.   uvicorn api.main:app --reload --port 8000"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import Depends, FastAPI, HTTPException, Query  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from retailpulse import __version__  # noqa: E402
from retailpulse.service import AnalyticsService  # noqa: E402

app = FastAPI(title="RetailPulse API", version=__version__,
              description="Forecasts, churn scoring, inventory planning and monitoring.")


@lru_cache(maxsize=1)
def get_service() -> AnalyticsService:
    return AnalyticsService()


class ChurnRequest(BaseModel):
    frequency: float = Field(..., gt=0, description="Number of purchase orders")
    monetary: float = Field(..., gt=0, description="Total spend")


class InventoryRequest(BaseModel):
    service_level: float = Field(0.95, gt=0.5, lt=1)
    lead_time_days: float = Field(7, gt=0, le=180)
    ordering_cost: float = Field(50, gt=0)
    holding_rate: float = Field(0.2, gt=0, le=1)


@app.get("/health")
def health(svc: AnalyticsService = Depends(get_service)):
    return svc.health()


@app.get("/forecast")
def forecast(horizon: int = Query(4, ge=1, le=4), svc: AnalyticsService = Depends(get_service)):
    return {"unit": "weekly revenue", "forecast": svc.forecast(horizon)}


@app.get("/customers/{customer_id}")
def customer(customer_id: int, svc: AnalyticsService = Depends(get_service)):
    try:
        return svc.customer(customer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"customer {customer_id} not found")


@app.post("/churn/predict")
def churn_predict(req: ChurnRequest, svc: AnalyticsService = Depends(get_service)):
    return svc.predict_churn(req.frequency, req.monetary)


@app.post("/inventory/plan")
def inventory_plan(req: InventoryRequest, svc: AnalyticsService = Depends(get_service)):
    return svc.inventory_plan(req.service_level, req.lead_time_days, req.ordering_cost,
                              req.holding_rate)


@app.get("/monitoring/drift")
def drift(svc: AnalyticsService = Depends(get_service)):
    return {"features": svc.drift(), "performance": svc.summary["performance_drift"]}


@app.get("/monitoring/runs")
def runs(svc: AnalyticsService = Depends(get_service)):
    return svc.runs()

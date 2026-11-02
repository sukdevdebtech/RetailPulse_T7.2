"""Framework-free service layer used by the FastAPI app (and unit-tested directly)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from . import churn, config, inventory, monitoring, pipeline


class ArtifactsMissing(RuntimeError):
    pass


class AnalyticsService:
    def __init__(self, artifact_dir: Path | None = None, auto_train: bool = True):
        self.dir = Path(artifact_dir or config.ARTIFACT_DIR)
        if not (self.dir / "summary.json").exists():
            if not auto_train:
                raise ArtifactsMissing(f"No artifacts in {self.dir}; run `python -m retailpulse.pipeline`")
            pipeline.run(self.dir)
        self.summary = json.loads((self.dir / "summary.json").read_text())
        self.customers = pd.read_csv(self.dir / "customer_scores.csv").set_index("Customer ID")
        self.forecast_df = pd.read_csv(self.dir / "forecast.csv", parse_dates=["date"])
        self.daily = pd.read_csv(self.dir / "daily_sales.csv", parse_dates=["Date"]).set_index("Date")
        self.model = joblib.load(self.dir / "churn_model.joblib")

    # ------------------------------------------------------------------ queries
    def health(self) -> dict:
        return {"status": "ok", "version": self.summary["version"],
                "customers": self.summary["rows"]["customers"],
                "retrain_recommended": self.summary["performance_drift"]["retrain_recommended"]}

    def forecast(self, horizon: int = 4) -> list[dict]:
        if not 1 <= horizon <= len(self.forecast_df):
            raise ValueError(f"horizon must be between 1 and {len(self.forecast_df)}")
        df = self.forecast_df.head(horizon)[["date", "forecast", "lower", "upper"]].copy()
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df.round(2).to_dict("records")

    def customer(self, customer_id: int) -> dict:
        if customer_id not in self.customers.index:
            raise KeyError(customer_id)
        row = self.customers.loc[customer_id]
        keep = ["recency", "frequency", "monetary", "r_score", "f_score", "m_score", "segment",
                "cluster_name", "lapsed", "churn_probability", "risk_tier"]
        rec = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in keep}
        rec["customer_id"] = int(customer_id)
        rec["churn_probability"] = round(float(rec["churn_probability"]), 4)
        return rec

    def predict_churn(self, frequency: float, monetary: float) -> dict:
        if frequency <= 0 or monetary <= 0:
            raise ValueError("frequency and monetary must be > 0")
        p = churn.predict_proba(self.model, frequency, monetary)
        return {"churn_probability": round(p, 4), "risk_tier": churn.risk_tier(p)}

    def inventory_plan(self, service_level: float | None = None, lead_time_days: float | None = None,
                       ordering_cost: float | None = None, holding_rate: float | None = None) -> dict:
        plan = inventory.plan_from_demand(self.daily["total_quantity"], self.summary["avg_price"],
                                          ordering_cost, holding_rate, lead_time_days, service_level)
        sim = inventory.simulate(self.daily["total_quantity"], plan.reorder_point, plan.eoq,
                                 plan.lead_time_days)
        return {"plan": plan.to_dict(), "simulation": inventory.sim_metrics(sim)}

    def drift(self) -> list[dict]:
        return pd.read_csv(self.dir / "drift_report.csv").to_dict("records")

    def runs(self) -> list[dict]:
        return monitoring.read_history(self.dir)

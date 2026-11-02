"""End-to-end training pipeline.

    python -m retailpulse.pipeline                       # bundled sample data
    python -m retailpulse.pipeline --transactions online_retail_II.csv   # raw UCI data

Writes every artifact the API and dashboard need into ./artifacts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from . import __version__, churn, config, data, drift, features, forecasting, inventory
from . import monitoring, segmentation


def run(artifact_dir: Path | None = None, transactions: Path | None = None,
        daily_path: Path | None = None, rfm_path: Path | None = None,
        use_mlflow: bool | None = None, include_prophet: bool = False) -> dict:
    t0 = time.time()
    out = Path(artifact_dir or config.ARTIFACT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Data layer -----------------------------------------------------------
    if transactions:
        raw = pd.read_csv(transactions) if str(transactions).endswith(".csv") else pd.read_excel(transactions)
        clean = data.clean_transactions(raw)
        daily, rfm = data.transactions_to_daily(clean), data.transactions_to_rfm(clean)
        source = str(transactions)
    else:
        daily, rfm = data.load_daily(daily_path), data.load_rfm(rfm_path)
        source = "bundled sample (Online Retail II aggregates)"

    # 2. Segmentation -----------------------------------------------------------
    scored = segmentation.score_rfm(rfm)
    clustered, cluster_info = segmentation.kmeans_segments(rfm)
    scored = scored.merge(clustered[["Customer ID", "cluster", "cluster_name"]], on="Customer ID")

    # 3. Churn --------------------------------------------------------------------
    ch = churn.train(rfm)
    scored = scored.merge(ch.scored[["Customer ID", "lapsed", "churn_probability", "risk_tier"]],
                          on="Customer ID")
    joblib.dump(ch.model, out / "churn_model.joblib")

    # 4. Forecasting ----------------------------------------------------------------
    weekly = features.to_weekly(daily)
    fc = forecasting.fit_and_forecast(weekly, include_prophet=include_prophet)

    # 5. Inventory ---------------------------------------------------------------------
    avg_price = float(daily["total_revenue"].sum() / max(daily["total_quantity"].sum(), 1))
    plan = inventory.plan_from_demand(daily["total_quantity"], avg_price)
    sim = inventory.simulate(daily["total_quantity"], plan.reorder_point, plan.eoq, plan.lead_time_days)
    inv_metrics = inventory.sim_metrics(sim)
    sl_scan = inventory.service_level_scan(daily["total_quantity"], avg_price)

    # 6. Monitoring -------------------------------------------------------------------------
    ref, cur = drift.split_reference_current(daily)
    drift_df = drift.detect_drift(ref, cur, ["total_revenue", "total_quantity", "transaction_count"])
    recent = fc.cv_frame[fc.cv_frame["fold"] >= fc.cv_frame["fold"].max() - 2]
    from .metrics import wape
    perf = drift.performance_drift(float(fc.cv_scores.loc["ensemble", "WAPE"]),
                                   wape(recent["actual"], recent["ensemble"]))

    # 7. Persist ----------------------------------------------------------------------------------
    daily.reset_index().to_csv(out / "daily_sales.csv", index=False)
    weekly.reset_index().to_csv(out / "weekly_sales.csv", index=False)
    scored.to_csv(out / "customer_scores.csv", index=False)
    churn.at_risk_active(ch.scored, 100).to_csv(out / "at_risk_customers.csv", index=False)
    ch.importance.to_csv(out / "churn_feature_importance.csv", index=False)
    segmentation.segment_summary(scored, "segment").to_csv(out / "segment_summary.csv")
    segmentation.segment_summary(scored, "cluster_name").to_csv(out / "cluster_summary.csv")
    fc.forecast.to_csv(out / "forecast.csv", index=False)
    fc.cv_scores.to_csv(out / "cv_scores.csv")
    fc.cv_frame.to_csv(out / "cv_predictions.csv", index=False)
    sim.reset_index().to_csv(out / "inventory_simulation.csv", index=False)
    sl_scan.to_csv(out / "service_level_scan.csv", index=False)
    drift_df.to_csv(out / "drift_report.csv", index=False)

    summary = {
        "version": __version__,
        "source": source,
        "rows": {"days": int(len(daily)), "weeks": int(len(weekly)), "customers": int(len(rfm))},
        "total_revenue": round(float(daily["total_revenue"].sum()), 2),
        "avg_price": round(avg_price, 4),
        "clustering": cluster_info,
        "churn": ch.metrics,
        "forecast": {"weights": fc.weights, "interval_coverage": round(fc.coverage, 3),
                     "best_model": str(fc.cv_scores.index[0]),
                     "ensemble": fc.cv_scores.loc["ensemble"].to_dict()},
        "inventory": {"plan": plan.to_dict(), "simulation": inv_metrics},
        "performance_drift": perf,
        "data_drift_features_flagged": int((drift_df["status"] != "Stable").sum()),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    flat = {"wape_ensemble": summary["forecast"]["ensemble"]["WAPE"],
            "churn_auc": ch.metrics["auc_holdout"], "fill_rate": inv_metrics["fill_rate_pct"],
            "silhouette": cluster_info["silhouette"]}
    monitoring.log_run(out, {"source": source, "version": __version__}, flat, use_mlflow)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the RetailPulse pipeline")
    ap.add_argument("--transactions", type=Path, help="raw Online Retail II csv/xlsx")
    ap.add_argument("--out", type=Path, default=config.ARTIFACT_DIR)
    ap.add_argument("--mlflow", action="store_true", help="also log to MLflow")
    ap.add_argument("--prophet", action="store_true", help="include Prophet if installed")
    a = ap.parse_args()
    s = run(a.out, a.transactions, use_mlflow=a.mlflow or None, include_prophet=a.prophet)
    print(json.dumps({k: s[k] for k in ("source", "rows", "forecast", "churn", "runtime_seconds")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()

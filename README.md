# RetailPulse v2 — AI-Powered Customer Analytics & Demand Forecasting

Production-ready upgrade of the RetailPulse prototype (Zidio Data Science & Analytics project).
Everything in the prototype's notebooks and dashboard is now an importable, tested Python package
with a REST API, an upgraded Streamlit dashboard, Docker/CI and a monitoring layer.

> **Scope of this release — "Week 4":** dashboard development, Docker containerisation,
> deployment, testing and the final report (see the brief's execution timeline).
> Weeks 1-3 analysis (EDA → segmentation → forecasting → inventory) is re-implemented here as
> clean, reusable modules instead of notebook cells.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt && pip install -e .

python -m retailpulse.pipeline        # trains everything into ./artifacts (~5 s)
streamlit run dashboard/app.py        # dashboard  → http://localhost:8501
uvicorn api.main:app --port 8000      # REST API   → http://localhost:8000/docs
pytest -v                             # 25 tests
```

Docker: `docker compose up --build` (dashboard :8501, API :8000).
Use your own data: `python -m retailpulse.pipeline --transactions online_retail_II.csv`
(raw UCI *Online Retail II* export; cleaning is automatic).

## What's new vs. the prototype

| Area | Prototype | v2 |
|---|---|---|
| Code | Notebook cells / one 200-line Streamlit file; empty `src/` | `src/retailpulse/` package: data, features, forecasting, segmentation, churn, inventory, drift, monitoring, pipeline, service |
| Forecast validation | One 5-week hold-out ("3.65% MAPE") | Walk-forward CV (13 folds), WAPE/MAPE/sMAPE/MAE/RMSE, 80% prediction intervals with measured coverage |
| Churn model | AUC = 1.00 — **label leakage** (label is `recency > 90`, and recency was a feature) | Recency excluded; honest AUC ≈ 0.77; applied to *active* customers for a win-back list ranked by revenue at risk |
| Weekly series | Last week was a partial week (looks like a sales crash) | Incomplete edge weeks dropped |
| Inventory | Daily re-ordering while orders in transit | Proper (s, Q) simulation using inventory *position*; EOQ, safety stock, ROP, service-level trade-off |
| Drift | Static report | PSI + KS data drift **and** model-performance drift with a retrain flag |
| Serving | Streamlit only | FastAPI (`/forecast`, `/churn/predict`, `/customers/{id}`, `/inventory/plan`, `/monitoring/*`) |
| Quality | None | pytest suite (no-look-ahead CV test, leakage guard, simulation conservation, API tests), ruff, GitHub Actions |
| Ops | Dockerfile | Docker image trains at build time, compose with API + optional MLflow, healthchecks |

## Architecture

```
Raw/aggregated sales ─► data.py (validate, clean) ─► features.py (weekly, lags, Fourier)
        │
        ├─► segmentation.py   RFM quintiles + rule labels + K-Means (k by silhouette)
        ├─► churn.py          HistGradientBoosting, permutation importance, win-back list
        ├─► forecasting.py    naive / MA / seasonal-naive / Ridge / GBM (+Prophet if installed)
        │                     → walk-forward CV → inverse-error ensemble → intervals
        ├─► inventory.py      EOQ · safety stock · reorder point · (s,Q) simulation
        └─► drift.py          PSI, KS, performance drift
                    │
              pipeline.py ──► artifacts/ (CSV, joblib, summary.json, run_history.jsonl [+MLflow])
                    │
        service.py ─┴─► api/main.py (FastAPI)      dashboard/app.py (Streamlit, 6 pages)
```

## Headline results (bundled sample: Online Retail II, Dec 2009 – Dec 2011)

| Metric | Result |
|---|---|
| Weekly demand forecast, ensemble | WAPE 23.1 % · MAPE 27.1 % (walk-forward CV; best single model WAPE 24.2 %) |
| Prediction interval | 80 % nominal, 77 % empirical coverage |
| Churn model (lapsed = 90+ days) | AUC 0.77 hold-out / 0.76 5-fold CV, recall 0.74 |
| Segmentation | 9 RFM segments; K-Means k = 4 (silhouette 0.36). Champions = 22 % of customers, 69 % of revenue |
| Inventory (EOQ 45.8k units, SS 54.6k, ROP 154k at 95 % SL, 7-day lead time) | 97.7 % simulated fill rate; raising SL to 99 % → 99.2 % fill, +29 % holding cost |
| Drift | All 3 monitored series drift upward vs. history (business growth), model error not degraded (recent WAPE 13.8 % vs 23.1 % baseline) |

See `reports/final_report.md` for method, figures and limitations.

## Repository layout

```
src/retailpulse/   package (see architecture)      api/main.py      FastAPI app
dashboard/app.py   Streamlit dashboard             tests/           pytest suite
data/sample/       bundled aggregates              artifacts/       pipeline outputs
docs/              week-4 plan, submission guide   reports/         final report + figures
Dockerfile · docker-compose.yml · Makefile · .github/workflows/ci.yml
```

## Dashboard pages
Executive Overview · Sales Analytics (date filter, daily/weekly/monthly, seasonality heatmap) ·
Customer Intelligence (RFM, clusters, churn, win-back list + CSV export) ·
Demand Forecast (what-if ±30 %, CV comparison) · Inventory Planner (service level, lead time, costs) ·
Model Monitoring (drift, retrain flag, run history).

## Honest limitations
- The bundled data are daily/customer **aggregates** from the prototype, not the 1M-row raw file;
  run with `--transactions` for the full dataset. Revenue is in GBP.
- Only ~104 weeks of history, so yearly seasonality rests on a single prior year → wide intervals
  around Christmas. More history is the biggest accuracy lever.
- Churn is defined by inactivity, so the model estimates *lapse propensity* from behaviour; it is
  not a causal retention model.
- Inventory is modelled at aggregate level (all units pooled); per-SKU planning needs SKU-level data.
- LSTM is not part of the default pipeline (it did not beat simple models on 104 points and
  adds a heavy dependency); Prophet is auto-included with `--prophet` when installed.

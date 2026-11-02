# Week 4 plan — dashboard, containerisation, deployment, testing, report

| Day | Goal | Done in this repo |
|---|---|---|
| 1 | Refactor notebooks into package; fix data issues (partial week, leakage) | `src/retailpulse/*`, `tests/test_data.py`, `tests/test_models.py` |
| 2 | Honest forecasting validation + ensemble + intervals | `forecasting.py`, `tests/test_forecasting.py` |
| 3 | Inventory + monitoring (drift, retrain flag, run history) | `inventory.py`, `drift.py`, `monitoring.py` |
| 4 | Pipeline + service layer + FastAPI | `pipeline.py`, `service.py`, `api/main.py`, `tests/test_api.py` |
| 5 | Streamlit dashboard (6 pages, what-if, exports) | `dashboard/app.py` |
| 6 | Docker, compose, CI, deploy to Streamlit Cloud | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` |
| 7 | Final report, screenshots, submission links | `reports/final_report.md`, `docs/SUBMISSION.md` |

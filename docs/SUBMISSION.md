# Submission checklist — a shareable link per deliverable

| Deliverable | How to get the link |
|---|---|
| Source code | Create a GitHub repo, push this folder (`git init && git add . && git commit -m "RetailPulse v2" && git remote add origin <url> && git push -u origin main`), make it public or share access. Link = repo URL. |
| Live dashboard | share.streamlit.io → New app → pick the repo → main file `dashboard/app.py`. Link = `https://<name>.streamlit.app`. The first start runs the pipeline automatically (~10 s). |
| API docs (optional) | Deploy the Docker image to Render / Railway / Fly.io with start command `uvicorn api.main:app --host 0.0.0.0 --port $PORT`; docs at `/docs`. |
| Final report | Upload `reports/final_report.md` (or exported PDF) to Google Drive / GitHub and share "anyone with the link". |
| Power BI dashboards | Publish to Power BI Service → File → Embed/“Publish to web”, or share the exported PDF via Drive. |
| Demo video (optional) | Record the dashboard walk-through (Loom / Drive) and share the link. |

Before submitting: run `pytest -v`, add 3-4 dashboard screenshots to `reports/figures/`, and fill in
your name and project dates in the final report.

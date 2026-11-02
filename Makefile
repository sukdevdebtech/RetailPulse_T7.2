.PHONY: install train test lint dashboard api docker report
install:   ; pip install -r requirements-dev.txt && pip install -e .
train:     ; python -m retailpulse.pipeline
test:      ; pytest -v
lint:      ; ruff check .
dashboard: ; streamlit run dashboard/app.py
api:       ; uvicorn api.main:app --reload --port 8000
report:    ; python scripts/make_report_figures.py
docker:    ; docker compose up --build

"""Central configuration. Every path/threshold can be overridden with an env variable."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("RETAILPULSE_DATA", ROOT / "data"))
SAMPLE_DIR = DATA_DIR / "sample"
ARTIFACT_DIR = Path(os.getenv("RETAILPULSE_ARTIFACTS", ROOT / "artifacts"))

CURRENCY = os.getenv("RETAILPULSE_CURRENCY", "£")  # Online Retail II is in GBP

# Business rules
CHURN_DAYS = int(os.getenv("RETAILPULSE_CHURN_DAYS", 90))  # no purchase for N days => lapsed
FORECAST_HORIZON_WEEKS = 4
CV_INITIAL_WEEKS = 52
CV_STEP_WEEKS = 4
RANDOM_STATE = 42

# Inventory defaults (all can be changed in the dashboard / API)
ORDERING_COST = 50.0       # cost of placing one purchase order
HOLDING_RATE = 0.20        # annual holding cost as a fraction of unit cost
UNIT_COST_RATIO = 0.75     # assumed unit cost = 75% of average selling price
LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95

# Drift thresholds (industry-standard PSI bands)
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

DAILY_FILE = "daily_sales_features.csv"
RFM_FILE = "customer_rfm.csv"

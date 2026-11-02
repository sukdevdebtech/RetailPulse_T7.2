"""Data loading, schema validation and raw-transaction cleaning."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

DAILY_REQUIRED = {"Date", "total_revenue", "total_quantity", "transaction_count", "unique_customers"}
RFM_REQUIRED = {"Customer ID", "recency", "frequency", "monetary"}

# Stock codes in the UCI Online Retail data that are not products
NON_PRODUCT_CODES = {"POST", "D", "DOT", "M", "BANK CHARGES", "AMAZONFEE", "S", "B", "CRUK", "PADS"}

_ALIASES = {
    "InvoiceNo": "Invoice",
    "UnitPrice": "Price",
    "CustomerID": "Customer ID",
}


class SchemaError(ValueError):
    """Raised when an input table is missing required columns."""


def _require(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise SchemaError(f"{name} is missing required columns: {sorted(missing)}")


def load_daily(path: str | Path | None = None) -> pd.DataFrame:
    """Daily sales table on a complete calendar (non-trading days filled with 0)."""
    path = Path(path) if path else config.SAMPLE_DIR / config.DAILY_FILE
    df = pd.read_csv(path, parse_dates=["Date"])
    return normalise_daily(df)


def normalise_daily(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, DAILY_REQUIRED, "daily sales table")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    core = ["total_revenue", "total_quantity", "transaction_count", "unique_customers"]
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    out = df[core].reindex(full).fillna(0.0)
    out.index.name = "Date"
    return out


def load_rfm(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else config.SAMPLE_DIR / config.RFM_FILE
    df = pd.read_csv(path)
    return normalise_rfm(df)


def normalise_rfm(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, RFM_REQUIRED, "customer RFM table")
    out = df[["Customer ID", "recency", "frequency", "monetary"]].copy()
    out = out.dropna().drop_duplicates("Customer ID")
    out = out[(out["frequency"] > 0) & (out["monetary"] > 0) & (out["recency"] >= 0)]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- raw transactions
def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean an Online-Retail style transaction table.

    Steps: rename aliases -> drop duplicates -> remove cancellations (invoice starting 'C')
    -> drop missing customers -> keep positive quantity/price -> drop non-product codes.
    """
    df = raw.rename(columns={k: v for k, v in _ALIASES.items() if k in raw.columns}).copy()
    _require(df, {"Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"},
             "transactions table")
    df = df.drop_duplicates()
    df = df[~df["Invoice"].astype(str).str.upper().str.startswith("C")]
    df = df.dropna(subset=["Customer ID"])
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[~df["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed")
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["Revenue"] = df["Quantity"] * df["Price"]
    return df.reset_index(drop=True)


def transactions_to_daily(clean: pd.DataFrame) -> pd.DataFrame:
    day = clean["InvoiceDate"].dt.normalize()
    g = clean.groupby(day)
    daily = pd.DataFrame({
        "total_revenue": g["Revenue"].sum(),
        "total_quantity": g["Quantity"].sum(),
        "transaction_count": g["Invoice"].nunique(),
        "unique_customers": g["Customer ID"].nunique(),
    })
    daily.index.name = "Date"
    return normalise_daily(daily.reset_index())


def transactions_to_rfm(clean: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    snapshot = snapshot or (clean["InvoiceDate"].max().normalize() + pd.Timedelta(days=1))
    g = clean.groupby("Customer ID")
    rfm = pd.DataFrame({
        "recency": (snapshot - g["InvoiceDate"].max()).dt.days,
        "frequency": g["Invoice"].nunique(),
        "monetary": g["Revenue"].sum(),
    }).reset_index()
    return normalise_rfm(rfm)

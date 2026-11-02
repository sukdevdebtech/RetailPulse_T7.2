import pandas as pd
import pytest

from retailpulse import data


def _raw():
    return pd.DataFrame({
        "Invoice": ["1001", "1001", "C1002", "1003", "1004", "1005", "1006"],
        "StockCode": ["P1", "P2", "P1", "POST", "P3", "P4", "P5"],
        "Quantity": [2, 1, -1, 1, 3, 0, 4],
        "InvoiceDate": ["2011-01-01 10:00", "2011-01-01 10:00", "2011-01-02", "2011-01-03",
                        "2011-01-04", "2011-01-05", "2011-01-06"],
        "Price": [2.0, 5.0, 2.0, 18.0, 1.5, 1.0, 2.0],
        "Customer ID": [1.0, 1.0, 1.0, 2.0, None, 3.0, 3.0],
    })


def test_clean_transactions_removes_bad_rows():
    c = data.clean_transactions(_raw())
    # kept: two lines of invoice 1001 and invoice 1006
    assert sorted(c["Invoice"]) == ["1001", "1001", "1006"]
    assert (c["Revenue"] == c["Quantity"] * c["Price"]).all()


def test_clean_transactions_schema_error():
    with pytest.raises(data.SchemaError):
        data.clean_transactions(pd.DataFrame({"foo": [1]}))


def test_rfm_and_daily_from_transactions():
    c = data.clean_transactions(_raw())
    rfm = data.transactions_to_rfm(c)
    assert set(rfm["Customer ID"]) == {1, 3}
    assert rfm.loc[rfm["Customer ID"] == 1, "monetary"].iloc[0] == 9.0
    daily = data.transactions_to_daily(c)
    assert daily["total_revenue"].sum() == pytest.approx(c["Revenue"].sum())
    assert (daily.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()  # full calendar


def test_load_sample_data():
    d, r = data.load_daily(), data.load_rfm()
    assert len(d) > 700 and d["total_revenue"].min() >= 0
    assert r["Customer ID"].is_unique and (r["monetary"] > 0).all()

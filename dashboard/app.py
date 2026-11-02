"""RetailPulse v2 dashboard.   streamlit run dashboard/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from retailpulse import config, inventory, monitoring, pipeline  # noqa: E402

CUR = config.CURRENCY
st.set_page_config(page_title="RetailPulse v2", page_icon="📊", layout="wide")


# ------------------------------------------------------------------ data access
@st.cache_data(show_spinner="Running the RetailPulse pipeline (first start only)...")
def load_all(artifact_dir: str, _stamp: float):
    import json

    d = Path(artifact_dir)
    if not (d / "summary.json").exists():
        pipeline.run(d)
    return {
        "summary": json.loads((d / "summary.json").read_text()),
        "daily": pd.read_csv(d / "daily_sales.csv", parse_dates=["Date"]),
        "weekly": pd.read_csv(d / "weekly_sales.csv", parse_dates=["Date"]),
        "customers": pd.read_csv(d / "customer_scores.csv"),
        "at_risk": pd.read_csv(d / "at_risk_customers.csv"),
        "importance": pd.read_csv(d / "churn_feature_importance.csv"),
        "seg": pd.read_csv(d / "segment_summary.csv"),
        "forecast": pd.read_csv(d / "forecast.csv", parse_dates=["date"]),
        "cv": pd.read_csv(d / "cv_scores.csv"),
        "cvp": pd.read_csv(d / "cv_predictions.csv", parse_dates=["date"]),
        "scan": pd.read_csv(d / "service_level_scan.csv"),
        "drift": pd.read_csv(d / "drift_report.csv"),
        "runs": monitoring.read_history(d),
    }


def money(x: float) -> str:
    return f"{CUR}{x:,.0f}"


art_dir = config.ARTIFACT_DIR
stamp = (art_dir / "summary.json").stat().st_mtime if (art_dir / "summary.json").exists() else 0.0
D = load_all(str(art_dir), stamp)
S = D["summary"]

# ---------------------------------------------------------------------- sidebar
st.sidebar.title("📊 RetailPulse v2")
page = st.sidebar.radio("Navigate", ["Executive Overview", "Sales Analytics", "Customer Intelligence",
                                     "Demand Forecast", "Inventory Planner", "Model Monitoring"])
st.sidebar.caption(f"Data: {S['source']}")
st.sidebar.caption(f"Pipeline v{S['version']} · {S['rows']['customers']:,} customers · "
                   f"{S['rows']['weeks']} weeks")
if st.sidebar.button("Re-run pipeline"):
    pipeline.run(art_dir)
    st.cache_data.clear()
    st.rerun()

# ------------------------------------------------------------------------ pages
if page == "Executive Overview":
    st.title("Executive Overview")
    daily, cust = D["daily"], D["customers"]
    nxt = D["forecast"]["forecast"].sum()
    last4 = D["weekly"]["total_revenue"].tail(4).sum()
    risk_rev = D["at_risk"]["revenue_at_risk"].sum()
    c = st.columns(4)
    c[0].metric("Total revenue", money(daily["total_revenue"].sum()))
    c[1].metric("Customers", f"{len(cust):,}")
    c[2].metric("Next 4-week forecast", money(nxt), f"{(nxt / last4 - 1) * 100:+.1f}% vs last 4 weeks")
    c[3].metric("Revenue at risk (top 100 active)", money(risk_rev))
    c = st.columns(4)
    c[0].metric("Forecast error (WAPE, CV)", f"{S['forecast']['ensemble']['WAPE']:.1f}%")
    c[1].metric("Churn model AUC", f"{S['churn']['auc_holdout']:.2f}")
    c[2].metric("Simulated fill rate", f"{S['inventory']['simulation']['fill_rate_pct']:.1f}%")
    c[3].metric("Drifting features", f"{S['data_drift_features_flagged']} / 3")

    wk = D["weekly"]
    fig = go.Figure()
    fig.add_scatter(x=wk["Date"], y=wk["total_revenue"], name="Actual", line=dict(width=2))
    f = D["forecast"]
    fig.add_scatter(x=f["date"], y=f["upper"], line=dict(width=0), showlegend=False)
    fig.add_scatter(x=f["date"], y=f["lower"], fill="tonexty", line=dict(width=0),
                    name="80% interval", fillcolor="rgba(255,127,14,0.25)")
    fig.add_scatter(x=f["date"], y=f["forecast"], name="Forecast", line=dict(dash="dash", width=3))
    fig.update_layout(title="Weekly revenue and 4-week forecast", height=420, yaxis_title=CUR)
    st.plotly_chart(fig, use_container_width=True)
    if S["performance_drift"]["retrain_recommended"]:
        st.warning("Model performance has degraded - retraining is recommended.")

elif page == "Sales Analytics":
    st.title("Sales Analytics")
    daily = D["daily"].copy()
    lo, hi = daily["Date"].min().date(), daily["Date"].max().date()
    picked = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    a, b = (picked[0], picked[1]) if isinstance(picked, (tuple, list)) and len(picked) == 2 else (lo, hi)
    m = daily[(daily["Date"].dt.date >= a) & (daily["Date"].dt.date <= b)]
    c = st.columns(4)
    c[0].metric("Revenue", money(m["total_revenue"].sum()))
    c[1].metric("Orders", f"{m['transaction_count'].sum():,.0f}")
    c[2].metric("Units", f"{m['total_quantity'].sum():,.0f}")
    c[3].metric("Avg order value", money(m["total_revenue"].sum() / max(m["transaction_count"].sum(), 1)))
    gran = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True, index=1)
    rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[gran]
    ts = m.set_index("Date")["total_revenue"].resample(rule).sum().reset_index()
    st.plotly_chart(px.line(ts, x="Date", y="total_revenue", title=f"{gran} revenue"),
                    use_container_width=True)
    mm = m.assign(month=m["Date"].dt.month_name(), year=m["Date"].dt.year)
    heat = mm.pivot_table(index="year", columns=m["Date"].dt.month, values="total_revenue", aggfunc="sum")
    st.plotly_chart(px.imshow(heat, aspect="auto", labels=dict(x="Month", y="Year", color=CUR),
                              title="Seasonality heatmap (revenue by month)"), use_container_width=True)
    st.download_button("Download filtered data (CSV)", m.to_csv(index=False), "sales_filtered.csv")

elif page == "Customer Intelligence":
    st.title("Customer Intelligence")
    cust = D["customers"]
    t1, t2, t3 = st.tabs(["RFM segments", "K-Means clusters", "Churn & at-risk"])
    with t1:
        seg = D["seg"]
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(seg, x="segment", y="revenue_share_pct", title="Revenue share by segment (%)"),
                        use_container_width=True)
        c2.plotly_chart(px.pie(seg, names="segment", values="customers", title="Customers by segment"),
                        use_container_width=True)
        st.dataframe(seg, use_container_width=True)
    with t2:
        st.caption(f"k = {S['clustering']['k']} chosen by silhouette ({S['clustering']['silhouette']}).")
        samp = cust.sample(min(2500, len(cust)), random_state=1)
        st.plotly_chart(px.scatter(samp, x="recency", y="monetary", color="cluster_name", log_y=True,
                                   hover_data=["frequency"], title="Clusters: recency vs spend (log)"),
                        use_container_width=True)
    with t3:
        ch = S["churn"]
        st.info(f"Lapsed = no purchase in {ch['churn_days']}+ days. Recency is excluded from model "
                f"inputs to avoid label leakage. Hold-out AUC {ch['auc_holdout']}, "
                f"5-fold CV AUC {ch['auc_cv5']}, recall {ch['recall@0.5']}.")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(D["importance"], x="feature", y="importance",
                               title="Permutation importance (AUC drop)"), use_container_width=True)
        tiers = cust[cust["lapsed"] == 0]["risk_tier"].value_counts().reset_index()
        c2.plotly_chart(px.bar(tiers, x="risk_tier", y="count", title="Active customers by risk tier"),
                        use_container_width=True)
        st.subheader("Win-back priority list (active customers, ranked by expected revenue loss)")
        st.dataframe(D["at_risk"][["Customer ID", "recency", "frequency", "monetary",
                                   "churn_probability", "revenue_at_risk"]].head(50),
                     use_container_width=True)
        st.download_button("Download at-risk list (CSV)", D["at_risk"].to_csv(index=False),
                           "at_risk_customers.csv")

elif page == "Demand Forecast":
    st.title("Demand Forecast")
    f = D["forecast"].copy()
    adj = st.slider("What-if: demand adjustment (promotions, shocks)", -30, 30, 0, format="%d%%")
    f[["forecast", "lower", "upper"]] *= 1 + adj / 100
    c = st.columns(3)
    c[0].metric("4-week total", money(f["forecast"].sum()))
    c[1].metric("Low / high case", f"{money(f['lower'].sum())} - {money(f['upper'].sum())}")
    c[2].metric("Ensemble WAPE (CV)", f"{S['forecast']['ensemble']['WAPE']:.1f}%")
    st.dataframe(f[["date", "forecast", "lower", "upper"]].round(0), use_container_width=True)
    st.subheader("Model comparison (walk-forward cross-validation)")
    st.dataframe(D["cv"].round(2), use_container_width=True)
    cvp = D["cvp"]
    st.plotly_chart(px.line(cvp.melt(id_vars=["date"], value_vars=["actual", "ensemble"]), x="date",
                            y="value", color="variable", title="Out-of-sample: actual vs ensemble"),
                    use_container_width=True)
    st.caption("Ensemble weights: " + ", ".join(f"{k} {v:.0%}" for k, v in S["forecast"]["weights"].items())
               + f" · interval coverage in CV {S['forecast']['interval_coverage']:.0%}")

elif page == "Inventory Planner":
    st.title("Inventory Planner")
    daily = D["daily"].set_index("Date")
    c = st.columns(4)
    sl = c[0].select_slider("Service level", [0.90, 0.95, 0.975, 0.98, 0.99], value=0.95)
    lt = c[1].slider("Lead time (days)", 1, 30, config.LEAD_TIME_DAYS)
    oc = c[2].number_input("Ordering cost", 1.0, 1000.0, float(config.ORDERING_COST))
    hr = c[3].slider("Holding rate / yr", 0.05, 0.5, float(config.HOLDING_RATE))
    plan = inventory.plan_from_demand(daily["total_quantity"], S["avg_price"], oc, hr, lt, sl)
    sim = inventory.simulate(daily["total_quantity"], plan.reorder_point, plan.eoq, lt)
    sm = inventory.sim_metrics(sim)
    c = st.columns(4)
    c[0].metric("EOQ (units)", f"{plan.eoq:,.0f}")
    c[1].metric("Safety stock", f"{plan.safety_stock:,.0f}")
    c[2].metric("Reorder point", f"{plan.reorder_point:,.0f}")
    c[3].metric("Simulated fill rate", f"{sm['fill_rate_pct']:.1f}%", f"{sm['stockout_days']} stockout days",
                delta_color="inverse")
    st.plotly_chart(px.line(sim.reset_index(), x="Date", y="on_hand", title="Simulated on-hand stock"),
                    use_container_width=True)
    st.subheader("Service-level trade-off")
    st.dataframe(D["scan"], use_container_width=True)
    st.caption("Demand is measured in units. Costs: unit cost = 75% of average price; "
               "all inputs are editable above.")

else:
    st.title("Model Monitoring")
    st.subheader("Data drift (last 180 days vs history)")
    st.dataframe(D["drift"], use_container_width=True)
    st.caption("PSI < 0.10 stable · 0.10-0.25 moderate · > 0.25 significant. "
               "Some drift is expected here because the business grew year over year.")
    pf = S["performance_drift"]
    c = st.columns(3)
    c[0].metric("Baseline WAPE", f"{pf['baseline_wape']}%")
    c[1].metric("Recent WAPE", f"{pf['recent_wape']}%")
    c[2].metric("Retrain recommended", "YES" if pf["retrain_recommended"] else "No")
    st.subheader("Pipeline run history")
    if D["runs"]:
        st.dataframe(pd.DataFrame([{"timestamp": r["timestamp"], **r["metrics"]} for r in D["runs"]]),
                     use_container_width=True)

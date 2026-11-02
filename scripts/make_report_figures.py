"""Create the figures used in reports/final_report.md from pipeline artifacts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from retailpulse import config  # noqa: E402

A, OUT = config.ARTIFACT_DIR, ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name)
    plt.close()


wk = pd.read_csv(A / "weekly_sales.csv", parse_dates=["Date"])
fc = pd.read_csv(A / "forecast.csv", parse_dates=["date"])
plt.figure(figsize=(9, 4))
plt.plot(wk["Date"], wk["total_revenue"], label="Actual")
plt.fill_between(fc["date"], fc["lower"], fc["upper"], alpha=0.3, color="tab:orange", label="80% interval")
plt.plot(fc["date"], fc["forecast"], "--", color="tab:orange", lw=2.5, label="Forecast")
plt.title("Weekly revenue and 4-week forecast"); plt.legend(); save("01_forecast.png")

cv = pd.read_csv(A / "cv_scores.csv")
plt.figure(figsize=(7, 3.6))
cv = cv.sort_values("WAPE", ascending=False)
plt.barh(cv["model"], cv["WAPE"], color=["tab:green" if m == "ensemble" else "tab:blue" for m in cv["model"]])
plt.xlabel("WAPE % (walk-forward CV, lower is better)"); plt.title("Forecast model comparison"); save("02_model_comparison.png")

seg = pd.read_csv(A / "segment_summary.csv")
plt.figure(figsize=(8, 3.8))
plt.bar(seg["segment"], seg["revenue_share_pct"], color="tab:purple"); plt.xticks(rotation=35, ha="right")
plt.ylabel("% of revenue"); plt.title("Revenue share by RFM segment"); save("03_segments.png")

imp = pd.read_csv(A / "churn_feature_importance.csv")
plt.figure(figsize=(6, 3.2)); plt.barh(imp["feature"], imp["importance"], color="tab:red")
plt.xlabel("AUC drop when shuffled"); plt.title("Churn model: permutation importance"); save("04_churn_importance.png")

scan = pd.read_csv(A / "service_level_scan.csv")
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(scan["service_level"] * 100, scan["simulated_fill_rate_pct"], "o-", label="Fill rate %")
ax.set_xlabel("Target service level %"); ax.set_ylabel("Simulated fill rate %")
ax2 = ax.twinx(); ax2.bar(scan["service_level"] * 100, scan["annual_holding_cost"], alpha=0.25, width=0.4, color="gray")
ax2.set_ylabel("Annual holding cost"); plt.title("Service level vs cost"); save("05_service_level.png")
print("figures written to", OUT)

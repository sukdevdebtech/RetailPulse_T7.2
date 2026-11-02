"""Data-drift (PSI + KS) and model-performance drift detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index using reference-quantile bins."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def status_from_psi(value: float) -> str:
    if value < config.PSI_MODERATE:
        return "Stable"
    if value < config.PSI_SIGNIFICANT:
        return "Moderate drift"
    return "Significant drift"


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        ref, cur = reference[f].dropna(), current[f].dropna()
        ks = stats.ks_2samp(ref, cur)
        p = psi(ref, cur)
        rows.append({"feature": f, "psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                     "ks_pvalue": round(float(ks.pvalue), 6), "status": status_from_psi(p),
                     "ref_mean": round(float(ref.mean()), 2), "cur_mean": round(float(cur.mean()), 2)})
    return pd.DataFrame(rows)


def split_reference_current(df: pd.DataFrame, current_days: int = 180):
    """Reference = everything before the last `current_days`; current = the last `current_days`."""
    cutoff = df.index.max() - pd.Timedelta(days=current_days)
    return df.loc[:cutoff], df.loc[cutoff + pd.Timedelta(days=1):]


def performance_drift(baseline_wape: float, recent_wape: float, tolerance: float = 1.5) -> dict:
    """Flag model degradation when recent error exceeds baseline error by `tolerance`x."""
    ratio = recent_wape / baseline_wape if baseline_wape else float("inf")
    return {"baseline_wape": round(baseline_wape, 2), "recent_wape": round(recent_wape, 2),
            "ratio": round(ratio, 2), "retrain_recommended": bool(ratio > tolerance)}

"""Customer segmentation: RFM quintile scoring, rule-based labels and K-Means."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config


def _quintile(s: pd.Series, ascending: bool = True) -> pd.Series:
    """1-5 score by rank (rank-first avoids qcut failures on heavily tied values)."""
    r = s.rank(method="first", ascending=ascending)
    return pd.qcut(r, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    out = rfm.copy()
    out["r_score"] = _quintile(out["recency"], ascending=False)   # recent = high score
    out["f_score"] = _quintile(out["frequency"])
    out["m_score"] = _quintile(out["monetary"])
    out["rfm_score"] = out[["r_score", "f_score", "m_score"]].sum(axis=1)
    out["segment"] = [label_segment(r, f, m) for r, f, m in
                      zip(out["r_score"], out["f_score"], out["m_score"])]
    return out


def label_segment(r: int, f: int, m: int) -> str:
    fm = (f + m) / 2
    if r >= 4 and fm >= 4:
        return "Champions"
    if r >= 3 and fm >= 3.5:
        return "Loyal"
    if r >= 4 and fm < 2:
        return "New / Promising"
    if r >= 4:
        return "Potential Loyalists"
    if r == 1 and fm >= 4:
        return "Can't Lose Them"
    if r <= 2 and fm >= 3.5:
        return "At Risk"
    if r == 1 and fm < 2:
        return "Lost"
    if r <= 2:
        return "Hibernating"
    return "Needs Attention"


_CLUSTER_NAMES = ["VIP", "Core", "Growing", "Occasional", "Fading", "Dormant"]


def kmeans_segments(rfm: pd.DataFrame, k_range=range(3, 7), random_state: int | None = None):
    """K-Means on log-scaled RFM. k is chosen by silhouette score; clusters are named by value."""
    rs = config.RANDOM_STATE if random_state is None else random_state
    X = np.log1p(rfm[["recency", "frequency", "monetary"]].astype(float))
    Xs = StandardScaler().fit_transform(X)
    best = (-1.0, None, None)
    sil = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=rs).fit(Xs)
        score = silhouette_score(Xs, km.labels_, sample_size=min(2500, len(Xs)), random_state=rs)
        sil[k] = round(float(score), 4)
        if score > best[0]:
            best = (score, k, km)
    _, k, km = best
    labels = km.labels_
    # value score per cluster: low recency, high frequency/monetary => high value
    cen = pd.DataFrame(km.cluster_centers_, columns=["r", "f", "m"])
    value = (-cen["r"] + cen["f"] + cen["m"]).rank(ascending=False, method="first").astype(int) - 1
    names = {c: _CLUSTER_NAMES[min(v, len(_CLUSTER_NAMES) - 1)] if k <= len(_CLUSTER_NAMES)
             else f"Cluster {v}" for c, v in value.items()}
    out = rfm.copy()
    out["cluster"] = labels
    out["cluster_name"] = [names[c] for c in labels]
    return out, {"k": int(k), "silhouette": round(float(best[0]), 4), "silhouette_by_k": sil}


def segment_summary(df: pd.DataFrame, col: str = "segment") -> pd.DataFrame:
    g = df.groupby(col)
    out = pd.DataFrame({
        "customers": g.size(),
        "avg_recency_days": g["recency"].mean().round(1),
        "avg_frequency": g["frequency"].mean().round(1),
        "avg_monetary": g["monetary"].mean().round(0),
        "total_revenue": g["monetary"].sum().round(0),
    })
    out["revenue_share_pct"] = (out["total_revenue"] / out["total_revenue"].sum() * 100).round(1)
    return out.sort_values("total_revenue", ascending=False)

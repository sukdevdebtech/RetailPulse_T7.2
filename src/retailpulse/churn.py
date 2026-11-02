"""Churn (lapse) propensity model.

Definition: a customer is *lapsed* when recency > CHURN_DAYS.

Why recency is NOT a model input: the label is a threshold on recency, so including it makes
the model trivially perfect (AUC = 1.00) and useless. The model instead learns from purchase
*behaviour* (frequency, spend, basket value) which customers look like lapsed ones. It is then
applied to still-active customers to find who is drifting - the actionable list.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from . import config

FEATURES = ["log_frequency", "log_monetary", "log_avg_order_value"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["log_frequency"] = np.log1p(df["frequency"].astype(float))
    X["log_monetary"] = np.log1p(df["monetary"].astype(float))
    X["log_avg_order_value"] = np.log1p(df["monetary"].astype(float) / df["frequency"].astype(float))
    return X[FEATURES]


def make_labels(df: pd.DataFrame, churn_days: int | None = None) -> pd.Series:
    return (df["recency"] > (churn_days or config.CHURN_DAYS)).astype(int)


@dataclass
class ChurnResult:
    model: HistGradientBoostingClassifier
    metrics: dict
    importance: pd.DataFrame
    scored: pd.DataFrame


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          l2_regularization=1.0, random_state=config.RANDOM_STATE)


def risk_tier(p: float) -> str:
    return "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"


def train(df: pd.DataFrame, churn_days: int | None = None) -> ChurnResult:
    X, y = build_features(df), make_labels(df, churn_days)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y,
                                              random_state=config.RANDOM_STATE)
    model = _model().fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, 1]
    pred = (p_te >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_te == 1)).sum())
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / max(int(y_te.sum()), 1)
    # 5-fold out-of-fold AUC on all data as a stability check
    oof = cross_val_predict(_model(), X, y, cv=StratifiedKFold(5, shuffle=True,
                            random_state=config.RANDOM_STATE), method="predict_proba")[:, 1]
    m = {
        "auc_holdout": round(float(roc_auc_score(y_te, p_te)), 4),
        "auc_cv5": round(float(roc_auc_score(y, oof)), 4),
        "avg_precision": round(float(average_precision_score(y_te, p_te)), 4),
        "precision@0.5": round(prec, 4),
        "recall@0.5": round(rec, 4),
        "base_rate": round(float(y.mean()), 4),
        "churn_days": churn_days or config.CHURN_DAYS,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    }
    pi = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=10,
                                random_state=config.RANDOM_STATE)
    imp = pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=False)

    final = _model().fit(X, y)           # refit on everything for scoring
    scored = df.copy()
    scored["lapsed"] = y.values
    scored["churn_probability"] = oof    # out-of-fold => honest probabilities for every customer
    scored["risk_tier"] = scored["churn_probability"].map(risk_tier)
    return ChurnResult(model=final, metrics=m, importance=imp, scored=scored)


def at_risk_active(scored: pd.DataFrame, top: int = 50) -> pd.DataFrame:
    """Active customers (not yet lapsed) ranked by expected revenue loss."""
    act = scored[scored["lapsed"] == 0].copy()
    act["revenue_at_risk"] = act["churn_probability"] * act["monetary"]
    return act.sort_values("revenue_at_risk", ascending=False).head(top)


def predict_proba(model, frequency: float, monetary: float) -> float:
    row = pd.DataFrame({"frequency": [frequency], "monetary": [monetary]})
    return float(model.predict_proba(build_features(row))[:, 1][0])

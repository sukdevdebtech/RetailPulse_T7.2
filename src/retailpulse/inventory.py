"""Inventory optimisation: EOQ, safety stock, reorder point and policy simulation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config


def z_value(service_level: float) -> float:
    if not 0.5 < service_level < 1:
        raise ValueError("service_level must be between 0.5 and 1 (exclusive)")
    return NormalDist().inv_cdf(service_level)


def eoq(annual_demand: float, ordering_cost: float, holding_cost_per_unit: float) -> float:
    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_per_unit <= 0:
        raise ValueError("annual_demand, ordering_cost and holding_cost_per_unit must be > 0")
    return float(np.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit))


def safety_stock(daily_std: float, lead_time_days: float, service_level: float) -> float:
    return float(z_value(service_level) * daily_std * np.sqrt(lead_time_days))


def reorder_point(daily_mean: float, lead_time_days: float, safety: float) -> float:
    return float(daily_mean * lead_time_days + safety)


@dataclass
class InventoryPlan:
    daily_mean: float
    daily_std: float
    annual_demand: float
    unit_cost: float
    holding_cost_per_unit: float
    eoq: float
    safety_stock: float
    reorder_point: float
    orders_per_year: float
    annual_ordering_cost: float
    annual_holding_cost: float
    service_level: float
    lead_time_days: float

    def to_dict(self) -> dict:
        return {k: round(float(v), 2) for k, v in asdict(self).items()}


def plan_from_demand(daily_units: pd.Series, avg_price: float,
                     ordering_cost: float | None = None, holding_rate: float | None = None,
                     lead_time_days: float | None = None, service_level: float | None = None,
                     unit_cost_ratio: float | None = None) -> InventoryPlan:
    ordering_cost = config.ORDERING_COST if ordering_cost is None else ordering_cost
    holding_rate = config.HOLDING_RATE if holding_rate is None else holding_rate
    lead_time_days = config.LEAD_TIME_DAYS if lead_time_days is None else lead_time_days
    service_level = config.SERVICE_LEVEL if service_level is None else service_level
    ratio = config.UNIT_COST_RATIO if unit_cost_ratio is None else unit_cost_ratio

    mean, std = float(daily_units.mean()), float(daily_units.std(ddof=1))
    annual = mean * 365
    unit_cost = avg_price * ratio
    h = unit_cost * holding_rate
    q = eoq(annual, ordering_cost, h)
    ss = safety_stock(std, lead_time_days, service_level)
    rop = reorder_point(mean, lead_time_days, ss)
    n_orders = annual / q
    return InventoryPlan(mean, std, annual, unit_cost, h, q, ss, rop, n_orders,
                         n_orders * ordering_cost, (q / 2 + ss) * h, service_level, lead_time_days)


def simulate(daily_units: pd.Series, reorder_pt: float, order_qty: float,
             lead_time_days: int, initial_stock: float | None = None) -> pd.DataFrame:
    """(s, Q) policy: when inventory position <= s, order Q (arrives after the lead time).

    Unmet demand is lost. Inventory position = on-hand + on-order, which avoids the classic
    bug of re-ordering every day while a previous order is still in transit.
    """
    lead_time_days = int(round(lead_time_days))
    on_hand = float(reorder_pt + order_qty if initial_stock is None else initial_stock)
    pipeline: list[tuple[int, float]] = []
    rows = []
    for i, demand in enumerate(daily_units.values.astype(float)):
        arrived = sum(q for t, q in pipeline if t == i)
        pipeline = [(t, q) for t, q in pipeline if t != i]
        on_hand += arrived
        sold = min(on_hand, demand)
        short = demand - sold
        on_hand -= sold
        position = on_hand + sum(q for _, q in pipeline)
        ordered = 0.0
        if position <= reorder_pt:
            ordered = float(order_qty)
            pipeline.append((i + lead_time_days, ordered))
        rows.append((demand, sold, short, on_hand, ordered))
    out = pd.DataFrame(rows, columns=["demand", "sold", "shortage", "on_hand", "ordered"],
                       index=daily_units.index)
    return out


def sim_metrics(sim: pd.DataFrame) -> dict:
    trading = sim[sim["demand"] > 0]
    return {
        "fill_rate_pct": round(float(sim["sold"].sum() / max(sim["demand"].sum(), 1) * 100), 2),
        "stockout_days": int((trading["shortage"] > 0).sum()),
        "trading_days": int(len(trading)),
        "orders_placed": int((sim["ordered"] > 0).sum()),
        "avg_on_hand": round(float(sim["on_hand"].mean()), 1),
    }


def service_level_scan(daily_units: pd.Series, avg_price: float, levels=(0.90, 0.95, 0.98, 0.99),
                       **kwargs) -> pd.DataFrame:
    """Trade-off table: higher service level => more safety stock => higher holding cost."""
    rows = []
    for sl in levels:
        p = plan_from_demand(daily_units, avg_price, service_level=sl, **kwargs)
        s = sim_metrics(simulate(daily_units, p.reorder_point, p.eoq, p.lead_time_days))
        rows.append({"service_level": sl, "safety_stock": round(p.safety_stock),
                     "reorder_point": round(p.reorder_point), "eoq": round(p.eoq),
                     "annual_holding_cost": round(p.annual_holding_cost),
                     "simulated_fill_rate_pct": s["fill_rate_pct"],
                     "stockout_days": s["stockout_days"]})
    return pd.DataFrame(rows)

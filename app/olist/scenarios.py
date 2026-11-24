"""What-if business scenarios with every input labelled as observed or assumed.

Each scenario starts from an observed monthly baseline (built from real orders by
``observed_baseline``) and applies explicit assumptions. Outputs are HYPOTHETICAL: they show
what would follow if the assumptions held, not forecasts and not measured effects. Where a
scenario borrows an observed association (e.g. late vs on-time reviews), it says that it
treats the association as causal, which makes the result an optimistic bound.

Money follows the metric layer: GMV = item revenue + freight; marketplace revenue and
contribution are estimates from the take-rate and payment-cost assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Input:
    name: str
    value: float
    label: str
    source: str  # "observed" or "assumption"


@dataclass(frozen=True)
class ObservedBaseline:
    months: float
    orders_per_month: float
    new_customers_per_month: float
    item_revenue_per_order: float
    freight_per_order: float
    payment_per_order: float
    late_rate: float
    review_on_time: float
    review_late: float
    repeat_rate_after_on_time: float
    repeat_rate_after_late: float
    orders_per_repeater: float
    cancellation_rate: float
    orders_placed_per_month: float
    seller_late_rates: np.ndarray
    seller_orders: np.ndarray
    item_value: np.ndarray
    freight: np.ndarray

    @property
    def aov(self) -> float:
        return self.item_revenue_per_order + self.freight_per_order


FOLLOW_UP_DAYS = 180


def repeat_after_first_order(orders: pd.DataFrame) -> tuple[float, float]:
    """Share of customers placing a second valid order 1-180 days after a delivered first
    order, for on-time and for late first orders. Only first orders with a full 180-day
    follow-up inside the data count (no right-censoring); same-day second orders are split
    baskets, not returns, and are excluded."""
    v = orders[orders["is_valid"] == 1]
    if v.empty:
        return float("nan"), float("nan")
    date = pd.to_datetime(v["purchase_date"])
    last = date.max()
    first = v[(v["customer_order_seq"] == 1) & (v["is_delivered"] == 1)]
    first = first[
        pd.to_datetime(first["purchase_date"]) <= last - pd.Timedelta(days=FOLLOW_UP_DAYS)
    ]
    second = v[v["customer_order_seq"] == 2].set_index("customer_unique_id")["purchase_date"]
    gap = (
        pd.to_datetime(first["customer_unique_id"].map(second))
        - pd.to_datetime(first["purchase_date"])
    ).dt.days
    came_back = gap.between(1, FOLLOW_UP_DAYS)
    late = first["is_late"] == 1

    def rate(mask: pd.Series) -> float:
        return float(came_back[mask].mean()) if mask.any() else float("nan")

    return rate(~late), rate(late)


def observed_baseline(
    orders: pd.DataFrame, repeat_history: pd.DataFrame | None = None
) -> ObservedBaseline:
    """Monthly averages over the months present in ``orders`` (a filtered fct_orders-like
    frame). The repeat gap between on-time and late first orders is measured on
    ``repeat_history`` (default: ``orders``), see ``repeat_after_first_order``."""
    v = orders[orders["is_valid"] == 1]
    months = max(pd.to_datetime(v["purchase_month"]).nunique(), 1) if len(v) else 1
    d = v[v["is_delivered"] == 1]
    late = d["is_late"] == 1
    rr_on_time, rr_late = repeat_after_first_order(
        repeat_history if repeat_history is not None else orders
    )
    repeaters = set(v.loc[v["customer_order_seq"] >= 2, "customer_unique_id"])
    n_rep = len(repeaters)
    rep_orders = int((v["customer_order_seq"] >= 2).sum())
    seller = (
        d.groupby("main_seller_id", observed=True)["is_late"].agg(["mean", "size"])
        if "main_seller_id" in d and len(d)
        else pd.DataFrame({"mean": [], "size": []})
    )

    def mean(s: pd.Series) -> float:
        return float(s.mean()) if len(s) else float("nan")

    return ObservedBaseline(
        months=float(months),
        orders_per_month=len(v) / months,
        new_customers_per_month=float((v["customer_order_seq"] == 1).sum()) / months,
        item_revenue_per_order=mean(v["item_revenue"]),
        freight_per_order=mean(v["freight_value"]),
        payment_per_order=mean(v["payment_value"].dropna()),
        late_rate=mean(late.astype(float)),
        review_on_time=mean(d.loc[~late, "review_score"].dropna()),
        review_late=mean(d.loc[late, "review_score"].dropna()),
        repeat_rate_after_on_time=rr_on_time,
        repeat_rate_after_late=rr_late,
        orders_per_repeater=rep_orders / n_rep if n_rep else 1.0,
        cancellation_rate=mean(orders["is_cancelled"].astype(float)),
        orders_placed_per_month=len(orders) / months,
        seller_late_rates=seller["mean"].to_numpy(float),
        seller_orders=seller["size"].to_numpy(float),
        item_value=v["item_revenue"].to_numpy(float),
        freight=v["freight_value"].to_numpy(float),
    )


def _money(
    orders: float,
    item_per_order: float,
    freight_per_order: float,
    take: float,
    pay_cost: float,
    subsidy: float = 0.0,
) -> dict:
    item = orders * item_per_order
    gmv = item + orders * freight_per_order
    mrev = take * item
    return {
        "orders": orders,
        "gmv": gmv,
        "item_revenue": item,
        "marketplace_revenue_estimate": mrev,
        # customers pay GMV (payment cost on it); the marketplace pays any freight subsidy
        "contribution_estimate": mrev - pay_cost * gmv - subsidy,
    }


def _result(
    name,
    b: ObservedBaseline,
    inputs: list[Input],
    base: dict,
    scen: dict,
    notes: str,
    extra: dict | None = None,
) -> dict:
    rows = [
        {"metric": k, "baseline": base[k], "scenario": scen[k], "change": scen[k] - base[k]}
        for k in base
    ]
    return {
        "scenario": name,
        "hypothetical": True,
        "inputs": [i.__dict__ for i in inputs],
        "monthly": rows,
        "notes": notes,
        **(extra or {}),
    }


def _common(b: ObservedBaseline, take: float, pay: float) -> list[Input]:
    return [
        Input("orders_per_month", b.orders_per_month, "Valid orders per month", "observed"),
        Input("aov", b.aov, "GMV per order (R$)", "observed"),
        Input("take_rate", take, "Marketplace take rate", "assumption"),
        Input("payment_cost_rate", pay, "Payment cost share", "assumption"),
    ]


def faster_delivery(
    b: ObservedBaseline,
    late_reduction: float = 0.5,
    take_rate: float = 0.10,
    payment_cost_rate: float = 0.02,
) -> dict:
    """A share ``late_reduction`` of late deliveries arrive on time instead."""
    new_late = b.late_rate * (1 - late_reduction)
    converted_new = b.new_customers_per_month * b.late_rate * late_reduction
    extra_repeaters = converted_new * (b.repeat_rate_after_on_time - b.repeat_rate_after_late)
    extra_orders = extra_repeaters * b.orders_per_repeater
    review_gain = (b.late_rate - new_late) * (b.review_on_time - b.review_late)
    base = _money(
        b.orders_per_month,
        b.item_revenue_per_order,
        b.freight_per_order,
        take_rate,
        payment_cost_rate,
    )
    scen = _money(
        b.orders_per_month + extra_orders,
        b.item_revenue_per_order,
        b.freight_per_order,
        take_rate,
        payment_cost_rate,
    )
    inputs = _common(b, take_rate, payment_cost_rate) + [
        Input("late_rate", b.late_rate, "Late-delivery rate", "observed"),
        Input("late_reduction", late_reduction, "Share of late deliveries fixed", "assumption"),
        Input(
            "review_gap", b.review_on_time - b.review_late, "On-time minus late review", "observed"
        ),
        Input(
            "repeat_gap",
            b.repeat_rate_after_on_time - b.repeat_rate_after_late,
            "Repeat within 180 days after an on-time vs a late first order",
            "observed",
        ),
    ]
    return _result(
        "Faster delivery",
        b,
        inputs,
        base,
        scen,
        "Treats the observed review and repeat gaps between on-time and late first orders as "
        "causal, so this is an upper bound. Extra repeat orders accrue over the following "
        "months; the cost of faster logistics is not modelled.",
        {"late_rate_after": new_late, "avg_review_change": review_gain},
    )


def shipping_subsidy(
    b: ObservedBaseline,
    threshold: float = 150.0,
    subsidy_share: float = 1.0,
    order_lift: float = 0.03,
    take_rate: float = 0.10,
    payment_cost_rate: float = 0.02,
) -> dict:
    """Pay ``subsidy_share`` of the freight on orders with item value >= ``threshold``."""
    qualifies = b.item_value >= threshold
    share_q = float(qualifies.mean()) if len(qualifies) else 0.0
    freight_q = float(b.freight[qualifies].mean()) if qualifies.any() else 0.0
    orders = b.orders_per_month * (1 + order_lift)
    subsidy = orders * share_q * freight_q * subsidy_share
    base = _money(
        b.orders_per_month,
        b.item_revenue_per_order,
        b.freight_per_order,
        take_rate,
        payment_cost_rate,
    )
    freight_charged = b.freight_per_order - share_q * freight_q * subsidy_share
    scen = _money(
        orders, b.item_revenue_per_order, freight_charged, take_rate, payment_cost_rate, subsidy
    )
    inputs = _common(b, take_rate, payment_cost_rate) + [
        Input(
            "qualifying_share", share_q, f"Orders with item value >= R$ {threshold:.0f}", "observed"
        ),
        Input(
            "freight_qualifying", freight_q, "Mean freight of qualifying orders (R$)", "observed"
        ),
        Input("threshold", threshold, "Free-shipping threshold (R$)", "assumption"),
        Input(
            "subsidy_share", subsidy_share, "Share of freight paid by the marketplace", "assumption"
        ),
        Input("order_lift", order_lift, "Relative lift in orders from the offer", "assumption"),
    ]
    return _result(
        "Shipping subsidy",
        b,
        inputs,
        base,
        scen,
        "Customers stop paying the subsidised freight (GMV falls by that amount) and the "
        "marketplace pays it to carriers. Orders that already qualified get the subsidy too.",
        {"subsidy_per_month": subsidy},
    )


def more_repeat_purchases(
    b: ObservedBaseline,
    repeat_uplift_pp: float = 0.01,
    take_rate: float = 0.10,
    payment_cost_rate: float = 0.02,
) -> dict:
    extra = b.new_customers_per_month * repeat_uplift_pp * b.orders_per_repeater
    base = _money(
        b.orders_per_month,
        b.item_revenue_per_order,
        b.freight_per_order,
        take_rate,
        payment_cost_rate,
    )
    scen = _money(
        b.orders_per_month + extra,
        b.item_revenue_per_order,
        b.freight_per_order,
        take_rate,
        payment_cost_rate,
    )
    inputs = _common(b, take_rate, payment_cost_rate) + [
        Input(
            "new_customers_per_month",
            b.new_customers_per_month,
            "New customers per month",
            "observed",
        ),
        Input(
            "orders_per_repeater",
            b.orders_per_repeater,
            "Repeat orders per returning customer",
            "observed",
        ),
        Input(
            "repeat_uplift_pp",
            repeat_uplift_pp,
            "Extra share of new customers who return",
            "assumption",
        ),
    ]
    return _result(
        "More repeat purchases",
        b,
        inputs,
        base,
        scen,
        "Steady state: each month's new customers return at the higher rate. The "
        "cost of the retention programme is not modelled.",
    )


def fewer_cancellations(
    b: ObservedBaseline,
    cancellation_reduction: float = 0.3,
    take_rate: float = 0.10,
    payment_cost_rate: float = 0.02,
) -> dict:
    recovered = b.orders_placed_per_month * b.cancellation_rate * cancellation_reduction
    base = _money(
        b.orders_per_month,
        b.item_revenue_per_order,
        b.freight_per_order,
        take_rate,
        payment_cost_rate,
    )
    scen = _money(
        b.orders_per_month + recovered,
        b.item_revenue_per_order,
        b.freight_per_order,
        take_rate,
        payment_cost_rate,
    )
    inputs = _common(b, take_rate, payment_cost_rate) + [
        Input(
            "cancellation_rate",
            b.cancellation_rate,
            "Cancelled or unavailable / placed",
            "observed",
        ),
        Input(
            "cancellation_reduction",
            cancellation_reduction,
            "Share of cancellations avoided",
            "assumption",
        ),
    ]
    return _result(
        "Fewer cancellations",
        b,
        inputs,
        base,
        scen,
        "Assumes recovered orders look like average orders (most cancelled orders "
        "are 'unavailable', i.e. stock-outs).",
    )


def better_sellers(
    b: ObservedBaseline,
    percentile: float = 90.0,
    take_rate: float = 0.10,
    payment_cost_rate: float = 0.02,
) -> dict:
    """Sellers whose late rate is above the ``percentile`` get the median seller's rate."""
    rates, n = b.seller_late_rates, b.seller_orders
    ok = np.isfinite(rates) & (n >= 10)
    if ok.sum() < 5:
        new_late = b.late_rate
    else:
        cap = np.percentile(rates[ok], percentile)
        med = float(np.median(rates[ok]))
        improved = np.where(ok & (rates > cap), med, rates)
        new_late = float(np.nansum(improved * n) / np.nansum(n))
    reduction = 1 - new_late / b.late_rate if b.late_rate > 0 else 0.0
    res = faster_delivery(b, reduction, take_rate, payment_cost_rate)
    res["scenario"] = "Better sellers"
    res["inputs"].append(
        Input(
            "percentile",
            percentile,
            "Sellers above this late-rate percentile are brought to the median",
            "assumption",
        ).__dict__
    )
    res["notes"] = (
        "Only the worst sellers (by late rate, 10+ delivered orders) improve; the rest "
        "is the faster-delivery logic, with the same causal caveat."
    )
    return res


def conversion_and_aov(
    b: ObservedBaseline,
    order_change: float = 0.05,
    aov_change: float = 0.0,
    take_rate: float = 0.10,
    payment_cost_rate: float = 0.02,
) -> dict:
    base = _money(
        b.orders_per_month,
        b.item_revenue_per_order,
        b.freight_per_order,
        take_rate,
        payment_cost_rate,
    )
    scen = _money(
        b.orders_per_month * (1 + order_change),
        b.item_revenue_per_order * (1 + aov_change),
        b.freight_per_order,
        take_rate,
        payment_cost_rate,
    )
    inputs = _common(b, take_rate, payment_cost_rate) + [
        Input("order_change", order_change, "Relative change in orders (conversion)", "assumption"),
        Input("aov_change", aov_change, "Relative change in item value per order", "assumption"),
    ]
    return _result(
        "Conversion and AOV",
        b,
        inputs,
        base,
        scen,
        "Visits are not in the dataset, so conversion is expressed as a change in "
        "orders. Freight per order is held constant.",
    )


SCENARIOS = {
    "Faster delivery": faster_delivery,
    "Shipping subsidy": shipping_subsidy,
    "More repeat purchases": more_repeat_purchases,
    "Fewer cancellations": fewer_cancellations,
    "Better sellers": better_sellers,
    "Conversion and AOV": conversion_and_aov,
}

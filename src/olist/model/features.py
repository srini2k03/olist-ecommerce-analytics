"""Point-in-time customer features and labels for the repeat-purchase model.

Everything a feature uses must have been *known* at the cutoff, not merely be
about an order placed before it:

* only orders purchased before the cutoff count;
* a delivery outcome counts only if the parcel arrived before the cutoff (an
  order already past its estimated date but still undelivered *is* known to be
  late at the cutoff, so that counts too);
* a review counts only if it was written before the cutoff.

The label is whether the customer places any order in
``[cutoff, cutoff + horizon_days)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "recency_days",
    "tenure_days",
    "n_orders",
    "n_items",
    "total_spend",
    "avg_order_value",
    "max_order_value",
    "avg_freight_ratio",
    "avg_installments",
    "used_voucher",
    "n_categories",
    "last_review_score",
    "min_review_score",
    "n_reviews_known",
    "any_comment",
    "last_delivery_days",
    "last_delay_days",
    "any_late_known",
    "pending_delivery",
]
CATEGORICAL_FEATURES = ["customer_state", "last_category", "last_payment_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_LABELS = {
    "recency_days": "days since last order",
    "tenure_days": "days since first order",
    "n_orders": "orders placed",
    "n_items": "items bought",
    "total_spend": "total spend (R$)",
    "avg_order_value": "average order value (R$)",
    "max_order_value": "largest order (R$)",
    "avg_freight_ratio": "freight as share of item price",
    "avg_installments": "average instalments",
    "used_voucher": "has used a voucher",
    "n_categories": "categories bought",
    "last_review_score": "last review score",
    "min_review_score": "lowest review score",
    "n_reviews_known": "reviews written",
    "any_comment": "has written a review comment",
    "last_delivery_days": "last delivery time (days)",
    "last_delay_days": "last delivery vs promise (days, + = late)",
    "any_late_known": "has had a late delivery",
    "pending_delivery": "has an undelivered order",
    "customer_state": "state",
    "last_category": "last category bought",
    "last_payment_type": "last payment method",
}

REQUIRED_COLUMNS = [
    "order_id",
    "customer_unique_id",
    "customer_state",
    "purchased_at",
    "delivered_at",
    "estimated_at",
    "n_items",
    "items_value",
    "freight_value",
    "order_value",
    "installments",
    "main_payment_type",
    "used_voucher",
    "category",
    "review_score",
    "review_created_at",
    "has_comment",
]


def build_snapshot(
    orders: pd.DataFrame, cutoff: str | pd.Timestamp, horizon_days: int = 180
) -> pd.DataFrame:
    """Return one row per customer with an order before ``cutoff``.

    Columns: ``customer_unique_id``, ``cutoff``, the features in ``FEATURES`` and
    ``label`` (1 if the customer orders again within ``horizon_days``).
    """
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon_days)
    o = orders[REQUIRED_COLUMNS]
    hist = o[o["purchased_at"] < cutoff].copy()
    if hist.empty:
        raise ValueError(f"no orders before {cutoff.date()}")

    # --- point-in-time masking of facts that only become known later ---------
    delivered_known = hist["delivered_at"].notna() & (hist["delivered_at"] < cutoff)
    hist["delivery_days"] = np.where(
        delivered_known, (hist["delivered_at"] - hist["purchased_at"]).dt.days, np.nan
    )
    hist["delay_days"] = np.where(
        delivered_known,
        (hist["delivered_at"].dt.normalize() - hist["estimated_at"].dt.normalize()).dt.days,
        np.nan,
    )
    overdue_undelivered = ~delivered_known & (hist["estimated_at"] < cutoff)
    hist["late_known"] = ((hist["delay_days"] > 0) | overdue_undelivered).astype(float)
    hist["pending"] = (~delivered_known).astype(float)
    review_known = hist["review_created_at"].notna() & (hist["review_created_at"] < cutoff)
    hist["review_known"] = hist["review_score"].where(review_known)
    hist["comment_known"] = hist["has_comment"].where(review_known, 0).fillna(0)
    hist["freight_ratio"] = hist["freight_value"] / hist["items_value"].where(
        hist["items_value"] > 0
    )

    hist = hist.sort_values(["customer_unique_id", "purchased_at", "order_id"])
    g = hist.groupby("customer_unique_id", sort=False)
    last = g.tail(1).set_index("customer_unique_id")
    first_ts = g["purchased_at"].min()

    feats = pd.DataFrame(index=last.index)
    feats["recency_days"] = (cutoff - last["purchased_at"]).dt.days
    feats["tenure_days"] = (cutoff - first_ts).dt.days
    feats["n_orders"] = g.size()
    feats["n_items"] = g["n_items"].sum()
    feats["total_spend"] = g["order_value"].sum()
    feats["avg_order_value"] = feats["total_spend"] / feats["n_orders"]
    feats["max_order_value"] = g["order_value"].max()
    feats["avg_freight_ratio"] = g["freight_ratio"].mean()
    feats["avg_installments"] = g["installments"].mean()
    feats["used_voucher"] = g["used_voucher"].max().astype(float)
    feats["n_categories"] = g["category"].nunique()
    feats["last_review_score"] = g["review_known"].last()
    feats["min_review_score"] = g["review_known"].min()
    feats["n_reviews_known"] = g["review_known"].count()
    feats["any_comment"] = g["comment_known"].max().astype(float)
    feats["last_delivery_days"] = last["delivery_days"]
    feats["last_delay_days"] = last["delay_days"]
    feats["any_late_known"] = g["late_known"].max()
    feats["pending_delivery"] = g["pending"].max()
    feats["customer_state"] = last["customer_state"]
    feats["last_category"] = last["category"]
    feats["last_payment_type"] = last["main_payment_type"]

    # Plain float64 (not nullable Float64) so downstream tools treat them uniformly.
    feats[NUMERIC_FEATURES] = feats[NUMERIC_FEATURES].astype("float64")

    future = o[(o["purchased_at"] >= cutoff) & (o["purchased_at"] < end)]
    repeaters = set(future["customer_unique_id"])
    feats["label"] = feats.index.isin(repeaters).astype(int)
    feats["cutoff"] = cutoff
    return feats.reset_index()


def build_training_set(
    orders: pd.DataFrame, cutoffs: list[str], horizon_days: int = 180
) -> pd.DataFrame:
    """Stack several snapshots (a customer can appear once per cutoff)."""
    return pd.concat([build_snapshot(orders, c, horizon_days) for c in cutoffs], ignore_index=True)


def check_no_overlap(train_cutoffs: list[str], test_cutoff: str, horizon_days: int) -> None:
    """Raise if any training label window reaches into the test period."""
    test_start = pd.Timestamp(test_cutoff)
    for c in train_cutoffs:
        label_end = pd.Timestamp(c) + pd.Timedelta(days=horizon_days)
        if label_end > test_start:
            raise ValueError(
                f"training cutoff {c} labels run to {label_end.date()}, past the test "
                f"cutoff {test_start.date()}"
            )

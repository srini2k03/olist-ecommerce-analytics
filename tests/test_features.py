"""The churn features must only use information available at the cutoff."""

import numpy as np
import pandas as pd
import pytest

from olist.model.features import FEATURES, build_snapshot, build_training_set, check_no_overlap

CUTOFF = "2018-01-01"


def test_snapshot_shape_and_label(synthetic_orders):
    snap = build_snapshot(synthetic_orders, CUTOFF, 180)
    assert snap.customer_unique_id.is_unique
    assert set(FEATURES) <= set(snap.columns)
    assert snap.label.isin([0, 1]).all()
    assert 0 < snap.label.mean() < 0.5
    assert (snap.recency_days >= 0).all()
    assert (snap.tenure_days >= snap.recency_days).all()


def test_label_matches_future_orders(synthetic_orders):
    snap = build_snapshot(synthetic_orders, CUTOFF, 90).set_index("customer_unique_id")
    o = synthetic_orders
    end = pd.Timestamp(CUTOFF) + pd.Timedelta(days=90)
    window = o[(o.purchased_at >= CUTOFF) & (o.purchased_at < end)]
    expected = snap.index.isin(set(window.customer_unique_id)).astype(int)
    assert (snap.label.to_numpy() == expected).all()


def test_features_ignore_everything_after_cutoff(synthetic_orders):
    """Scramble every post-cutoff fact; features must not move."""
    cutoff = pd.Timestamp(CUTOFF)
    before = build_snapshot(synthetic_orders, cutoff)

    o = synthetic_orders.copy()
    rng = np.random.default_rng(0)
    # 1. Orders after the cutoff get random values.
    after = o.purchased_at >= cutoff
    o.loc[after, "order_value"] = rng.uniform(1, 1e4, after.sum())
    o.loc[after, "category"] = "scrambled"
    # 2. Deliveries that happened after the cutoff are changed.
    late_info = o.delivered_at >= cutoff
    o.loc[late_info, "delivered_at"] = o.loc[late_info, "delivered_at"] + pd.Timedelta(days=40)
    # 3. Reviews answered after the cutoff are changed.
    late_rev = o.review_answered_at >= cutoff
    o.loc[late_rev, "review_score"] = 1.0

    after_snap = build_snapshot(o, cutoff)
    cols = [c for c in FEATURES]
    pd.testing.assert_frame_equal(before[cols], after_snap[cols])


def test_post_cutoff_delivery_is_not_used(synthetic_orders):
    o = synthetic_orders.copy()
    cutoff = pd.Timestamp(CUTOFF)
    # Pick an order purchased before the cutoff but delivered after it.
    straddle = o[(o.purchased_at < cutoff) & (o.delivered_at >= cutoff)]
    assert len(straddle) > 0
    snap = build_snapshot(o, cutoff).set_index("customer_unique_id")
    cust = straddle.customer_unique_id.iloc[0]
    assert snap.loc[cust, "pending_delivery"] == 1


def test_review_sent_before_but_answered_after_cutoff_is_not_used(synthetic_orders):
    """review_creation_date is when the survey was sent; the score is only known once
    it is answered, so a survey sent before the cutoff and answered after must not count."""
    cutoff = pd.Timestamp(CUTOFF)
    o = synthetic_orders.copy()
    has_rev = (o.purchased_at < cutoff) & o.review_score.notna()
    last = o[has_rev].sort_values("purchased_at").groupby("customer_unique_id").tail(1)
    one_order = o[o.purchased_at < cutoff].customer_unique_id.value_counts() == 1
    row = last[last.customer_unique_id.map(one_order).fillna(False)].index[0]
    o.loc[row, "review_created_at"] = cutoff - pd.Timedelta(days=1)
    o.loc[row, "review_answered_at"] = cutoff + pd.Timedelta(days=1)
    snap = build_snapshot(o, cutoff).set_index("customer_unique_id")
    cust = o.loc[row, "customer_unique_id"]
    assert np.isnan(snap.loc[cust, "last_review_score"])
    assert snap.loc[cust, "n_reviews_known"] == 0


def test_training_windows_must_not_reach_test_period():
    check_no_overlap(["2017-06-01", "2017-09-01"], "2018-03-01", 180)
    with pytest.raises(ValueError):
        check_no_overlap(["2017-12-01"], "2018-03-01", 180)


def test_training_set_stacks_snapshots(synthetic_orders):
    tr = build_training_set(synthetic_orders, ["2017-09-01", "2018-01-01"], 90)
    assert tr.cutoff.nunique() == 2

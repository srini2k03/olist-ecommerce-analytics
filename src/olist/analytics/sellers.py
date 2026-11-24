"""Seller scorecard and an uncertainty-aware ranking that does not reward size alone.

Problem: a seller with 3 orders and no late delivery has a 0% late rate; one with 2,000
orders and 4% late is the better bet. Raw rates rank small sellers at both extremes.

Method (browser-safe: numpy + pandas only):
1. Minimum volume: sellers need ``min_orders`` delivered orders to be ranked; smaller ones are
   listed as "insufficient volume" instead of being ranked.
2. Wilson 95% intervals for the raw late, bad-review and cancellation rates.
3. Empirical-Bayes shrinkage: a Beta prior fitted by the method of moments to the eligible
   sellers pulls each rate towards the marketplace average in proportion to how little data
   the seller has. Mean reviews are shrunk with a normal-normal model the same way.
4. Composite score = weighted sum of standardised, shrunk metrics (weights in
   ``RANKING_WEIGHTS``: service quality dominates, revenue has a small weight).
5. Rank uncertainty: Monte Carlo draws from each seller's Beta posteriors give a 90% interval
   for the rank and the probability of being in the top decile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from olist.stats.core import wilson_interval

MIN_ORDERS = 30
RANKING_WEIGHTS = {
    "late_rate": -0.35,  # lower is better
    "bad_review_rate": -0.35,
    "cancellation_rate": -0.10,
    "log_revenue": 0.20,  # some credit for scale, not the whole ranking
}


def scorecard_from_frames(orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """The 21_seller_scorecard columns computed from dashboard frames (filter-aware).

    ``items``: one row per item with seller_id, order_id, item_revenue, is_valid.
    ``orders``: one row per order with order_id, is_valid, is_cancelled, is_delivered,
    is_late, delivery_days, review_score.
    """
    so = items.groupby(["seller_id", "order_id"], observed=True, as_index=False).agg(
        item_revenue=("item_revenue", "sum")
    )
    cols = [
        "order_id",
        "is_valid",
        "is_cancelled",
        "is_delivered",
        "is_late",
        "delivery_days",
        "review_score",
    ]
    so = so.merge(orders[cols], on="order_id", how="left")
    valid = so["is_valid"].fillna(0).eq(1)
    delivered = so["is_delivered"].fillna(0).eq(1)
    so["rev_valid"] = so["item_revenue"].where(valid, 0.0)
    so["late"] = (so["is_late"].fillna(0).eq(1) & delivered).astype(int)
    so["reviewed"] = (so["review_score"].notna() & valid).astype(int)
    so["bad"] = (so["review_score"].le(2) & valid).astype(int)
    so["review_valid"] = so["review_score"].where(valid)
    so["days_delivered"] = so["delivery_days"].where(delivered)
    g = so.groupby("seller_id", observed=True)
    out = pd.DataFrame(
        {
            "orders_placed": g.size(),
            "orders": g["is_valid"].sum(),
            "item_revenue": g["rev_valid"].sum(),
            "delivered_orders": g["is_delivered"].sum(),
            "late_orders": g["late"].sum(),
            "avg_delivery_days": g["days_delivered"].mean(),
            "reviewed_orders": g["reviewed"].sum(),
            "avg_review": g["review_valid"].mean(),
            "bad_reviews": g["bad"].sum(),
            "cancelled_orders": g["is_cancelled"].sum(),
        }
    )
    return out.reset_index()


def beta_prior(successes: np.ndarray, trials: np.ndarray) -> tuple[float, float]:
    """Method-of-moments Beta(a, b) prior for rates, from sellers with trials > 0.

    The between-seller variance is the observed variance of the rates minus the average
    binomial sampling variance; if that is not positive the prior is made very strong
    (rates are indistinguishable from the pooled mean).
    """
    k, n = np.asarray(successes, float), np.asarray(trials, float)
    mask = n > 0
    k, n = k[mask], n[mask]
    if len(n) == 0:
        return 1.0, 1.0
    p = k / n
    mu = float(k.sum() / n.sum())
    mu = min(max(mu, 1e-4), 1 - 1e-4)
    var_obs = float(np.average((p - mu) ** 2, weights=n))
    var_sampling = float(np.average(mu * (1 - mu) / n, weights=n))
    tau2 = var_obs - var_sampling
    # tau2 <= 0: sellers are indistinguishable from the pooled mean, so a very strong prior
    strength = 1e4 if tau2 <= 1e-8 else max(mu * (1 - mu) / tau2 - 1, 1.0)
    return mu * strength, (1 - mu) * strength


def normal_shrinkage(means: np.ndarray, counts: np.ndarray, within_var: float) -> np.ndarray:
    """Shrink group means towards the weighted grand mean: w = n / (n + within_var / tau2)."""
    m, n = np.asarray(means, float), np.asarray(counts, float)
    ok = np.isfinite(m) & (n > 0)
    if ok.sum() < 2:
        return m
    grand = float(np.average(m[ok], weights=n[ok]))
    tau2 = float(np.average((m[ok] - grand) ** 2, weights=n[ok])) - float(
        np.average(within_var / n[ok], weights=n[ok])
    )
    if tau2 <= 1e-8:
        return np.where(ok, grand, np.nan)
    k = within_var / tau2
    return np.where(ok, (n * m + k * grand) / (n + k), np.nan)


def rank_sellers(
    card: pd.DataFrame,
    min_orders: int = MIN_ORDERS,
    weights: dict[str, float] | None = None,
    n_draws: int = 400,
    review_var: float = 1.9,
    seed: int = 0,
) -> pd.DataFrame:
    """Add rates, Wilson intervals, shrunk estimates, a composite score and rank intervals.

    ``card`` has the 21_seller_scorecard columns. ``review_var`` is the within-seller
    variance of review scores (1.9 on the real data, about a 1.4-star SD).
    """
    w = weights or RANKING_WEIGHTS
    c = card.copy()
    n_del = c["delivered_orders"].to_numpy(float)
    n_rev = c["reviewed_orders"].to_numpy(float)
    n_all = c["orders_placed"].to_numpy(float)
    specs = {
        "late_rate": (c["late_orders"].to_numpy(float), n_del),
        "bad_review_rate": (c["bad_reviews"].to_numpy(float), n_rev),
        "cancellation_rate": (c["cancelled_orders"].to_numpy(float), n_all),
    }
    eligible = n_del >= min_orders
    c["eligible"] = eligible
    posteriors = {}
    for name, (k, n) in specs.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            c[name] = np.where(n > 0, k / n, np.nan)
        lo, hi = wilson_interval(k, n)
        c[f"{name}_ci_low"], c[f"{name}_ci_high"] = lo, hi
        a0, b0 = beta_prior(k[eligible], n[eligible]) if eligible.any() else beta_prior(k, n)
        a_post, b_post = a0 + k, b0 + n - k
        c[f"{name}_shrunk"] = a_post / (a_post + b_post)
        posteriors[name] = (a_post, b_post)
    c["avg_review_shrunk"] = normal_shrinkage(c["avg_review"].to_numpy(float), n_rev, review_var)
    c["log_revenue"] = np.log1p(c["item_revenue"].clip(lower=0))

    elig = c[eligible]
    if elig.empty:
        for col in ("score", "rank", "rank_p05", "rank_p95", "p_top_decile"):
            c[col] = np.nan
        c["status"] = "insufficient volume"
        return c
    centre = {k: float(elig[f"{k}_shrunk" if k != "log_revenue" else k].mean()) for k in w}
    scale = {
        k: float(elig[f"{k}_shrunk" if k != "log_revenue" else k].std(ddof=0)) or 1.0 for k in w
    }

    def composite(values: dict[str, np.ndarray]) -> np.ndarray:
        return sum(w[k] * (values[k] - centre[k]) / scale[k] for k in w)

    point = {k: c[f"{k}_shrunk"].to_numpy(float) for k in specs}
    point["log_revenue"] = c["log_revenue"].to_numpy(float)
    c["score"] = np.where(eligible, composite(point), np.nan)
    c["rank"] = c["score"].rank(ascending=False, method="min")

    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(eligible)
    draws = {
        k: rng.beta(a[idx][None, :], b[idx][None, :], size=(n_draws, len(idx)))
        for k, (a, b) in posteriors.items()
    }
    draws["log_revenue"] = np.broadcast_to(point["log_revenue"][idx], (n_draws, len(idx)))
    scores = composite(draws)
    ranks = (-scores).argsort(axis=1, kind="stable").argsort(axis=1, kind="stable") + 1
    top = max(1, int(round(0.1 * len(idx))))
    c["rank_p05"] = np.nan
    c["rank_p95"] = np.nan
    c["p_top_decile"] = np.nan
    c.loc[c.index[idx], "rank_p05"] = np.percentile(ranks, 5, axis=0)
    c.loc[c.index[idx], "rank_p95"] = np.percentile(ranks, 95, axis=0)
    c.loc[c.index[idx], "p_top_decile"] = (ranks <= top).mean(axis=0)
    c["status"] = np.where(eligible, "ranked", "insufficient volume")
    return c.sort_values(["rank", "item_revenue"], ascending=[True, False]).reset_index(drop=True)

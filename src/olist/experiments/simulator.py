"""A/B test simulator for a free-shipping threshold, calibrated on real orders.

Design being simulated: orders whose item value (excluding freight) reaches ``threshold``
ship free; the marketplace pays the carrier freight of those orders (the subsidy).

What is observed (from the data) and what is assumed (labelled, configurable):

- observed: daily order volume, the baskets themselves (item value, freight, item count,
  review score), resampled with replacement for both arms;
- assumed: the true effect of the treatment on the share of multi-item orders
  (``lift_multi_item``), on order volume (``lift_orders``; visits are not in the data, so
  conversion cannot be measured), the marketplace take rate and payment cost.

Treatment mechanics: a resampled single-item order becomes a two-item order with the
probability that produces the assumed lift; the extra item is another resampled item
(price and freight). Qualifying treatment orders have their freight paid by the marketplace.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from olist.stats.core import (
    bootstrap_diff_ci,
    norm_cdf,
    power_proportions,
    sample_size_proportions,
    two_proportion_test,
    z_crit,
)


@dataclass(frozen=True)
class Baseline:
    """Observed inputs, built from real valid orders by ``Baseline.from_orders``."""

    item_value: np.ndarray  # per order, excluding freight
    freight: np.ndarray
    n_items: np.ndarray
    review: np.ndarray  # NaN where no review
    unit_price: np.ndarray  # pool of single item prices
    unit_freight: np.ndarray
    daily_orders: float
    period: str

    @classmethod
    def from_orders(
        cls,
        orders: pd.DataFrame,
        items: pd.DataFrame | None = None,
        period: str = "2018-01-01/2018-09-01",
    ) -> Baseline:
        lo, hi = period.split("/")
        o = orders[(orders["is_valid"] == 1)]
        t = pd.to_datetime(o["purchase_date"])
        o = o[(t >= lo) & (t < hi)]
        days = max((pd.Timestamp(hi) - pd.Timestamp(lo)).days, 1)
        if items is not None:
            it = items[items["order_id"].isin(o["order_id"])]
            unit_price, unit_freight = (
                it["item_revenue"].to_numpy(float),
                it["freight_value"].to_numpy(float),
            )
        else:  # approximate items by per-order averages
            unit_price = (o["item_revenue"] / o["n_items"]).to_numpy(float)
            unit_freight = (o["freight_value"] / o["n_items"]).to_numpy(float)
        return cls(
            item_value=o["item_revenue"].to_numpy(float),
            freight=o["freight_value"].to_numpy(float),
            n_items=o["n_items"].to_numpy(float),
            review=o["review_score"].to_numpy(float),
            unit_price=unit_price,
            unit_freight=unit_freight,
            daily_orders=len(o) / days,
            period=period,
        )

    @property
    def multi_item_rate(self) -> float:
        return float((self.n_items >= 2).mean())

    def qualifying_share(self, threshold: float) -> float:
        return float((self.item_value >= threshold).mean())


@dataclass(frozen=True)
class Assumptions:
    """Labelled assumptions of the simulated experiment (not estimated from data)."""

    threshold: float = 150.0
    lift_multi_item: float = 0.15  # relative lift in the multi-item share
    lift_orders: float = 0.0  # relative change in order volume
    take_rate: float = 0.10
    payment_cost_rate: float = 0.02
    review_shift: float = 0.0  # assumed change in mean review of treatment orders


@dataclass(frozen=True)
class Design:
    alpha: float = 0.05
    power: float = 0.80
    mde_rel: float = 0.15  # minimum detectable relative lift for the primary metric
    share_treatment: float = 0.5
    duration_days: int | None = None  # None = as long as the power calculation requires
    seed: int = 42


@dataclass
class ArmData:
    item_value: np.ndarray
    freight: np.ndarray
    n_items: np.ndarray
    review: np.ndarray
    subsidy: np.ndarray

    @property
    def gmv_customer(self) -> np.ndarray:
        """What the customer pays per the marketplace's GMV definition (items + freight
        charged); under free shipping the freight charged is 0."""
        return self.item_value + self.freight - self.subsidy

    def contribution(self, a: Assumptions) -> np.ndarray:
        """Estimated marketplace contribution per order (assumption-based)."""
        paid = self.item_value + self.freight - self.subsidy
        return a.take_rate * self.item_value - a.payment_cost_rate * paid - self.subsidy


def plan(base: Baseline, design: Design) -> dict:
    p0 = base.multi_item_rate
    n = sample_size_proportions(p0, design.mde_rel, design.alpha, design.power)
    per_day_per_arm = base.daily_orders * min(design.share_treatment, 1 - design.share_treatment)
    days = math.ceil(n / per_day_per_arm)
    duration = design.duration_days or days
    n_run = int(per_day_per_arm * duration)
    return {
        "baseline_multi_item_rate": p0,
        "daily_orders": base.daily_orders,
        "n_per_arm_required": n,
        "days_required": days,
        "duration_days": duration,
        "n_per_arm_planned": n_run,
        "power_at_planned_n": power_proportions(p0, design.mde_rel, max(n_run, 1), design.alpha),
    }


def _arm(base: Baseline, n: int, rng: np.random.Generator, a: Assumptions | None) -> ArmData:
    idx = rng.integers(0, len(base.item_value), n)
    value = base.item_value[idx].copy()
    freight = base.freight[idx].copy()
    items = base.n_items[idx].copy()
    review = base.review[idx].copy()
    subsidy = np.zeros(n)
    if a is not None:
        p0 = base.multi_item_rate
        p1 = min(p0 * (1 + a.lift_multi_item), 0.999)
        q = max((p1 - p0) / (1 - p0), 0.0)  # share of single-item orders that add an item
        convert = (items == 1) & (rng.random(n) < q)
        k = int(convert.sum())
        j = rng.integers(0, len(base.unit_price), k)
        value[convert] += base.unit_price[j]
        freight[convert] += base.unit_freight[j]
        items[convert] += 1
        review = review + a.review_shift
        subsidy = np.where(value >= a.threshold, freight, 0.0)
    return ArmData(value, freight, items, review, subsidy)


def _mean_test(x_a: np.ndarray, x_b: np.ndarray, alpha: float, n_boot: int, seed: int) -> dict:
    x_a, x_b = x_a[np.isfinite(x_a)], x_b[np.isfinite(x_b)]
    diff = float(x_b.mean() - x_a.mean())
    se = math.sqrt(x_a.var(ddof=1) / len(x_a) + x_b.var(ddof=1) / len(x_b))
    z = diff / se if se > 0 else 0.0
    pooled_sd = math.sqrt((x_a.var(ddof=1) + x_b.var(ddof=1)) / 2)
    return {
        "mean_a": float(x_a.mean()),
        "mean_b": float(x_b.mean()),
        "diff": diff,
        "diff_ci": (diff - z_crit(alpha) * se, diff + z_crit(alpha) * se),
        "diff_ci_bootstrap": bootstrap_diff_ci(x_a, x_b, n_boot=n_boot, seed=seed),
        "p_value": 2 * (1 - norm_cdf(abs(z))),
        "cohens_d": diff / pooled_sd if pooled_sd > 0 else 0.0,
    }


def simulate(
    base: Baseline,
    assumptions: Assumptions = Assumptions(),  # noqa: B008 (frozen dataclass)
    design: Design = Design(),  # noqa: B008
    n_boot: int = 400,
) -> dict:
    """Run one simulated experiment and analyse it like a real one."""
    rng = np.random.default_rng(design.seed)
    p = plan(base, design)
    n_total = int(round(base.daily_orders * p["duration_days"]))
    in_b = rng.random(n_total) < design.share_treatment
    n_a = int((~in_b).sum())
    n_b = int(round(in_b.sum() * (1 + assumptions.lift_orders)))  # volume effect (assumed)
    A = _arm(base, n_a, rng, None)
    B = _arm(base, n_b, rng, assumptions)

    # SRM: compare the assignment counts (before the assumed volume effect).
    n_b_assigned = int(in_b.sum())
    exp_b = n_total * design.share_treatment
    z_srm = (n_b_assigned - exp_b) / math.sqrt(
        n_total * design.share_treatment * (1 - design.share_treatment)
    )
    srm_p = 2 * (1 - norm_cdf(abs(z_srm)))

    x_a, x_b = int((A.n_items >= 2).sum()), int((B.n_items >= 2).sum())
    primary = two_proportion_test(x_a, n_a, x_b, n_b, design.alpha).to_dict()
    draws_a = rng.beta(1 + x_a, 1 + n_a - x_a, 20000)
    draws_b = rng.beta(1 + x_b, 1 + n_b - x_b, 20000)
    primary["prob_b_better"] = float((draws_b > draws_a).mean())
    primary["expected_loss_choose_b"] = float(np.maximum(draws_a - draws_b, 0).mean())

    seed = design.seed
    secondary = {
        "aov_customer": _mean_test(A.gmv_customer, B.gmv_customer, design.alpha, n_boot, seed),
        "item_value_per_order": _mean_test(A.item_value, B.item_value, design.alpha, n_boot, seed),
        "items_per_order": _mean_test(A.n_items, B.n_items, design.alpha, n_boot, seed),
    }
    guardrails = {"review_score": _mean_test(A.review, B.review, design.alpha, n_boot, seed)}
    c_a, c_b = A.contribution(assumptions), B.contribution(assumptions)
    contrib = _mean_test(c_a, c_b, design.alpha, n_boot, seed)
    days = p["duration_days"]
    # Per-day economics at 100% rollout: treatment per-order values x treatment volume.
    orders_day_b = base.daily_orders * (1 + assumptions.lift_orders)
    economics = {
        "qualifying_share_control": float((A.item_value >= assumptions.threshold).mean()),
        "qualifying_share_treatment": float((B.subsidy > 0).mean()),
        "subsidy_per_treatment_order": float(B.subsidy.mean()),
        "subsidy_per_month_at_rollout": float(B.subsidy.mean() * orders_day_b * 30),
        "incremental_item_value_per_order": secondary["item_value_per_order"]["diff"],
        "incremental_gmv_per_month_at_rollout": float(
            (B.gmv_customer.mean() * orders_day_b - A.gmv_customer.mean() * base.daily_orders) * 30
        ),
        "contribution_per_order_control": float(c_a.mean()),
        "contribution_per_order_treatment": float(c_b.mean()),
        "incremental_contribution_per_month_at_rollout": float(
            (c_b.mean() * orders_day_b - c_a.mean() * base.daily_orders) * 30
        ),
        "incremental_contribution_per_order": contrib,
    }
    boots = _contribution_bootstrap(c_a, c_b, base.daily_orders, orders_day_b, n_boot, seed)
    economics["prob_contribution_positive"] = float((boots > 0).mean())
    economics["incremental_contribution_per_month_ci"] = (
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )
    return {
        "simulated": True,
        "assumptions": asdict(assumptions),
        "design": asdict(design),
        "plan": p,
        "n_a": n_a,
        "n_b": n_b,
        "days": days,
        "srm_p_value": srm_p,
        "primary_multi_item_rate": primary,
        "secondary": secondary,
        "guardrails": guardrails,
        "economics": economics,
    }


def _contribution_bootstrap(c_a, c_b, day_a, day_b, n_boot, seed) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    out = np.empty(n_boot)
    for i in range(n_boot):
        ma = c_a[rng.integers(0, len(c_a), len(c_a))].mean()
        mb = c_b[rng.integers(0, len(c_b), len(c_b))].mean()
        out[i] = (mb * day_b - ma * day_a) * 30
    return out


def expected_monthly_contribution(
    base: Baseline, a: Assumptions, n: int = 20000, seed: int = 7
) -> float:
    """Expected incremental contribution per month at rollout, by Monte Carlo with common
    random numbers (same seed for every call), so sensitivity grids are smooth."""
    rng_a, rng_b = np.random.default_rng(seed), np.random.default_rng(seed)
    A = _arm(base, n, rng_a, None)
    B = _arm(base, n, rng_b, a)
    day_b = base.daily_orders * (1 + a.lift_orders)
    return float(
        (B.contribution(a).mean() * day_b - A.contribution(a).mean() * base.daily_orders) * 30
    )


def sensitivity_grid(
    base: Baseline, a: Assumptions, x: str, xs: list[float], y: str, ys: list[float]
) -> pd.DataFrame:
    """Expected incremental monthly contribution over a grid of two assumptions."""
    rows = []
    for vx in xs:
        for vy in ys:
            val = expected_monthly_contribution(base, replace(a, **{x: vx, y: vy}))
            rows.append({x: vx, y: vy, "incremental_contribution_month": val})
    return pd.DataFrame(rows)


def tornado(base: Baseline, a: Assumptions, ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """One-at-a-time sensitivity: contribution at the low and high value of each assumption."""
    centre = expected_monthly_contribution(base, a)
    rows = []
    for name, (lo, hi) in ranges.items():
        v_lo = expected_monthly_contribution(base, replace(a, **{name: lo}))
        v_hi = expected_monthly_contribution(base, replace(a, **{name: hi}))
        rows.append(
            {
                "assumption": name,
                "low": lo,
                "high": hi,
                "at_low": v_lo,
                "at_high": v_hi,
                "swing": abs(v_hi - v_lo),
            }
        )
    return pd.DataFrame(rows).sort_values("swing", ascending=False).assign(centre=centre)


DEFAULT_TORNADO = {
    "lift_multi_item": (0.0, 0.30),
    "lift_orders": (-0.02, 0.05),
    "take_rate": (0.05, 0.20),
    "threshold": (100.0, 250.0),
    "payment_cost_rate": (0.01, 0.03),
}
_ = field  # keep dataclasses.field importable for callers extending Assumptions

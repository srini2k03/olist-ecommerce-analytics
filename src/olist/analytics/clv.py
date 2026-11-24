"""Customer lifetime value: observed history kept apart from a predicted future.

Observed value is what customers have already spent (GMV of valid orders). Predicted value
uses the standard probabilistic models for non-contractual purchasing:

- BG/NBD (Fader, Hardie & Lee, 2005) for the number of future repeat orders, from each
  customer's frequency (repeat orders), recency (weeks from first to last order) and age T
  (weeks from first order to the cut-off);
- Gamma-Gamma (Fader, Hardie & Lee, 2005b) for the expected GMV per repeat order, fitted on
  customers with at least one repeat order.

On Olist about 3% of customers ever order twice, so predicted future value is small next to
the first order and individual predictions are close to zero for almost everyone. The
model is validated on a later, held-out period against a naive baseline, and is useful for
aggregate planning and for ranking, not for per-customer promises.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1

WEEK = pd.Timedelta(days=7)


def rfm_summary(orders: pd.DataFrame, cutoff, start=None) -> pd.DataFrame:
    """Per customer from valid orders before ``cutoff`` (and after ``start``, if given):
    frequency (repeat orders), recency and T in weeks, mean GMV of repeat orders, and the
    observed GMV to date. Orders on the same day count as one purchase occasion, as in the
    BG/NBD literature (30% of Olist second orders are same-day split baskets)."""
    cutoff = pd.Timestamp(cutoff)
    o = orders[(orders["is_valid"] == 1) & (orders["purchased_at"] < cutoff)].copy()
    if start is not None:
        o = o[o["purchased_at"] >= pd.Timestamp(start)]
    o["day"] = pd.to_datetime(o["purchased_at"]).dt.normalize()
    daily = o.groupby(["customer_unique_id", "day"], as_index=False)["gmv"].sum()
    g = daily.groupby("customer_unique_id")
    first, last = g["day"].min(), g["day"].max()
    n = g.size()
    total = g["gmv"].sum()
    first_value = daily.sort_values("day").groupby("customer_unique_id")["gmv"].first()
    out = pd.DataFrame(
        {
            "frequency": n - 1,
            "recency": (last - first) / WEEK,
            "T": (cutoff - first) / WEEK,
            "observed_gmv": total,
            "first_order_gmv": first_value,
        }
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out["monetary_repeat"] = np.where(
            out.frequency > 0, (total - first_value) / out.frequency.clip(lower=1), np.nan
        )
    return out


# ---------------------------------------------------------------------------- BG/NBD
@dataclass(frozen=True)
class BGNBD:
    r: float
    alpha: float
    a: float
    b: float
    log_likelihood: float
    n_customers: int

    def to_dict(self) -> dict:
        return asdict(self)

    def _ab(self, x, t_x, T):
        x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
        r, al, a, b = self.r, self.alpha, self.a, self.b
        # log of the "dropped out after the last purchase" term relative to "alive"
        with np.errstate(divide="ignore"):
            log_ratio = (
                np.log(a)
                - np.log(np.maximum(b + x - 1, 1e-12))
                + (r + x) * (np.log(al + T) - np.log(al + t_x))
            )
        return x, t_x, T, np.where(x > 0, np.exp(np.minimum(log_ratio, 700)), 0.0)

    def p_alive(self, x, t_x, T) -> np.ndarray:
        _, _, _, d = self._ab(x, t_x, T)
        return 1.0 / (1.0 + d)

    def expected_purchases(self, t: float, x, t_x, T) -> np.ndarray:
        """E[repeat purchases in (T, T + t] | x, t_x, T]."""
        x, t_x, T, d = self._ab(x, t_x, T)
        r, al, a, b = self.r, self.alpha, self.a, self.b
        if a <= 1:
            raise ValueError("BG/NBD expectation needs a > 1")
        z = t / (al + T + t)
        hyp = hyp2f1(r + x, b + x, a + b + x - 1, z)
        num = (a + b + x - 1) / (a - 1) * (1 - ((al + T) / (al + T + t)) ** (r + x) * hyp)
        return num / (1 + d)


def bgnbd_neg_loglik(log_params: np.ndarray, x, t_x, T) -> float:
    r, al, a, b = np.exp(log_params)
    common = gammaln(r + x) - gammaln(r) + r * np.log(al)
    a1 = betaln(a, b + x) - betaln(a, b) - (r + x) * np.log(al + T)
    a2 = np.where(
        x > 0,
        betaln(a + 1, b + x - 1) - betaln(a, b) - (r + x) * np.log(al + t_x),
        -np.inf,
    )
    ll = common + np.logaddexp(a1, a2)
    return -float(ll.sum())


def fit_bgnbd(x, t_x, T, init=(0.5, 5.0, 1.0, 5.0)) -> BGNBD:
    x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
    res = minimize(
        bgnbd_neg_loglik,
        np.log(init),
        args=(x, t_x, T),
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6},
    )
    r, al, a, b = np.exp(res.x)
    return BGNBD(float(r), float(al), float(a), float(b), float(-res.fun), len(x))


# ---------------------------------------------------------------------------- Gamma-Gamma
@dataclass(frozen=True)
class GammaGamma:
    p: float
    q: float
    v: float
    n_customers: int
    corr_frequency_monetary: float

    def to_dict(self) -> dict:
        return asdict(self)

    def expected_spend(self, x, m) -> np.ndarray:
        """E[GMV per repeat order]; customers without repeats get the population mean."""
        x = np.asarray(x, float)
        m = np.asarray(m, float)
        pop = self.p * self.v / (self.q - 1)
        with np.errstate(invalid="ignore"):
            cond = self.p * (self.v + x * m) / (self.p * x + self.q - 1)
        return np.where((x > 0) & np.isfinite(m), cond, pop)


def gg_neg_loglik(log_params, x, m) -> float:
    p, q, v = np.exp(log_params)
    ll = (
        gammaln(p * x + q)
        - gammaln(p * x)
        - gammaln(q)
        + q * np.log(v)
        + (p * x - 1) * np.log(m)
        + p * x * np.log(x)
        - (p * x + q) * np.log(x * m + v)
    )
    return -float(ll.sum())


def fit_gamma_gamma(x, m, init=(2.0, 3.0, 100.0)) -> GammaGamma:
    x, m = np.asarray(x, float), np.asarray(m, float)
    keep = (x > 0) & (m > 0) & np.isfinite(m)
    x, m = x[keep], m[keep]
    res = minimize(
        gg_neg_loglik,
        np.log(init),
        args=(x, m),
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6},
    )
    p, q, v = np.exp(res.x)
    corr = float(np.corrcoef(x, m)[0, 1]) if len(x) > 2 else math.nan
    return GammaGamma(float(p), float(q), float(v), int(keep.sum()), corr)


# ---------------------------------------------------------------------------- validation
def holdout_validation(orders: pd.DataFrame, calibration_end, holdout_end, start=None) -> dict:
    """Fit on purchases before ``calibration_end``, predict repeat purchases and GMV in
    [calibration_end, holdout_end) for the customers acquired before calibration_end, and
    compare with what happened and with a naive baseline (every customer gets the
    calibration-period repeat rate per week of exposure)."""
    cal_end, hold_end = pd.Timestamp(calibration_end), pd.Timestamp(holdout_end)
    cal = rfm_summary(orders, cal_end, start)
    weeks = (hold_end - cal_end) / WEEK
    bg = fit_bgnbd(cal.frequency, cal.recency, cal["T"])
    gg = fit_gamma_gamma(cal.frequency, cal.monetary_repeat)

    o = orders[
        (orders.is_valid == 1) & (orders.purchased_at >= cal_end) & (orders.purchased_at < hold_end)
    ].copy()
    o["day"] = pd.to_datetime(o["purchased_at"]).dt.normalize()
    daily = o.groupby(["customer_unique_id", "day"], as_index=False)["gmv"].sum()
    actual_n = daily.groupby("customer_unique_id").size().reindex(cal.index, fill_value=0)
    actual_gmv = daily.groupby("customer_unique_id")["gmv"].sum().reindex(cal.index, fill_value=0)

    pred_n = bg.expected_purchases(weeks, cal.frequency, cal.recency, cal["T"])
    spend = gg.expected_spend(cal.frequency, cal.monetary_repeat)
    pred_gmv = pred_n * spend
    rate = cal.frequency.sum() / (cal["T"].sum())  # repeat occasions per customer-week
    base_n = np.full(len(cal), rate * weeks)
    base_gmv = base_n * float(np.nanmean(cal.monetary_repeat))

    def errors(pred, actual):
        e = np.asarray(pred) - np.asarray(actual, float)
        return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e**2).mean()))}

    def ranking(score, actual):
        from sklearn.metrics import average_precision_score, roc_auc_score

        y = (np.asarray(actual) > 0).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            return {"roc_auc": math.nan, "pr_auc": math.nan, "base_rate": float(y.mean())}
        return {
            "roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)),
            "base_rate": float(y.mean()),
        }

    return {
        "calibration_end": str(cal_end.date()),
        "holdout_end": str(hold_end.date()),
        "holdout_weeks": float(weeks),
        "customers": int(len(cal)),
        "calibration_repeaters": int((cal.frequency > 0).sum()),
        "bgnbd": bg.to_dict(),
        "gamma_gamma": gg.to_dict(),
        "actual": {
            "repeat_purchases": int(actual_n.sum()),
            "customers_with_purchase": int((actual_n > 0).sum()),
            "gmv": float(actual_gmv.sum()),
        },
        "model": {
            "repeat_purchases": float(pred_n.sum()),
            "gmv": float(np.nansum(pred_gmv)),
            "count_errors": errors(pred_n, actual_n),
            "gmv_errors": errors(np.nan_to_num(pred_gmv), actual_gmv),
            "ranking": ranking(pred_n, actual_n),
        },
        "baseline_constant_rate": {
            "repeat_purchases": float(base_n.sum()),
            "gmv": float(base_gmv.sum()),
            "count_errors": errors(base_n, actual_n),
            "gmv_errors": errors(base_gmv, actual_gmv),
            "ranking": {"roc_auc": 0.5, "pr_auc": float((actual_n > 0).mean())},
        },
    }


def clv_table(orders: pd.DataFrame, as_of, horizon_weeks: float = 52) -> tuple[pd.DataFrame, dict]:
    """Observed GMV to date and predicted GMV over the next ``horizon_weeks`` per customer,
    fitted on all valid orders before ``as_of``."""
    s = rfm_summary(orders, as_of)
    bg = fit_bgnbd(s.frequency, s.recency, s["T"])
    gg = fit_gamma_gamma(s.frequency, s.monetary_repeat)
    s["p_alive"] = bg.p_alive(s.frequency, s.recency, s["T"])
    s["expected_repeat_orders"] = bg.expected_purchases(
        horizon_weeks, s.frequency, s.recency, s["T"]
    )
    s["expected_gmv_per_order"] = gg.expected_spend(s.frequency, s.monetary_repeat)
    s["predicted_gmv"] = s["expected_repeat_orders"] * s["expected_gmv_per_order"]
    params = {"bgnbd": bg.to_dict(), "gamma_gamma": gg.to_dict(), "horizon_weeks": horizon_weeks}
    return s.reset_index(), params

"""Data-quality and referential-integrity checks for every warehouse layer.

A check is a portable SQL query that returns ``failing`` (rows violating the rule) and
``total`` (rows the rule applies to). Its outcome:

- ``pass``: no failing rows;
- ``warn``: failing rows within the check's tolerance, or any failure of a ``warn`` check
  (a known, documented issue of the source data that staging handles);
- ``fail``: an ``error`` check above its tolerance. The pipeline stops on any ``fail``.

Tolerances are shares of ``total`` and are set just above the known issue in the real
data (documented in ``docs/data_quality.md``), so a new problem of the same kind fails.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from olist.db import Engine

BRAZIL_STATES = (
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA",
    "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
)  # fmt: skip
ORDER_STATUSES = (
    "approved", "canceled", "created", "delivered", "invoiced", "processing", "shipped",
    "unavailable",
)  # fmt: skip
PAYMENT_TYPES = ("boleto", "credit_card", "debit_card", "not_defined", "voucher")


@dataclass(frozen=True)
class Check:
    name: str
    layer: str
    table: str
    kind: str
    description: str
    sql: str
    severity: str = "error"
    tolerance: float = 0.0


@dataclass
class CheckResult:
    name: str
    layer: str
    table: str
    kind: str
    description: str
    severity: str
    tolerance: float
    failing: int
    total: int
    status: str

    @property
    def failing_share(self) -> float:
        return self.failing / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["failing_share"] = round(self.failing_share, 6)
        return d


def _quote(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


# ---------------------------------------------------------------------------- check builders
def not_null(layer: str, table: str, col: str, **kw) -> Check:
    return Check(
        f"{table}.{col} not null",
        layer,
        table,
        "not_null",
        kw.pop("description", f"{col} is always present"),
        f"SELECT SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS failing, COUNT(*) AS total "
        f"FROM {table}",
        **kw,
    )


def row_count(layer: str, table: str) -> Check:
    return Check(
        f"{table} not empty",
        layer,
        table,
        "row_count",
        "the table has rows",
        f"SELECT CASE WHEN COUNT(*) = 0 THEN 1 ELSE 0 END AS failing, 1 AS total FROM {table}",
    )


def unique(layer: str, table: str, cols: tuple[str, ...], **kw) -> Check:
    key = ", ".join(cols)
    return Check(
        f"{table}({key}) unique",
        layer,
        table,
        "unique",
        kw.pop("description", f"({key}) identifies one row"),
        f"SELECT COALESCE(SUM(n - 1), 0) AS failing, (SELECT COUNT(*) FROM {table}) AS total "
        f"FROM (SELECT {key}, COUNT(*) AS n FROM {table} GROUP BY {key} HAVING COUNT(*) > 1) d",
        **kw,
    )


def accepted_values(layer: str, table: str, col: str, values: tuple[str, ...], **kw) -> Check:
    return Check(
        f"{table}.{col} accepted values",
        layer,
        table,
        "accepted_values",
        kw.pop("description", f"{col} is one of {len(values)} known values"),
        f"SELECT SUM(CASE WHEN {col} IS NOT NULL AND {col} NOT IN ({_quote(values)}) "
        f"THEN 1 ELSE 0 END) AS failing, COUNT(*) AS total FROM {table}",
        **kw,
    )


def condition(layer: str, table: str, name: str, ok: str, description: str, **kw) -> Check:
    """Rows where the boolean SQL ``ok`` is false fail (NULL counts as not applicable)."""
    return Check(
        f"{table}: {name}",
        layer,
        table,
        kw.pop("kind", "range"),
        description,
        f"SELECT SUM(CASE WHEN NOT ({ok}) THEN 1 ELSE 0 END) AS failing, COUNT(*) AS total "
        f"FROM {table}",
        **kw,
    )


def foreign_key(layer: str, child: str, col: str, parent: str, pcol: str, **kw) -> Check:
    return Check(
        f"{child}.{col} -> {parent}.{pcol}",
        layer,
        child,
        "foreign_key",
        kw.pop("description", f"every {child}.{col} exists in {parent}"),
        f"SELECT SUM(CASE WHEN c.{col} IS NOT NULL AND NOT EXISTS ("
        f"SELECT 1 FROM {parent} AS p WHERE p.{pcol} = c.{col}) THEN 1 ELSE 0 END) AS failing, "
        f"COUNT(*) AS total FROM {child} AS c",
        **kw,
    )


def has_child(layer: str, parent: str, pcol: str, child: str, col: str, **kw) -> Check:
    """Completeness: every parent row has at least one child row."""
    return Check(
        f"{parent} has {child}",
        layer,
        parent,
        "completeness",
        kw.pop("description", f"every {parent} row has a {child} row"),
        f"SELECT SUM(CASE WHEN NOT EXISTS (SELECT 1 FROM {child} AS c WHERE c.{col} = p.{pcol}) "
        f"THEN 1 ELSE 0 END) AS failing, COUNT(*) AS total FROM {parent} AS p",
        **kw,
    )


def reconcile(layer: str, table: str, name: str, a: str, b: str, description: str) -> Check:
    """Two scalar SQL expressions must agree to within a cent (``a`` and ``b`` are queries)."""
    return Check(
        f"{table}: {name}",
        layer,
        table,
        "reconciliation",
        description,
        f"SELECT CASE WHEN ABS(COALESCE(({a}), 0) - COALESCE(({b}), 0)) > 0.01 THEN 1 ELSE 0 END "
        "AS failing, 1 AS total",
    )


# ---------------------------------------------------------------------------- the checks
def raw_checks() -> list[Check]:
    L = "raw"
    return [
        *(row_count(L, t) for t in ("orders", "order_items", "customers", "products", "sellers")),
        unique(L, "orders", ("order_id",)),
        unique(L, "customers", ("customer_id",)),
        unique(L, "products", ("product_id",)),
        unique(L, "sellers", ("seller_id",)),
        unique(L, "order_items", ("order_id", "order_item_id")),
        unique(L, "order_payments", ("order_id", "payment_sequential")),
        unique(L, "order_reviews", ("review_id", "order_id")),
        unique(
            L,
            "order_reviews",
            ("review_id",),
            severity="warn",
            description="review_id alone is not a key: one survey can cover several orders",
        ),
        unique(L, "category_translation", ("product_category_name",)),
        not_null(L, "orders", "customer_id"),
        not_null(L, "orders", "order_purchase_timestamp"),
        not_null(L, "orders", "order_estimated_delivery_date"),
        not_null(L, "customers", "customer_unique_id"),
        not_null(L, "order_items", "price"),
        not_null(L, "order_reviews", "review_score"),
        accepted_values(L, "orders", "order_status", ORDER_STATUSES),
        accepted_values(L, "order_payments", "payment_type", PAYMENT_TYPES),
        accepted_values(L, "customers", "customer_state", BRAZIL_STATES),
        accepted_values(L, "sellers", "seller_state", BRAZIL_STATES),
        condition(L, "order_reviews", "score 1-5", "review_score BETWEEN 1 AND 5", "1-5 stars"),
        condition(L, "order_items", "price > 0", "price > 0", "items have a positive price"),
        condition(L, "order_items", "freight >= 0", "freight_value >= 0", "freight not negative"),
        condition(L, "order_payments", "value >= 0", "payment_value >= 0", "payments not negative"),
        condition(
            L,
            "orders",
            "purchase in 2016-2018",
            "order_purchase_timestamp >= DATE '2016-01-01' "
            "AND order_purchase_timestamp < DATE '2019-01-01'",
            "purchase dates fall inside the extract's period",
        ),
        foreign_key(L, "orders", "customer_id", "customers", "customer_id"),
        foreign_key(L, "order_items", "order_id", "orders", "order_id"),
        foreign_key(L, "order_items", "product_id", "products", "product_id"),
        foreign_key(L, "order_items", "seller_id", "sellers", "seller_id"),
        foreign_key(L, "order_payments", "order_id", "orders", "order_id"),
        foreign_key(L, "order_reviews", "order_id", "orders", "order_id"),
        foreign_key(
            L,
            "products",
            "product_category_name",
            "category_translation",
            "product_category_name",
            severity="warn",
            description="2 categories (13 products) are missing from the translation table; "
            "stg_products translates them",
        ),
        condition(
            L,
            "orders",
            "approved after purchase",
            "order_approved_at IS NULL OR order_approved_at >= order_purchase_timestamp",
            "approval never precedes purchase",
            kind="date_order",
        ),
        condition(
            L,
            "orders",
            "delivered after purchase",
            "order_delivered_customer_date IS NULL "
            "OR order_delivered_customer_date >= order_purchase_timestamp",
            "delivery never precedes purchase",
            kind="date_order",
        ),
        condition(
            L,
            "orders",
            "estimate after purchase",
            "order_estimated_delivery_date >= CAST(order_purchase_timestamp AS DATE)",
            "the promised date is never before the purchase",
            kind="date_order",
        ),
        condition(
            L,
            "orders",
            "shipped after approval",
            "order_delivered_carrier_date IS NULL OR order_approved_at IS NULL "
            "OR order_delivered_carrier_date >= order_approved_at",
            "carrier hand-over after approval (1.4% of orders violate it; flagged in staging)",
            kind="date_order",
            severity="warn",
        ),
        condition(
            L,
            "orders",
            "delivered after shipping",
            "order_delivered_customer_date IS NULL OR order_delivered_carrier_date IS NULL "
            "OR order_delivered_customer_date >= order_delivered_carrier_date",
            "delivery after carrier hand-over (23 orders violate it; flagged in staging)",
            kind="date_order",
            severity="warn",
        ),
        condition(
            L,
            "orders",
            "delivered has date",
            "order_status <> 'delivered' OR order_delivered_customer_date IS NOT NULL",
            "'delivered' orders carry a delivery date (8 do not; excluded from delivery metrics)",
            kind="completeness",
            severity="warn",
        ),
        has_child(
            L,
            "orders",
            "order_id",
            "order_items",
            "order_id",
            severity="warn",
            tolerance=0.01,
            description="orders without items (mostly 'unavailable'/'canceled'): not sales",
        ),
        has_child(
            L,
            "orders",
            "order_id",
            "order_payments",
            "order_id",
            severity="warn",
            tolerance=0.001,
            description="orders without a payment record",
        ),
        has_child(
            L,
            "orders",
            "order_id",
            "order_reviews",
            "order_id",
            severity="warn",
            tolerance=0.02,
            description="orders without a review (excluded from review metrics)",
        ),
    ]


def staging_checks() -> list[Check]:
    L = "staging"
    return [
        condition(
            L, "stg_customers", "zip has 5 digits", "LENGTH(zip_prefix) = 5", "zip normalised"
        ),
        condition(L, "stg_sellers", "zip has 5 digits", "LENGTH(zip_prefix) = 5", "zip normalised"),
        unique(L, "stg_geolocation", ("zip_prefix",)),
        unique(L, "stg_order_reviews", ("review_id", "order_id")),
        unique(L, "stg_products", ("product_id",)),
        not_null(L, "stg_products", "category"),
        foreign_key(
            L,
            "stg_customers",
            "zip_prefix",
            "stg_geolocation",
            "zip_prefix",
            severity="warn",
            tolerance=0.01,
            description="customer zip prefixes with coordinates (distance features need them)",
        ),
        foreign_key(
            L,
            "stg_sellers",
            "zip_prefix",
            "stg_geolocation",
            "zip_prefix",
            severity="warn",
            tolerance=0.01,
            description="seller zip prefixes with coordinates",
        ),
        condition(
            L,
            "stg_order_payments",
            "instalments >= 1",
            "installments >= 1",
            "every payment has at least one instalment",
        ),
    ]


def mart_checks() -> list[Check]:
    L = "marts"
    return [
        unique(L, "fct_orders", ("order_id",)),
        unique(L, "fct_order_items", ("order_id", "order_item_id")),
        unique(L, "dim_customers", ("customer_unique_id",)),
        unique(L, "dim_sellers", ("seller_id",)),
        unique(L, "dim_products", ("product_id",)),
        reconcile(
            L,
            "fct_orders",
            "row count = raw orders",
            "SELECT COUNT(*) FROM fct_orders",
            "SELECT COUNT(*) FROM orders",
            "no order is lost or duplicated by the joins",
        ),
        reconcile(
            L,
            "fct_order_items",
            "row count = raw items",
            "SELECT COUNT(*) FROM fct_order_items",
            "SELECT COUNT(*) FROM order_items",
            "no item is lost or duplicated",
        ),
        reconcile(
            L,
            "fct_orders",
            "item revenue = raw prices",
            "SELECT SUM(item_revenue) FROM fct_orders",
            "SELECT SUM(price) FROM order_items",
            "order-level item revenue reconciles with the raw items",
        ),
        reconcile(
            L,
            "fct_orders",
            "freight = raw freight",
            "SELECT SUM(freight_value) FROM fct_orders",
            "SELECT SUM(freight_value) FROM order_items",
            "order-level freight reconciles with the raw items",
        ),
        reconcile(
            L,
            "fct_orders",
            "payments = raw payments",
            "SELECT SUM(payment_value) FROM fct_orders",
            "SELECT SUM(payment_value) FROM order_payments "
            "WHERE order_id IN (SELECT order_id FROM orders)",
            "order-level payment value reconciles with the raw payments",
        ),
        reconcile(
            L,
            "dim_customers",
            "people = raw unique ids",
            "SELECT COUNT(*) FROM dim_customers",
            "SELECT COUNT(DISTINCT customer_unique_id) FROM customers",
            "one row per person",
        ),
        reconcile(
            L,
            "fct_orders",
            "first orders = buyers",
            "SELECT COUNT(*) FROM fct_orders WHERE customer_order_seq = 1",
            "SELECT COUNT(*) FROM dim_customers WHERE n_orders > 0",
            "each buyer has exactly one first order",
        ),
        condition(
            L,
            "fct_orders",
            "gmv = items + freight",
            "gmv IS NULL OR ABS(gmv - item_revenue - freight_value) < 0.000001",
            "GMV identity holds on every order",
            kind="identity",
        ),
        condition(
            L,
            "fct_orders",
            "valid orders have a cohort",
            "is_valid = 0 OR (cohort_month IS NOT NULL AND customer_order_seq >= 1)",
            "every valid order belongs to its customer's acquisition cohort",
            kind="identity",
        ),
        condition(
            L,
            "fct_orders",
            "cohort <= purchase month",
            "cohort_month IS NULL OR cohort_month <= purchase_month",
            "a customer's cohort is never after one of their orders",
            kind="identity",
        ),
        condition(
            L,
            "fct_orders",
            "late only when delivered",
            "is_late IS NULL OR is_delivered = 1",
            "lateness is defined only for delivered orders",
            kind="identity",
        ),
        condition(
            L,
            "fct_orders",
            "delivery days >= 0",
            "delivery_days IS NULL OR delivery_days >= 0",
            "no negative delivery times",
        ),
        condition(
            L,
            "fct_orders",
            "payment close to gmv",
            "payment_value IS NULL OR gmv IS NULL OR ABS(payment_minus_gmv) <= 1",
            "payments within R$ 1 of GMV (instalment interest and vouchers explain the rest)",
            severity="warn",
            tolerance=0.01,
        ),
        foreign_key(L, "fct_order_items", "seller_id", "dim_sellers", "seller_id"),
        foreign_key(L, "fct_order_items", "product_id", "dim_products", "product_id"),
        foreign_key(L, "fct_orders", "customer_unique_id", "dim_customers", "customer_unique_id"),
    ]


ALL_CHECKS = {"raw": raw_checks, "staging": staging_checks, "marts": mart_checks}


def run_check(engine: Engine, check: Check) -> CheckResult:
    row = engine.query(check.sql).iloc[0]
    failing = 0 if pd.isna(row["failing"]) else int(row["failing"])
    total = 0 if pd.isna(row["total"]) else int(row["total"])
    share = failing / total if total else 0.0
    if failing == 0:
        status = "pass"
    elif check.severity == "warn" or share <= check.tolerance:
        status = "warn"
    else:
        status = "fail"
    return CheckResult(
        check.name,
        check.layer,
        check.table,
        check.kind,
        check.description,
        check.severity,
        check.tolerance,
        failing,
        total,
        status,
    )


def run_checks(engine: Engine, layer: str) -> list[CheckResult]:
    return [run_check(engine, c) for c in ALL_CHECKS[layer]()]


class DataQualityError(RuntimeError):
    """At least one error-severity check failed above its tolerance."""

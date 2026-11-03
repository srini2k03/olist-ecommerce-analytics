"""A small, schema-faithful synthetic Olist dataset for tests.

It is NOT used for any reported number. It exists so the SQL, the feature
pipeline and the app build can be tested without network access. Volumes are
chosen so every query in ``sql/`` (some have minimum-volume filters) returns
rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from olist.schema import TABLES_BY_NAME

STATES = ["SP", "RJ", "MG", "BA"]
SELLER_STATES = ["SP", "PR"]
CATEGORIES = [
    ("beleza_saude", "health_beauty"),
    ("cama_mesa_banho", "bed_bath_table"),
    ("telefonia", "telephony"),
    ("categoria_sem_traducao", None),
]
COMMENTS = [
    "Recomendo, chegou rápido",
    "produto excelente",
    "ainda não recebi o produto",
    "veio com defeito",
    "entrega atrasada",
    None,
    None,
]


def _hex(rng: np.random.Generator, n: int) -> list[str]:
    return [rng.bytes(16).hex() for _ in range(n)]


def generate(n_customers: int = 4500, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2017-01-01")
    span_days = 600  # to late August 2018

    people = _hex(rng, n_customers)
    person_state = rng.choice(STATES, n_customers, p=[0.55, 0.2, 0.15, 0.1])
    # 8% of people buy twice, 2% three times.
    n_orders = rng.choice([1, 2, 3], n_customers, p=[0.9, 0.08, 0.02])
    rows = []
    for i, k in enumerate(n_orders):
        t0 = start + pd.Timedelta(minutes=int(rng.integers(0, span_days * 1440)))
        for j in range(k):
            gap = 0 if j == 0 else int(rng.integers(0, 200))
            t = t0 + pd.Timedelta(days=gap * j, minutes=int(rng.integers(0, 600)))
            rows.append((people[i], person_state[i], t))
    orders_base = pd.DataFrame(rows, columns=["person", "state", "ts"])
    orders_base = orders_base[orders_base["ts"] < pd.Timestamp("2018-09-01")]
    n = len(orders_base)
    order_ids = _hex(rng, n)
    customer_ids = _hex(rng, n)  # one customer_id per order, as in the real data

    status = rng.choice(
        ["delivered", "shipped", "canceled", "unavailable", "invoiced"],
        n,
        p=[0.94, 0.02, 0.015, 0.01, 0.015],
    )
    purchase = orders_base["ts"].to_numpy()
    approved = purchase + pd.to_timedelta(rng.integers(10, 2000, n), unit="m")
    carrier = approved + pd.to_timedelta(rng.integers(1, 5, n), unit="D")
    promised = rng.integers(15, 35, n)
    estimated = pd.to_datetime(purchase).normalize() + pd.to_timedelta(promised, unit="D")
    estimated = estimated.to_numpy()
    actual = rng.gamma(4.0, 3.5, n).round() + 2
    late = rng.random(n) < 0.08
    actual = np.where(late, promised + rng.integers(1, 20, n), actual)
    delivered = (
        purchase
        + pd.to_timedelta(actual, unit="D")
        + pd.to_timedelta(rng.integers(0, 1440, n), unit="m")
    )
    is_delivered = status == "delivered"
    nat = np.datetime64("NaT", "ns")
    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
            "order_status": status,
            "order_purchase_timestamp": purchase,
            "order_approved_at": approved,
            "order_delivered_carrier_date": np.where(is_delivered, carrier, nat),
            "order_delivered_customer_date": np.where(is_delivered, delivered, nat),
            "order_estimated_delivery_date": estimated,
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_unique_id": orders_base["person"].to_numpy(),
            "customer_zip_code_prefix": [f"{z:05d}" for z in rng.integers(1000, 99999, n)],
            "customer_city": "cidade",
            "customer_state": orders_base["state"].to_numpy(),
        }
    )

    n_sellers, n_products = 12, 120
    sellers = pd.DataFrame(
        {
            "seller_id": _hex(rng, n_sellers),
            "seller_zip_code_prefix": [f"{z:05d}" for z in rng.integers(1000, 99999, n_sellers)],
            "seller_city": "cidade",
            "seller_state": rng.choice(SELLER_STATES, n_sellers, p=[0.75, 0.25]),
        }
    )
    cat_idx = rng.integers(0, len(CATEGORIES), n_products)
    cat_names = [CATEGORIES[i][0] for i in cat_idx]
    cat_names[0] = None  # a product without a category, as in the real data
    products = pd.DataFrame(
        {
            "product_id": _hex(rng, n_products),
            "product_category_name": cat_names,
            "product_name_lenght": rng.integers(10, 60, n_products),
            "product_description_lenght": rng.integers(50, 3000, n_products),
            "product_photos_qty": rng.integers(1, 6, n_products),
            "product_weight_g": rng.integers(100, 10000, n_products),
            "product_length_cm": rng.integers(10, 60, n_products),
            "product_height_cm": rng.integers(2, 40, n_products),
            "product_width_cm": rng.integers(10, 50, n_products),
        }
    )
    base_price = rng.lognormal(4.3, 0.8, n_products).round(2)

    item_rows = []
    n_items = rng.choice([1, 2, 3, 4], n, p=[0.88, 0.08, 0.025, 0.015])
    for oid, k, ts in zip(order_ids, n_items, purchase, strict=True):
        p = int(rng.integers(0, n_products))
        s = int(rng.integers(0, n_sellers))
        for item in range(1, k + 1):
            item_rows.append(
                (
                    oid,
                    item,
                    products["product_id"].iat[p],
                    sellers["seller_id"].iat[s],
                    pd.Timestamp(ts) + pd.Timedelta(days=3),
                    float(base_price[p]),
                    round(float(rng.gamma(2.0, 9.0)), 2),
                )
            )
    order_items = pd.DataFrame(item_rows, columns=TABLES_BY_NAME["order_items"].column_names)

    totals = order_items.groupby("order_id")[["price", "freight_value"]].sum().sum(axis=1)
    pay_rows = []
    for oid in order_ids:
        total = round(float(totals.get(oid, 50.0)), 2)
        r = rng.random()
        if r < 0.05:  # voucher + card split
            v = round(total * 0.3, 2)
            pay_rows.append((oid, 1, "voucher", 1, v))
            pay_rows.append((oid, 2, "credit_card", int(rng.integers(1, 10)), round(total - v, 2)))
        elif r < 0.75:
            pay_rows.append((oid, 1, "credit_card", int(rng.integers(1, 10)), total))
        elif r < 0.97:
            pay_rows.append((oid, 1, "boleto", 1, total))
        else:
            pay_rows.append((oid, 1, "debit_card", 1, total))
    order_payments = pd.DataFrame(pay_rows, columns=TABLES_BY_NAME["order_payments"].column_names)

    delay = (pd.to_datetime(delivered).normalize() - pd.to_datetime(estimated)).days.to_numpy()
    score = np.clip(np.round(4.4 - 0.25 * np.clip(delay, 0, None) + rng.normal(0, 1, n)), 1, 5)
    has_review = rng.random(n) < 0.98
    review_created = pd.to_datetime(np.where(is_delivered, delivered, estimated)) + pd.Timedelta(
        days=1
    )
    review_created = review_created.normalize()
    comments = rng.choice(np.array(COMMENTS, dtype=object), n)
    order_reviews = pd.DataFrame(
        {
            "review_id": _hex(rng, n),
            "order_id": order_ids,
            "review_score": score.astype(int),
            "review_comment_title": None,
            "review_comment_message": comments,
            "review_creation_date": review_created,
            "review_answer_timestamp": review_created + pd.Timedelta(hours=20),
        }
    )[has_review]

    geolocation = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": customers["customer_zip_code_prefix"].head(200),
            "geolocation_lat": rng.uniform(-30, -3, 200),
            "geolocation_lng": rng.uniform(-60, -35, 200),
            "geolocation_city": "cidade",
            "geolocation_state": customers["customer_state"].head(200),
        }
    )
    category_translation = pd.DataFrame(
        [(pt, en) for pt, en in CATEGORIES if en is not None],
        columns=["product_category_name", "product_category_name_english"],
    )
    return {
        "customers": customers,
        "geolocation": geolocation,
        "order_items": order_items,
        "order_payments": order_payments,
        "order_reviews": order_reviews,
        "orders": orders,
        "products": products,
        "sellers": sellers,
        "category_translation": category_translation,
    }


def write_csvs(out_dir: Path, **kwargs) -> Path:
    """Write the synthetic tables using the real CSV file names."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in generate(**kwargs).items():
        table = TABLES_BY_NAME[name]
        df[table.column_names].to_csv(
            out_dir / table.csv, index=False, date_format="%Y-%m-%d %H:%M:%S"
        )
    return out_dir

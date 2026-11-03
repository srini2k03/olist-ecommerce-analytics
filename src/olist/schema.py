"""Table schema for the nine Olist CSVs.

One portable DDL is used for both DuckDB and Postgres so the same SQL runs on
either engine. Types are limited to TEXT, INTEGER, DOUBLE PRECISION and
TIMESTAMP, which mean the same thing in both.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Table:
    name: str
    csv: str
    columns: tuple[tuple[str, str], ...]

    @property
    def column_names(self) -> list[str]:
        return [c for c, _ in self.columns]

    def ddl(self) -> str:
        cols = ",\n  ".join(f"{c} {t}" for c, t in self.columns)
        return f"CREATE TABLE {self.name} (\n  {cols}\n)"


TABLES: tuple[Table, ...] = (
    Table(
        "customers",
        "olist_customers_dataset.csv",
        (
            ("customer_id", "TEXT"),
            ("customer_unique_id", "TEXT"),
            ("customer_zip_code_prefix", "TEXT"),
            ("customer_city", "TEXT"),
            ("customer_state", "TEXT"),
        ),
    ),
    Table(
        "geolocation",
        "olist_geolocation_dataset.csv",
        (
            ("geolocation_zip_code_prefix", "TEXT"),
            ("geolocation_lat", "DOUBLE PRECISION"),
            ("geolocation_lng", "DOUBLE PRECISION"),
            ("geolocation_city", "TEXT"),
            ("geolocation_state", "TEXT"),
        ),
    ),
    Table(
        "order_items",
        "olist_order_items_dataset.csv",
        (
            ("order_id", "TEXT"),
            ("order_item_id", "INTEGER"),
            ("product_id", "TEXT"),
            ("seller_id", "TEXT"),
            ("shipping_limit_date", "TIMESTAMP"),
            ("price", "DOUBLE PRECISION"),
            ("freight_value", "DOUBLE PRECISION"),
        ),
    ),
    Table(
        "order_payments",
        "olist_order_payments_dataset.csv",
        (
            ("order_id", "TEXT"),
            ("payment_sequential", "INTEGER"),
            ("payment_type", "TEXT"),
            ("payment_installments", "INTEGER"),
            ("payment_value", "DOUBLE PRECISION"),
        ),
    ),
    Table(
        "order_reviews",
        "olist_order_reviews_dataset.csv",
        (
            ("review_id", "TEXT"),
            ("order_id", "TEXT"),
            ("review_score", "INTEGER"),
            ("review_comment_title", "TEXT"),
            ("review_comment_message", "TEXT"),
            ("review_creation_date", "TIMESTAMP"),
            ("review_answer_timestamp", "TIMESTAMP"),
        ),
    ),
    Table(
        "orders",
        "olist_orders_dataset.csv",
        (
            ("order_id", "TEXT"),
            ("customer_id", "TEXT"),
            ("order_status", "TEXT"),
            ("order_purchase_timestamp", "TIMESTAMP"),
            ("order_approved_at", "TIMESTAMP"),
            ("order_delivered_carrier_date", "TIMESTAMP"),
            ("order_delivered_customer_date", "TIMESTAMP"),
            ("order_estimated_delivery_date", "TIMESTAMP"),
        ),
    ),
    Table(
        "products",
        "olist_products_dataset.csv",
        (
            ("product_id", "TEXT"),
            ("product_category_name", "TEXT"),
            # The misspellings are the original column names.
            ("product_name_lenght", "INTEGER"),
            ("product_description_lenght", "INTEGER"),
            ("product_photos_qty", "INTEGER"),
            ("product_weight_g", "INTEGER"),
            ("product_length_cm", "INTEGER"),
            ("product_height_cm", "INTEGER"),
            ("product_width_cm", "INTEGER"),
        ),
    ),
    Table(
        "sellers",
        "olist_sellers_dataset.csv",
        (
            ("seller_id", "TEXT"),
            ("seller_zip_code_prefix", "TEXT"),
            ("seller_city", "TEXT"),
            ("seller_state", "TEXT"),
        ),
    ),
    Table(
        "category_translation",
        "product_category_name_translation.csv",
        (
            ("product_category_name", "TEXT"),
            ("product_category_name_english", "TEXT"),
        ),
    ),
)

TABLES_BY_NAME = {t.name: t for t in TABLES}

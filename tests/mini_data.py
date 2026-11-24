"""A hand-made, 6-order Olist dataset whose correct answers are worked out by hand.

The known-answer tests assert exact values (money, cohorts, repeat flags, delivery days,
lateness, review averages, data-quality outcomes) on it. Every edge case below exists in
the real data:

- customer u1 orders twice (o1 in Jan, o2 in Mar 2017) under two customer_ids;
- o1 has two items from two sellers; o2 is paid with a voucher + card;
- o1's payment includes R$ 5 of instalment interest (payment 170 vs GMV 165);
- o2 is delivered 5 days late and has two reviews (scores 1 and 3, mean 2.0);
- o4 is shipped but not delivered; its review reuses o3's review_id;
- o5 is cancelled (with an item), o6 is unavailable without items;
- zip prefixes lost their leading zero ("1234"); one geolocation point lies outside Brazil;
- product p2 has no category, p3's category ('pc_gamer') is missing from the translation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from olist.schema import TABLES_BY_NAME

T = pd.Timestamp


def tables() -> dict[str, pd.DataFrame]:
    customers = pd.DataFrame(
        [
            ("c1a", "u1", "1234", "sao paulo", "SP"),
            ("c1b", "u1", "1234", "sao paulo", "SP"),
            ("c2", "u2", "20000", "rio de janeiro", "RJ"),
            ("c3", "u3", "30000", "belo horizonte", "MG"),
            ("c3b", "u3", "30000", "belo horizonte", "MG"),
            ("c4", "u4", "01000", "sao paulo", "SP"),
        ],
        columns=TABLES_BY_NAME["customers"].column_names,
    )
    sellers = pd.DataFrame(
        [("s1", "1310", "sao paulo", "SP"), ("s2", "80000", "curitiba", "PR")],
        columns=TABLES_BY_NAME["sellers"].column_names,
    )
    products = pd.DataFrame(
        [
            ("p1", "beleza_saude", 40, 500, 2, 500, 20, 10, 15),
            ("p2", None, 30, 300, 1, 200, 10, 10, 10),
            ("p3", "pc_gamer", 50, 900, 4, 2000, 40, 30, 20),
        ],
        columns=TABLES_BY_NAME["products"].column_names,
    )
    category_translation = pd.DataFrame(
        [("beleza_saude", "health_beauty")],
        columns=TABLES_BY_NAME["category_translation"].column_names,
    )
    nat = pd.NaT
    orders = pd.DataFrame(
        [
            # id, customer, status, purchase, approved, carrier, delivered, estimated
            ("o1", "c1a", "delivered", T("2017-01-10 10:00"), T("2017-01-10 11:00"),
             T("2017-01-12 09:00"), T("2017-01-20 15:00"), T("2017-01-25")),
            ("o2", "c1b", "delivered", T("2017-03-05 09:00"), T("2017-03-05 10:00"),
             T("2017-03-07 09:00"), T("2017-03-30 12:00"), T("2017-03-25")),
            ("o3", "c2", "delivered", T("2017-01-20 20:00"), T("2017-01-21 08:00"),
             T("2017-01-23 09:00"), T("2017-02-01 10:00"), T("2017-02-10")),
            ("o4", "c3", "shipped", T("2017-02-15 12:00"), T("2017-02-15 13:00"),
             T("2017-02-17 09:00"), nat, T("2017-03-10")),
            ("o5", "c4", "canceled", T("2017-02-01 08:00"), T("2017-02-01 09:00"),
             nat, nat, T("2017-02-20")),
            ("o6", "c3b", "unavailable", T("2017-04-01 08:00"), T("2017-04-01 09:00"),
             nat, nat, T("2017-04-20")),
        ],
        columns=TABLES_BY_NAME["orders"].column_names,
    )  # fmt: skip
    order_items = pd.DataFrame(
        [
            ("o1", 1, "p1", "s1", T("2017-01-12"), 100.0, 10.0),
            ("o1", 2, "p2", "s2", T("2017-01-12"), 50.0, 5.0),
            ("o2", 1, "p3", "s2", T("2017-03-07"), 200.0, 20.0),
            ("o3", 1, "p1", "s1", T("2017-01-23"), 80.0, 8.0),
            ("o4", 1, "p2", "s1", T("2017-02-17"), 40.0, 4.0),
            ("o5", 1, "p1", "s1", T("2017-02-03"), 30.0, 3.0),
        ],
        columns=TABLES_BY_NAME["order_items"].column_names,
    )
    order_payments = pd.DataFrame(
        [
            ("o1", 1, "credit_card", 3, 170.0),
            ("o2", 1, "voucher", 1, 20.0),
            ("o2", 2, "credit_card", 2, 200.0),
            ("o3", 1, "boleto", 1, 88.0),
            ("o4", 1, "credit_card", 0, 44.0),
            ("o5", 1, "credit_card", 1, 33.0),
        ],
        columns=TABLES_BY_NAME["order_payments"].column_names,
    )
    order_reviews = pd.DataFrame(
        [
            ("r1", "o1", 5, None, "ótimo, recomendo", T("2017-01-21"), T("2017-01-22 10:00")),
            ("r2", "o2", 1, None, "atrasou muito", T("2017-03-31"), T("2017-04-01 10:00")),
            ("r3", "o2", 3, None, None, T("2017-04-02"), T("2017-04-03 10:00")),
            ("r4", "o3", 4, None, None, T("2017-02-02"), T("2017-02-03 10:00")),
            ("r4", "o4", 2, None, "ainda não recebi", T("2017-03-11"), T("2017-03-12 10:00")),
        ],
        columns=TABLES_BY_NAME["order_reviews"].column_names,
    )
    geolocation = pd.DataFrame(
        [
            ("01234", -23.5, -46.6, "sao paulo", "SP"),
            ("01234", -23.7, -46.8, "sao paulo", "SP"),
            ("01234", 40.0, 10.0, "sao paulo", "SP"),  # outside Brazil: dropped
            ("01310", -23.56, -46.66, "sao paulo", "SP"),
            ("20000", -22.9, -43.2, "rio de janeiro", "RJ"),
            ("80000", -25.4, -49.3, "curitiba", "PR"),
        ],
        columns=TABLES_BY_NAME["geolocation"].column_names,
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


def write_csvs(out_dir: Path, overrides: dict[str, pd.DataFrame] | None = None) -> Path:
    """Write the mini tables with the real CSV names (zip prefixes keep their text form)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = tables() | (overrides or {})
    for name, df in data.items():
        table = TABLES_BY_NAME[name]
        df[table.column_names].to_csv(
            out_dir / table.csv, index=False, date_format="%Y-%m-%d %H:%M:%S"
        )
    return out_dir

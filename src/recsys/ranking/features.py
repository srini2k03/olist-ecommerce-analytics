"""Point-in-time ranking features.

A :class:`FeatureStore` is built from the transactions strictly before
``as_of`` (a fold's start) and holds per-user, per-article and per-user-history
tables. :func:`assemble` joins them onto a candidate frame and adds the
interaction features. The same code serves offline training and the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data.split import check_strictly_before

DAY = pd.Timedelta(days=1)

ITEM_ATTR_COLUMNS = [
    "product_type_no",
    "product_group_code",
    "colour_group_code",
    "department_no",
    "garment_group_no",
    "index_group_no",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "user": [
        "u_age",
        "u_recency_days",
        "u_frequency",
        "u_mean_price",
        "u_channel2_share",
        "u_active_weeks",
        "u_n_items",
    ],
    "item": [
        "i_pop_1w",
        "i_pop_4w",
        "i_trend",
        "i_mean_price",
        "i_age_days",
        *ITEM_ATTR_COLUMNS,
    ],
    "two_tower": ["tt_score", "tt_rank"],
    "cooccurrence": ["cooc_score"],
    "history": [
        "ui_count",
        "ui_days_since",
        "uc_days_since",
        "u_type_affinity",
        "u_garment_affinity",
        "u_index_affinity",
        "price_vs_user",
    ],
    "sources": [
        "src_two_tower",
        "src_cooc",
        "src_repurchase",
        "src_age_pop",
        "n_sources",
        "agepop_rank",
    ],
}
FEATURES = [f for cols in FEATURE_GROUPS.values() for f in cols]
CATEGORICAL = ["product_group_code", "garment_group_no", "index_group_no"]


@dataclass
class FeatureStore:
    as_of: pd.Timestamp
    users: pd.DataFrame  # index customer_id
    items: pd.DataFrame  # index article_id
    user_items: pd.DataFrame  # customer_id, article_id, ui_count, ui_last
    user_codes: pd.DataFrame  # customer_id, product_code, uc_last
    user_affinity: dict[str, pd.DataFrame]  # attr -> (customer_id, attr, share)

    @classmethod
    def build(
        cls,
        history: pd.DataFrame,
        articles: pd.DataFrame,
        customers: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> FeatureStore:
        check_strictly_before(history, as_of, "feature history")
        as_of = pd.Timestamp(as_of)
        h = history.merge(
            articles[
                [
                    "article_id",
                    "product_code",
                    "product_type_no",
                    "garment_group_no",
                    "index_group_no",
                ]
            ],
            on="article_id",
            how="left",
        )
        week = ((as_of - h["t_dat"]).dt.days // 7).rename("week")
        g = h.groupby("customer_id")
        users = pd.DataFrame(
            {
                "u_recency_days": (as_of - g["t_dat"].max()).dt.days,
                "u_frequency": g.size(),
                "u_mean_price": g["price"].mean(),
                "u_channel2_share": (h["sales_channel_id"] == 2).groupby(h["customer_id"]).mean(),
                "u_active_weeks": week.groupby(h["customer_id"]).nunique(),
                "u_n_items": g["article_id"].nunique(),
            }
        )
        age = customers.set_index("customer_id")["age"]
        users = users.reindex(users.index.union(age.index))
        users["u_age"] = age.reindex(users.index)
        users.index.name = "customer_id"

        items = articles.set_index("article_id")[
            [
                "product_code",
                "product_type_no",
                "colour_group_code",
                "department_no",
                "garment_group_no",
                "index_group_no",
            ]
        ].copy()
        items["product_group_code"] = (
            articles["product_group_name"].astype("category").cat.codes.to_numpy()
        )
        ia = h.groupby("article_id")
        recent = h[h["t_dat"] >= as_of - 7 * DAY].groupby("article_id").size()
        month = h[h["t_dat"] >= as_of - 28 * DAY].groupby("article_id").size()
        items["i_pop_1w"] = recent.reindex(items.index).fillna(0)
        items["i_pop_4w"] = month.reindex(items.index).fillna(0)
        items["i_trend"] = (items["i_pop_1w"] + 1) / (items["i_pop_4w"] / 4 + 1)
        items["i_mean_price"] = ia["price"].mean().reindex(items.index)
        items["i_age_days"] = (as_of - ia["t_dat"].min()).dt.days.reindex(items.index)

        ui = (
            h.groupby(["customer_id", "article_id"])["t_dat"]
            .agg(ui_count="size", ui_last="max")
            .reset_index()
        )
        uc = (
            h.groupby(["customer_id", "product_code"])["t_dat"]
            .max()
            .rename("uc_last")
            .reset_index()
        )
        affinity = {}
        for attr in ("product_type_no", "garment_group_no", "index_group_no"):
            counts = h.groupby(["customer_id", attr]).size().rename("n").reset_index()
            counts["share"] = counts["n"] / counts.groupby("customer_id")["n"].transform("sum")
            affinity[attr] = counts.drop(columns="n")
        return cls(as_of, users, items, ui, uc, affinity)

    # ------------------------------------------------------------------ persistence

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.users.to_parquet(folder / "users.parquet")
        self.items.to_parquet(folder / "items.parquet")
        self.user_items.to_parquet(folder / "user_items.parquet", index=False)
        self.user_codes.to_parquet(folder / "user_codes.parquet", index=False)
        for attr, df in self.user_affinity.items():
            df.to_parquet(folder / f"affinity_{attr}.parquet", index=False)
        (folder / "as_of.txt").write_text(self.as_of.date().isoformat())

    @classmethod
    def load(cls, folder: Path) -> FeatureStore:
        affinity = {
            p.stem.removeprefix("affinity_"): pd.read_parquet(p)
            for p in sorted(folder.glob("affinity_*.parquet"))
        }
        return cls(
            pd.Timestamp((folder / "as_of.txt").read_text().strip()),
            pd.read_parquet(folder / "users.parquet"),
            pd.read_parquet(folder / "items.parquet"),
            pd.read_parquet(folder / "user_items.parquet"),
            pd.read_parquet(folder / "user_codes.parquet"),
            affinity,
        )


def assemble(cands: pd.DataFrame, store: FeatureStore) -> pd.DataFrame:
    """Add every feature in :data:`FEATURES` to a candidate frame.

    ``cands`` needs ``customer_id``, ``article_id`` and the candidate-source
    columns (``tt_score``, ``tt_rank``, ``cooc_score``, ``agepop_rank``, ``src_*``).
    Row order is preserved.
    """
    df = cands.reset_index(drop=True)
    df = df.join(store.users, on="customer_id")
    df = df.join(store.items, on="article_id")
    df = df.merge(store.user_items, on=["customer_id", "article_id"], how="left")
    df = df.merge(store.user_codes, on=["customer_id", "product_code"], how="left")
    df["ui_count"] = df["ui_count"].fillna(0)
    df["ui_days_since"] = (store.as_of - df["ui_last"]).dt.days
    df["uc_days_since"] = (store.as_of - df["uc_last"]).dt.days
    for attr, name in (
        ("product_type_no", "u_type_affinity"),
        ("garment_group_no", "u_garment_affinity"),
        ("index_group_no", "u_index_affinity"),
    ):
        aff = store.user_affinity[attr].rename(columns={"share": name})
        df = df.merge(aff, on=["customer_id", attr], how="left")
        df[name] = df[name].fillna(0)
    df["price_vs_user"] = df["i_mean_price"] / df["u_mean_price"]
    df["cooc_score"] = df["cooc_score"].fillna(0)
    src = ["src_two_tower", "src_cooc", "src_repurchase", "src_age_pop"]
    df[src] = df[src].fillna(0).astype(np.int8)
    df["n_sources"] = df[src].sum(axis=1)
    return df

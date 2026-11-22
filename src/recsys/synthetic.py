"""A small, seeded, H&M-shaped synthetic dataset for tests and CI.

It is NOT used for any reported number. It has the same tables, columns and
dtypes as the processed H&M parquet files, with the structure the baselines
are meant to pick up:

* power-law (Zipf) article popularity;
* age-dependent taste (each index group is popular with one age band);
* repeat purchases of the customer's own past articles;
* co-purchases of colour variants of the same ``product_code`` and of a fixed
  "goes with" partner article;
* articles released and customers first seen in the final week (cold start).

``to_raw_csvs`` writes the same data in the raw competition format (hex
customer ids, zero-padded article ids) so the converter can be tested too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import RANDOM_SEED
from recsys.data.schema import HMData, hash_customer_ids

INDEX_GROUPS = [  # (index_group_no, name, age band it is popular with)
    (1, "Ladieswear", (25, 45)),
    (2, "Divided", (16, 25)),
    (3, "Menswear", (25, 60)),
    (4, "Baby/Children", (30, 45)),
    (26, "Sport", (16, 35)),
]
PRODUCT_TYPES = [
    (253, "Vest top"),
    (272, "Trousers"),
    (255, "T-shirt"),
    (306, "Bra"),
    (265, "Dress"),
]
PRODUCT_GROUPS = ["Upper body", "Lower body", "Garment Full body", "Underwear", "Accessories"]
COLOURS = [(9, "Black"), (10, "White"), (73, "Dark Blue"), (13, "Beige"), (6, "Light Grey")]
APPEARANCES = [(1010016, "Solid"), (1010017, "Stripe"), (1010001, "All over pattern")]


def _hex_ids(rng: np.random.Generator, n: int) -> list[str]:
    return [rng.bytes(32).hex() for _ in range(n)]


def _articles(rng: np.random.Generator, n_articles: int) -> pd.DataFrame:
    n_products = max(1, n_articles // 3)
    product_codes = rng.choice(np.arange(100_000, 960_000), n_products, replace=False)
    product_of = np.sort(rng.integers(0, n_products, n_articles))
    variant = np.zeros(n_articles, dtype=int)
    for i in range(1, n_articles):
        variant[i] = variant[i - 1] + 1 if product_of[i] == product_of[i - 1] else 0
    codes = product_codes[product_of]
    group = rng.integers(0, len(INDEX_GROUPS), n_products)[product_of]
    ptype = rng.integers(0, len(PRODUCT_TYPES), n_products)[product_of]
    colour = rng.integers(0, len(COLOURS), n_articles)
    appear = rng.integers(0, len(APPEARANCES), n_articles)
    dept = 1000 + 10 * group + ptype
    return pd.DataFrame(
        {
            "article_id": codes * 1000 + 1 + variant,
            "product_code": codes,
            "prod_name": [f"Product {c}" for c in codes],
            "product_type_no": [PRODUCT_TYPES[t][0] for t in ptype],
            "product_type_name": [PRODUCT_TYPES[t][1] for t in ptype],
            "product_group_name": [PRODUCT_GROUPS[t] for t in ptype],
            "graphical_appearance_no": [APPEARANCES[a][0] for a in appear],
            "graphical_appearance_name": [APPEARANCES[a][1] for a in appear],
            "colour_group_code": [COLOURS[c][0] for c in colour],
            "colour_group_name": [COLOURS[c][1] for c in colour],
            "perceived_colour_value_id": colour % 4 + 1,
            "perceived_colour_value_name": [
                ["Dark", "Light", "Dusty", "Bright"][c % 4] for c in colour
            ],
            "perceived_colour_master_id": colour + 1,
            "perceived_colour_master_name": [COLOURS[c][1] for c in colour],
            "department_no": dept,
            "department_name": [f"Department {d}" for d in dept],
            "index_code": [chr(ord("A") + g) for g in group],
            "index_name": [INDEX_GROUPS[g][1] for g in group],
            "index_group_no": [INDEX_GROUPS[g][0] for g in group],
            "index_group_name": [INDEX_GROUPS[g][1] for g in group],
            "section_no": 10 + group,
            "section_name": [f"Section {g}" for g in group],
            "garment_group_no": 1001 + ptype,
            "garment_group_name": [f"Garment group {t}" for t in ptype],
            "detail_desc": [f"Synthetic article, variant {v + 1}." for v in variant],
        }
    )


def _customers(rng: np.random.Generator, n_customers: int) -> tuple[pd.DataFrame, list[str]]:
    hex_ids = _hex_ids(rng, n_customers)
    age = np.clip(rng.gamma(6.0, 6.0, n_customers) + 14, 16, 90).round()
    age[rng.random(n_customers) < 0.01] = np.nan
    fn = np.where(rng.random(n_customers) < 0.35, 1.0, np.nan)
    customers = pd.DataFrame(
        {
            "customer_id": hash_customer_ids(hex_ids),
            "FN": fn,
            "Active": fn,
            "club_member_status": rng.choice(["ACTIVE", "PRE-CREATE", "LEFT CLUB"], n_customers),
            "fashion_news_frequency": rng.choice(["NONE", "Regularly", "Monthly"], n_customers),
            "age": age,
            "postal_code": [h[:20] for h in _hex_ids(rng, n_customers)],
        }
    )
    return customers, hex_ids


def _generate(
    n_customers: int = 2_000,
    n_articles: int = 1_000,
    n_transactions: int = 30_000,
    n_weeks: int = 20,
    end_date: str = "2020-09-22",
    new_item_frac: float = 0.03,
    new_customer_frac: float = 0.05,
    seed: int = RANDOM_SEED,
) -> tuple[HMData, list[str]]:
    rng = np.random.default_rng(seed)
    articles = _articles(rng, n_articles)
    customers, hex_ids = _customers(rng, n_customers)

    end = pd.Timestamp(end_date)
    days = pd.date_range(end - pd.Timedelta(days=7 * n_weeks - 1), end, freq="D")
    last_week_start = len(days) - 7

    # Article popularity: Zipf over a random ranking; new articles only in the last week.
    rank = rng.permutation(n_articles) + 1
    base = 1.0 / rank**0.8
    is_new_item = rng.random(n_articles) < new_item_frac
    group_of_article = articles["index_group_no"].to_numpy()
    price = rng.lognormal(np.log(0.025), 0.5, n_articles).astype(np.float32)
    partner = rng.integers(0, n_articles, n_articles)
    variants = articles.groupby("product_code").indices
    product_of = articles["product_code"].to_numpy()

    def cum_probs(weights: np.ndarray) -> np.ndarray:
        c = np.cumsum(weights)
        return c / c[-1]

    # Per age band x {before last week, last week} cumulative sampling tables.
    age = customers["age"].to_numpy()
    bands = []
    for g_no, _, (lo, hi) in INDEX_GROUPS:
        boost = np.where(group_of_article == g_no, 4.0, 1.0)
        bands.append((g_no, lo, hi, boost))
    tables: dict[tuple[int, bool], np.ndarray] = {}
    user_band = np.full(n_customers, -1)
    for b, (_, lo, hi, boost) in enumerate(bands):
        in_band = (age >= lo) & (age < hi) & (user_band == -1)
        user_band[in_band] = b
        for late in (False, True):
            w = base * boost * (1.0 if late else ~is_new_item)
            tables[(b, late)] = cum_probs(w)
    for late in (False, True):
        tables[(-1, late)] = cum_probs(base * (1.0 if late else ~is_new_item))

    # Customer activity: log-normal; new customers only buy in the last week.
    activity = rng.lognormal(0.0, 1.0, n_customers)
    is_new_customer = rng.random(n_customers) < new_customer_frac

    # Baskets in chronological order, so "recent" history really is recent.
    sizes = rng.integers(1, 5, n_transactions)
    n_baskets = int(np.searchsorted(np.cumsum(sizes), n_transactions)) + 1
    basket_day = np.sort(rng.integers(0, len(days), n_baskets))
    old_users = np.flatnonzero(~is_new_customer)
    early_user = rng.choice(old_users, n_baskets, p=activity[old_users] / activity[old_users].sum())
    late_user = rng.choice(n_customers, n_baskets, p=activity / activity.sum())
    basket_user = np.where(basket_day >= last_week_start, late_user, early_user)
    basket_channel = np.where(rng.random(n_baskets) < 0.7, 2, 1)

    rows: list[tuple[int, int, int, int]] = []  # (day, customer, article, channel)
    history: dict[int, list[int]] = {}
    baskets = zip(basket_day, basket_user, basket_channel, sizes[:n_baskets], strict=True)
    for day, user, channel, size in baskets:
        late = day >= last_week_start
        own = history.setdefault(int(user), [])
        for _ in range(size):
            r = rng.random()
            item = -1
            if own and r < 0.2:  # repurchase, biased to recent items
                item = own[-min(int(rng.geometric(0.4)), len(own))]
            elif own and r < 0.45:  # colour variant or partner of a recent item
                anchor = own[-min(int(rng.geometric(0.5)), len(own))]
                if rng.random() < 0.5:
                    item = int(rng.choice(variants[product_of[anchor]]))
                else:
                    item = int(partner[anchor])
                if is_new_item[item] and not late:
                    item = -1  # not released yet: fall back to a popularity draw
            if item < 0:
                table = tables[(int(user_band[user]), bool(late))]
                item = int(np.searchsorted(table, rng.random(), side="right"))
            rows.append((int(day), int(user), item, int(channel)))
            own.append(item)
    rows = rows[:n_transactions]

    day_idx, user_idx, item_idx, channel = (np.array(c) for c in zip(*rows, strict=True))
    tx = pd.DataFrame(
        {
            "t_dat": days[day_idx],
            "customer_id": customers["customer_id"].to_numpy()[user_idx],
            "article_id": articles["article_id"].to_numpy()[item_idx],
            "price": price[item_idx] * rng.uniform(0.8, 1.0, len(rows)).astype(np.float32),
            "sales_channel_id": channel,
        }
    ).sort_values(["t_dat", "customer_id", "article_id"], kind="stable")
    return HMData(tx, articles, customers).coerced(), hex_ids


def generate(**kwargs) -> HMData:
    """Generate an H&M-shaped dataset with exactly ``n_transactions`` rows.

    Keyword arguments: ``n_customers`` (2,000), ``n_articles`` (1,000),
    ``n_transactions`` (30,000), ``n_weeks`` (20), ``end_date`` ("2020-09-22"),
    ``new_item_frac`` (0.03), ``new_customer_frac`` (0.05), ``seed`` (42).
    """
    return _generate(**kwargs)[0]


def generate_with_hex_ids(**kwargs) -> tuple[HMData, dict[int, str]]:
    """``generate`` plus the raw hex id behind each hashed customer id."""
    data, hex_ids = _generate(**kwargs)
    return data, dict(zip(hash_customer_ids(hex_ids).tolist(), hex_ids, strict=True))


def to_raw_csvs(data: HMData, hex_ids: dict[int, str], out_dir: Path) -> dict[str, Path]:
    """Write ``data`` in the raw competition CSV format (for converter tests)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tx = data.transactions.copy()
    tx["t_dat"] = tx["t_dat"].dt.strftime("%Y-%m-%d")
    tx["customer_id"] = tx["customer_id"].map(hex_ids)
    tx["article_id"] = tx["article_id"].map("{:010d}".format)
    customers = data.customers.copy()
    customers["customer_id"] = customers["customer_id"].map(hex_ids)
    articles = data.articles.copy()
    articles["article_id"] = articles["article_id"].map("{:010d}".format)
    articles["product_code"] = articles["product_code"].map("{:07d}".format)
    paths = {
        "transactions_train.csv": tx,
        "articles.csv": articles,
        "customers.csv": customers,
    }
    for name, df in paths.items():
        df.to_csv(out_dir / name, index=False)
    return {name: out_dir / name for name in paths}

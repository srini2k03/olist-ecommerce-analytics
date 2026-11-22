"""Phase 4 stage: cold users (strategy chosen on validation) and new articles
(attribute-only retrieval), evaluated separately from warm users."""

from __future__ import annotations

import logging
from dataclasses import replace

import pandas as pd

from recsys.baselines import PopularityRecommender
from recsys.coldstart.items import new_item_metrics, users_buying_new_items
from recsys.coldstart.users import ColdUserRecommender
from recsys.context import Context, records
from recsys.data.folds import Fold
from recsys.evaluate.metrics import evaluate, per_user_metrics
from recsys.ranking.run import best_epochs
from recsys.retrieval.retriever import TwoTowerRetriever
from recsys.retrieval.run import retrieval_config

log = logging.getLogger(__name__)

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EVAL_KS = (12, 50)


def cold_users(fold: Fold) -> list[int]:
    """Target users with no purchase before the fold's start."""
    seen = set(fold.history["customer_id"].unique())
    return [u for u in fold.users if u not in seen]


def _subset(truth: dict, users: list[int]) -> dict:
    return {u: truth[u] for u in users}


def choose_alpha(ctx: Context, fold: Fold) -> tuple[float, list[dict]]:
    users = cold_users(fold)
    truth = _subset(fold.truth, users)
    grid = []
    for alpha in ALPHAS:
        model = ColdUserRecommender(ctx.data.customers, alpha=alpha).fit(fold.history)
        pu = per_user_metrics(model.recommend(users, 12), truth, 12)
        grid.append(
            {
                "alpha": alpha,
                "val_map@12": float(pu["map"].mean()) if len(pu) else 0.0,
                "val_recall@12": float(pu["recall"].mean()) if len(pu) else 0.0,
                "cold_users": len(users),
            }
        )
    best = max(grid, key=lambda r: (r["val_map@12"], r["val_recall@12"], r["alpha"]))
    log.info("cold users: alpha %.2f (validation MAP@12 %.4f)", best["alpha"], best["val_map@12"])
    return best["alpha"], grid


def choose_cold_path(ctx: Context, fold: Fold, alpha: float) -> dict:
    """Serve cold users from the blend or from the ranker (whose candidates include
    age-bucket popularity), whichever has the higher MAP@12 on validation cold users."""
    users = cold_users(fold)
    truth = _subset(fold.truth, users)
    ranker = ctx.load_recs("validation").get("two_tower_ranker")
    blend = ColdUserRecommender(ctx.data.customers, alpha=alpha).fit(fold.history)
    blend_map = float(per_user_metrics(blend.recommend(users, 12), truth, 12)["map"].mean() or 0)
    if ranker is None or not users:
        return {"choice": "cold_blend", "val_map@12_blend": blend_map, "val_map@12_ranker": None}
    ranker_map = float(per_user_metrics(ranker, truth, 12)["map"].mean())
    choice = "cold_blend" if blend_map > ranker_map else "ranker"
    log.info(
        "cold users on validation: blend %.4f vs ranker %.4f -> %s", blend_map, ranker_map, choice
    )
    return {"choice": choice, "val_map@12_blend": blend_map, "val_map@12_ranker": ranker_map}


def cold_item_retrievers(ctx: Context, fold: Fold) -> dict[str, TwoTowerRetriever]:
    """The test-fold retriever (item-id dropout on) and an otherwise identical one
    trained without item-id dropout, whose OOV embedding is never trained."""
    data = ctx.data
    epochs = best_epochs(ctx)
    saved = ctx.artifacts_dir / "folds" / "test" / "retriever"
    cfg = retrieval_config(ctx)
    if (saved / "retriever.json").exists():
        with_dropout = TwoTowerRetriever.load(saved, data.articles, data.customers, fold.history)
    else:
        with_dropout = TwoTowerRetriever(data.articles, data.customers, cfg)
        with_dropout.fit(fold.history, epochs=epochs)
    no_dropout = TwoTowerRetriever(
        data.articles, data.customers, replace(cfg, item_id_dropout=0.0)
    ).fit(fold.history, epochs=epochs)
    return {"attribute_path": with_dropout, "no_id_dropout": no_dropout}


def run_coldstart(ctx: Context) -> pd.DataFrame:
    data = ctx.data
    val, test = ctx.folds["validation"], ctx.folds["test"]
    alpha, grid = choose_alpha(ctx, val)

    # ---- users: cold vs warm on the test week
    truth = test.truth
    cold = cold_users(test)
    warm = [u for u in test.users if u not in set(cold)]
    item_counts = test.history.groupby("article_id").size()
    saved = ctx.load_recs("test")
    strategies_cold = {
        "popularity_1w": PopularityRecommender(1).fit(test.history),
        "age_popularity": ColdUserRecommender(data.customers, alpha=1.0).fit(test.history),
        "trending": ColdUserRecommender(data.customers, alpha=0.0).fit(test.history),
        "cold_blend": ColdUserRecommender(data.customers, alpha=alpha).fit(test.history),
    }
    tables = []

    def add(segment: str, name: str, recs: dict, users: list[int]) -> None:
        if not users:
            return
        t = evaluate(
            recs, _subset(truth, users), EVAL_KS, item_counts, n_boot=ctx.n_boot, seed=ctx.seed
        )
        tables.append(t.assign(segment=segment, model=name))

    blend = strategies_cold["cold_blend"]
    for name, model in strategies_cold.items():
        add("cold_users", name, model.recommend(cold, max(EVAL_KS)), cold)
    for name in ("two_tower_ranker", "repurchase", "popularity_1w"):
        if name in saved:
            add("warm_users", name, saved[name], warm)
            add("cold_users", f"{name} (no cold-start path)", saved[name], cold)
    cold_path = choose_cold_path(ctx, val, alpha)
    if "two_tower_ranker" in saved:
        hybrid = dict(saved["two_tower_ranker"])
        if cold_path["choice"] == "cold_blend":
            hybrid.update(blend.recommend(cold, max(EVAL_KS)))
        ctx.save_recs("test", "ranker_with_cold_start", hybrid)
        add("all_users", "ranker_with_cold_start", hybrid, test.users)
        add("all_users", "two_tower_ranker", saved["two_tower_ranker"], test.users)
    users_table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    # ---- articles: new in the test week, retrievable only through the attribute path
    seen = set(test.history["article_id"].unique())
    new_items = {i for rel in truth.values() for i in rel} - seen
    buyers = users_buying_new_items(truth, new_items)
    item_rows = []
    for variant, retriever in cold_item_retrievers(ctx, test).items():
        for include_unseen in (False, True):
            if variant == "no_id_dropout" and not include_unseen:
                continue
            name = f"{variant}{'+unseen' if include_unseen else ' (seen only)'}"
            items = retriever.index_items(include_unseen=include_unseen)
            ids, _ = retriever.topk(test.users, max(EVAL_KS), test.history, items)
            recs = {u: [int(i) for i in row] for u, row in zip(test.users, ids, strict=True)}
            overall = evaluate(recs, truth, (12,), n_boot=ctx.n_boot, seed=ctx.seed)
            row = {
                "model": name,
                "index_size": len(items),
                "recall@12_all_users": float(overall.query("metric == 'recall'").value.iloc[0]),
                "map@12_all_users": float(overall.query("metric == 'map'").value.iloc[0]),
            }
            for k in EVAL_KS:
                m = new_item_metrics(recs, truth, new_items, k)
                row[f"new_item_recall@{k}"] = m["new_item_recall"]
                row[f"new_item_share@{k}"] = m["new_item_share"]
            if buyers:
                slice_recall = per_user_metrics(recs, _subset(truth, buyers), 50)["recall"]
                row["recall@50_new_item_buyers"] = float(slice_recall.mean())
            item_rows.append(row)
            log.info("%s: new-item recall@50 %.4f", name, row["new_item_recall@50"])
    items_table = pd.DataFrame(item_rows)

    ctx.write_result(
        "coldstart",
        {
            "cold_users": {
                "test_users": len(test.users),
                "cold_test_users": len(cold),
                "cold_share": len(cold) / max(len(test.users), 1),
                "alpha_grid_validation": grid,
                "chosen_alpha": alpha,
                "cold_path_validation": cold_path,
                "metrics": records(users_table),
            },
            "new_items": {
                "new_articles_bought": len(new_items),
                "new_item_purchases": int(sum(len(r & new_items) for r in truth.values())),
                "users_buying_new_items": len(buyers),
                "unseen_catalogue_articles": int(len(set(data.articles["article_id"]) - seen)),
                "variants": records(items_table),
            },
        },
    )
    return users_table

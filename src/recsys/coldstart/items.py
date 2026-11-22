"""New articles: retrieval through the item tower's attribute-only (OOV id) path."""

from __future__ import annotations

from collections.abc import Mapping


def new_item_metrics(
    recs: Mapping[int, list[int]],
    truth: Mapping[int, set[int]],
    new_items: set[int],
    k: int,
) -> dict[str, float]:
    """Recall of new-article purchases in the top ``k`` and the share of new articles shown.

    ``new_item_recall`` = retrieved (user, new article) purchases / all (user, new
    article) purchases; ``new_item_share`` = new articles among all recommended slots.
    """
    hits = total = shown_new = shown = 0
    for u, rel in truth.items():
        top = list(recs.get(u, []))[:k]
        rel_new = rel & new_items
        total += len(rel_new)
        hits += len(rel_new & set(top))
        shown += len(top)
        shown_new += sum(i in new_items for i in top)
    return {
        "k": k,
        "new_item_purchases": total,
        "new_item_recall": hits / total if total else float("nan"),
        "new_item_share": shown_new / shown if shown else 0.0,
    }


def users_buying_new_items(truth: Mapping[int, set[int]], new_items: set[int]) -> list[int]:
    return sorted(u for u, rel in truth.items() if rel & new_items)

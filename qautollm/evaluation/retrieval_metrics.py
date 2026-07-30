"""Recall@k, nDCG@k, MRR against gold evidence chunk ids."""

from __future__ import annotations

import math
from collections.abc import Iterable


def recall_at_k(retrieved_ids: list[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & gold) / len(gold)


def mrr(retrieved_ids: list[str], gold_ids: Iterable[str]) -> float:
    gold = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    dcg = sum(
        (1.0 if cid in gold else 0.0) / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved_ids[:k], start=1)
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

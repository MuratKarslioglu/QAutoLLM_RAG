"""Dense / hybrid retrieval with ``dense_weight`` (Brief §7's rag_v2.yaml).

No extra hyperparameters beyond what ``rag_v2.yaml`` declares: hybrid
combination fetches exactly ``top_k`` candidates from each of dense/sparse
(no separate, undeclared candidate-pool size) before min-max normalizing
and blending by ``dense_weight``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qautollm.rag.indexing import DenseIndex, SparseIndex, search_dense, search_sparse


@dataclass(frozen=True)
class RetrievedDoc:
    chunk_id: str
    score: float


def _minmax_normalize(pairs: list[tuple[str, float]]) -> dict[str, float]:
    if not pairs:
        return {}
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {cid: 1.0 for cid, _ in pairs}
    return {cid: (s - lo) / (hi - lo) for cid, s in pairs}


def _combine(
    dense_hits: list[tuple[str, float]],
    sparse_hits: list[tuple[str, float]],
    dense_weight: float,
) -> dict[str, float]:
    dense_norm = _minmax_normalize(dense_hits)
    sparse_norm = _minmax_normalize(sparse_hits)
    all_ids = sorted(set(dense_norm) | set(sparse_norm))  # sorted: deterministic iteration
    return {
        cid: (
            dense_weight * dense_norm.get(cid, 0.0)
            + (1 - dense_weight) * sparse_norm.get(cid, 0.0)
        )
        for cid in all_ids
    }


def retrieve(
    query_vector: np.ndarray,
    query_text: str,
    dense_index: DenseIndex,
    sparse_index: SparseIndex,
    top_k: int,
    hybrid_on: bool,
    dense_weight: float,
) -> list[RetrievedDoc]:
    dense_hits = search_dense(dense_index, query_vector, top_k)
    if not hybrid_on:
        return [RetrievedDoc(chunk_id=cid, score=score) for cid, score in dense_hits[:top_k]]

    sparse_hits = search_sparse(sparse_index, query_text, top_k)
    combined = _combine(dense_hits, sparse_hits, dense_weight)
    ranked = sorted(combined.items(), key=lambda kv: -kv[1])[:top_k]
    return [RetrievedDoc(chunk_id=cid, score=score) for cid, score in ranked]

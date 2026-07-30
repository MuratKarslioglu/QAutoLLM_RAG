"""FAISS dense + BM25 sparse index build/load (Brief D2: indexes live on
disk/RAM, never VRAM — ``faiss-cpu`` only).

Saved to plain files rather than pickled as a single blob: FAISS's own
serialization (``faiss.write_index``/``read_index``) is the only reliably
version-stable way to persist an index, and ``rank_bm25`` objects are
pure-Python and pickle cleanly on their own.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

_DENSE_INDEX_FILE = "dense.faiss"
_DENSE_IDS_FILE = "chunk_ids.json"
_SPARSE_FILE = "bm25.pkl"


@dataclass
class DenseIndex:
    index: faiss.Index
    chunk_ids: list[str]


@dataclass
class SparseIndex:
    bm25: BM25Okapi
    chunk_ids: list[str]


def build_dense_index(vectors: np.ndarray, chunk_ids: list[str]) -> DenseIndex:
    """Build a flat inner-product index. Embeddings must be L2-normalized
    beforehand (Brief: embedding.py normalizes) so inner product == cosine.
    """
    if vectors.shape[0] != len(chunk_ids):
        raise ValueError(f"{vectors.shape[0]} vectors but {len(chunk_ids)} chunk_ids")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return DenseIndex(index=index, chunk_ids=list(chunk_ids))


def search_dense(
    index: DenseIndex, query_vector: np.ndarray, top_k: int
) -> list[tuple[str, float]]:
    query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
    scores, positions = index.index.search(query, min(top_k, len(index.chunk_ids)))
    return [
        (index.chunk_ids[pos], float(score))
        for score, pos in zip(scores[0], positions[0], strict=True)
        if pos != -1
    ]


def save_dense_index(index: DenseIndex, dest: Path) -> None:
    faiss.write_index(index.index, str(dest / _DENSE_INDEX_FILE))
    (dest / _DENSE_IDS_FILE).write_text(json.dumps(index.chunk_ids))


def load_dense_index(dest: Path) -> DenseIndex:
    idx = faiss.read_index(str(dest / _DENSE_INDEX_FILE))
    chunk_ids = json.loads((dest / _DENSE_IDS_FILE).read_text())
    return DenseIndex(index=idx, chunk_ids=chunk_ids)


def build_sparse_index(chunks: list[str], chunk_ids: list[str]) -> SparseIndex:
    if len(chunks) != len(chunk_ids):
        raise ValueError(f"{len(chunks)} chunks but {len(chunk_ids)} chunk_ids")
    tokenized = [c.split() for c in chunks]
    return SparseIndex(bm25=BM25Okapi(tokenized), chunk_ids=list(chunk_ids))


def search_sparse(index: SparseIndex, query: str, top_k: int) -> list[tuple[str, float]]:
    scores = index.bm25.get_scores(query.split())
    ranked = sorted(zip(index.chunk_ids, scores, strict=True), key=lambda pair: -pair[1])
    return [(cid, float(s)) for cid, s in ranked[:top_k]]


def save_sparse_index(index: SparseIndex, dest: Path) -> None:
    with open(dest / _SPARSE_FILE, "wb") as f:
        pickle.dump(index, f)


def load_sparse_index(dest: Path) -> SparseIndex:
    with open(dest / _SPARSE_FILE, "rb") as f:
        return pickle.load(f)

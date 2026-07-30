import numpy as np
import pytest

from qautollm.rag.indexing import (
    build_dense_index,
    build_sparse_index,
    load_dense_index,
    load_sparse_index,
    save_dense_index,
    save_sparse_index,
    search_dense,
    search_sparse,
)
from qautollm.rag.retrieval import retrieve

CHUNK_IDS = ["c0", "c1", "c2", "c3"]
CHUNK_TEXTS = [
    "cats are small domestic animals",
    "dogs are loyal domestic animals",
    "quantum computers use qubits",
    "bayesian optimization uses surrogate models",
]


def _synthetic_vectors() -> np.ndarray:
    # deterministic, well-separated unit vectors so nearest-neighbor is unambiguous
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(CHUNK_IDS), 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def test_dense_index_returns_nearest_first():
    vectors = _synthetic_vectors()
    index = build_dense_index(vectors, CHUNK_IDS)
    hits = search_dense(index, vectors[2], top_k=2)
    assert hits[0][0] == "c2"  # querying with a stored vector must retrieve itself first


def test_dense_index_rejects_mismatched_lengths():
    vectors = _synthetic_vectors()
    with pytest.raises(ValueError):
        build_dense_index(vectors, CHUNK_IDS[:-1])


def test_sparse_index_favors_keyword_overlap():
    index = build_sparse_index(CHUNK_TEXTS, CHUNK_IDS)
    hits = search_sparse(index, "quantum qubits", top_k=1)
    assert hits[0][0] == "c2"


def test_dense_index_roundtrips_through_disk(tmp_path):
    vectors = _synthetic_vectors()
    index = build_dense_index(vectors, CHUNK_IDS)
    save_dense_index(index, tmp_path)
    loaded = load_dense_index(tmp_path)
    assert loaded.chunk_ids == CHUNK_IDS
    hits_before = search_dense(index, vectors[0], top_k=4)
    hits_after = search_dense(loaded, vectors[0], top_k=4)
    assert hits_before == hits_after


def test_sparse_index_roundtrips_through_disk(tmp_path):
    index = build_sparse_index(CHUNK_TEXTS, CHUNK_IDS)
    save_sparse_index(index, tmp_path)
    loaded = load_sparse_index(tmp_path)
    before = search_sparse(index, "domestic animals", 4)
    after = search_sparse(loaded, "domestic animals", 4)
    assert before == after


def test_retrieve_dense_only_matches_search_dense():
    vectors = _synthetic_vectors()
    dense = build_dense_index(vectors, CHUNK_IDS)
    sparse = build_sparse_index(CHUNK_TEXTS, CHUNK_IDS)
    docs = retrieve(
        query_vector=vectors[1],
        query_text="dogs",
        dense_index=dense,
        sparse_index=sparse,
        top_k=2,
        hybrid_on=False,
        dense_weight=0.5,
    )
    assert docs[0].chunk_id == "c1"


def test_retrieve_hybrid_is_deterministic():
    vectors = _synthetic_vectors()
    dense = build_dense_index(vectors, CHUNK_IDS)
    sparse = build_sparse_index(CHUNK_TEXTS, CHUNK_IDS)
    kwargs = dict(
        query_vector=vectors[3],
        query_text="bayesian surrogate models",
        dense_index=dense,
        sparse_index=sparse,
        top_k=3,
        hybrid_on=True,
        dense_weight=0.7,
    )
    a = retrieve(**kwargs)
    b = retrieve(**kwargs)
    assert a == b
    assert a[0].chunk_id == "c3"

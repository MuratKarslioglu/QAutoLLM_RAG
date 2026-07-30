from pathlib import Path

import pytest

from qautollm.space.search_space import SearchSpace, SearchSpaceError
from qautollm.utils.seeds import make_rng

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"


@pytest.fixture(scope="module")
def space() -> SearchSpace:
    return SearchSpace.from_yaml(RAG_V2_PATH)


def test_loads_expected_groups_and_variable_count(space: SearchSpace):
    assert space.group_order == (
        "G1_chunking",
        "G2_embedding",
        "G3_retrieval",
        "G4_reranking",
        "G5_context",
    )
    assert {v.name for v in space.variables} == {
        "chunking_strategy",
        "chunk_size",
        "chunk_overlap",
        "embedding_model",
        "top_k",
        "hybrid_on",
        "dense_weight",
        "reranker_on",
        "reranker_top_n",
        "context_budget",
    }


def test_sample_is_always_valid(space: SearchSpace):
    rng = make_rng(0)
    for _ in range(500):
        raw = space.sample(rng)
        space.validate(raw)  # must not raise


def test_sample_respects_conditionals(space: SearchSpace):
    rng = make_rng(1)
    saw_hybrid_off_without_dense_weight = False
    saw_reranker_off_without_top_n = False
    for _ in range(500):
        raw = space.sample(rng)
        if raw["hybrid_on"] == 0:
            assert "dense_weight" not in raw
            saw_hybrid_off_without_dense_weight = True
        else:
            assert "dense_weight" in raw
        if raw["reranker_on"] == 0:
            assert "reranker_top_n" not in raw
            saw_reranker_off_without_top_n = True
        else:
            assert "reranker_top_n" in raw
    assert saw_hybrid_off_without_dense_weight
    assert saw_reranker_off_without_top_n


def test_validate_rejects_unknown_parameter(space: SearchSpace):
    raw = space.sample(make_rng(2))
    raw["not_a_real_param"] = 1
    with pytest.raises(SearchSpaceError):
        space.validate(raw)


def test_validate_rejects_out_of_domain_value(space: SearchSpace):
    raw = space.sample(make_rng(3))
    raw["top_k"] = 999
    with pytest.raises(SearchSpaceError):
        space.validate(raw)


def test_validate_rejects_conditional_violation(space: SearchSpace):
    raw = space.sample(make_rng(4))
    raw["hybrid_on"] = 0
    raw["dense_weight"] = 0.5  # present despite hybrid_on == 0
    with pytest.raises(SearchSpaceError):
        space.validate(raw)


def test_validate_rejects_missing_required_parameter(space: SearchSpace):
    raw = space.sample(make_rng(5))
    del raw["chunk_size"]
    with pytest.raises(SearchSpaceError):
        space.validate(raw)

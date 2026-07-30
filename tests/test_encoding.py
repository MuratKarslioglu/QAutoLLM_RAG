from pathlib import Path

import numpy as np
import pytest

from qautollm.space.encoding import decode, encode
from qautollm.space.search_space import SearchSpace, SearchSpaceError
from qautollm.utils.seeds import make_rng

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"


@pytest.fixture(scope="module")
def space() -> SearchSpace:
    return SearchSpace.from_yaml(RAG_V2_PATH)


def test_round_trip_on_many_random_valid_configs(space: SearchSpace):
    rng = make_rng(42)
    for _ in range(1000):
        raw = space.sample(rng)
        config = encode(raw, space)
        assert decode(config, space) == raw


def test_encode_is_deterministic(space: SearchSpace):
    raw = space.sample(make_rng(7))
    a = encode(raw, space)
    b = encode(raw, space)
    assert a.config_hash == b.config_hash
    assert np.array_equal(a.z, b.z)
    assert np.array_equal(a.mask, b.mask)
    assert a == b
    assert hash(a) == hash(b)


def test_different_configs_get_different_hashes(space: SearchSpace):
    rng = make_rng(9)
    seen_hashes = set()
    for _ in range(200):
        raw = space.sample(rng)
        config = encode(raw, space)
        seen_hashes.add(config.config_hash)
    # not a strict guarantee for arbitrary spaces, but this space is small
    # enough and the sample size large enough that collisions would indicate
    # a hashing bug, not bad luck.
    assert len(seen_hashes) > 1


def test_invalid_combo_rejected_at_construction(space: SearchSpace):
    raw = space.sample(make_rng(11))
    raw["hybrid_on"] = 0
    raw["dense_weight"] = 0.5  # present despite hybrid_on == 0
    with pytest.raises(SearchSpaceError):
        encode(raw, space)


def test_inactive_conditional_dims_get_sentinel(space: SearchSpace):
    rng = make_rng(13)
    for _ in range(200):
        raw = space.sample(rng)
        config = encode(raw, space)
        g3_slice = config.z[config.group_slices["G3_retrieval"]]
        g4_slice = config.z[config.group_slices["G4_reranking"]]
        if raw["hybrid_on"] == 0:
            assert g3_slice[-1] == pytest.approx(0.5)
            assert config.mask[space.group_order.index("G3_retrieval")] == 0.0
        else:
            assert config.mask[space.group_order.index("G3_retrieval")] == 1.0
        if raw["reranker_on"] == 0:
            assert g4_slice[-1] == pytest.approx(0.5)
            assert config.mask[space.group_order.index("G4_reranking")] == 0.0
        else:
            assert config.mask[space.group_order.index("G4_reranking")] == 1.0


def test_groups_without_conditionals_are_always_active(space: SearchSpace):
    rng = make_rng(17)
    raw = space.sample(rng)
    config = encode(raw, space)
    for group_name in ("G1_chunking", "G2_embedding", "G5_context"):
        assert config.mask[space.group_order.index(group_name)] == 1.0


def test_group_slices_are_contiguous_and_cover_z(space: SearchSpace):
    raw = space.sample(make_rng(19))
    config = encode(raw, space)
    slices = [config.group_slices[g] for g in space.group_order]
    cursor = 0
    for s in slices:
        assert s.start == cursor
        cursor = s.stop
    assert cursor == len(config.z)

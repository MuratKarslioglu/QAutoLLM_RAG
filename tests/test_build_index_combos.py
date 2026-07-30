from scripts.build_index import CHUNK_OVERLAPS, CHUNK_SIZES, CHUNKING_STRATEGIES, _chunking_combos


def test_combos_only_include_overlap_smaller_than_size():
    combos = _chunking_combos()
    assert all(overlap < size for _, size, overlap in combos)


def test_combos_cover_every_strategy():
    combos = _chunking_combos()
    assert {strategy for strategy, _, _ in combos} == set(CHUNKING_STRATEGIES)


def test_combos_are_deterministic_and_unique():
    a = _chunking_combos()
    b = _chunking_combos()
    assert a == b
    assert len(a) == len(set(a))


def test_combo_count_matches_expected_upper_bound():
    combos = _chunking_combos()
    max_possible = len(CHUNKING_STRATEGIES) * len(CHUNK_SIZES) * len(CHUNK_OVERLAPS)
    assert 0 < len(combos) <= max_possible

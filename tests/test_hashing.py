from qautollm.store.hashing import canonical_hash, canonical_json, pair_hash


def test_canonical_hash_is_key_order_independent():
    a = canonical_hash({"x": 1, "y": 2})
    b = canonical_hash({"y": 2, "x": 1})
    assert a == b


def test_canonical_hash_distinguishes_values():
    assert canonical_hash({"x": 1}) != canonical_hash({"x": 2})


def test_canonical_json_is_deterministic():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_pair_hash_is_symmetric():
    assert pair_hash("h1", "h2") == pair_hash("h2", "h1")


def test_pair_hash_distinguishes_pairs():
    assert pair_hash("h1", "h2") != pair_hash("h1", "h3")

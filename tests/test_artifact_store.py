from qautollm.store.artifact_store import ArtifactStore
from qautollm.store.db import ArtifactDB


def _store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts", ArtifactDB(tmp_path / "db.sqlite"))


def test_db_level_miss_then_hit(tmp_path):
    store = _store(tmp_path)
    calls = []

    def compute():
        calls.append(1)
        return {"value": 42}

    a = store.get_or_compute("L4", {"q": 1}, compute)
    b = store.get_or_compute("L4", {"q": 1}, compute)

    assert a.hit is False
    assert b.hit is True
    assert a.value == b.value == {"value": 42}
    assert len(calls) == 1  # compute_fn only ran once


def test_db_level_hit_reuses_cold_latency_not_lookup_time(tmp_path):
    store = _store(tmp_path)

    def slow_compute():
        return "result"

    a = store.get_or_compute("L4", {"q": 2}, slow_compute)
    b = store.get_or_compute("L4", {"q": 2}, slow_compute)
    assert a.latency_s == b.latency_s  # hit must reuse the stored cold latency exactly


def test_file_level_miss_then_hit(tmp_path):
    store = _store(tmp_path)
    calls = []

    def compute(dest):
        calls.append(1)
        (dest / "data.txt").write_text("hello")

    a = store.get_or_compute("L1", {"corpus_id": "toy"}, compute)
    b = store.get_or_compute("L1", {"corpus_id": "toy"}, compute)

    assert a.hit is False
    assert b.hit is True
    assert a.value == b.value  # same destination directory
    assert (a.value / "data.txt").read_text() == "hello"
    assert len(calls) == 1


def test_different_key_fields_get_different_cache_entries(tmp_path):
    store = _store(tmp_path)
    a = store.get_or_compute("L4", {"q": 1}, lambda: "a")
    b = store.get_or_compute("L4", {"q": 2}, lambda: "b")
    assert a.value != b.value
    assert a.key != b.key


def test_report_tracks_hits_and_misses_per_level(tmp_path):
    store = _store(tmp_path)
    store.get_or_compute("L4", {"q": 1}, lambda: 1)  # miss
    store.get_or_compute("L4", {"q": 1}, lambda: 1)  # hit
    store.get_or_compute("L4", {"q": 1}, lambda: 1)  # hit
    store.get_or_compute("L3", {"q": 1}, lambda: 1)  # miss

    report = store.report()
    assert report["L4"] == {"hits": 2, "misses": 1, "hit_rate": 2 / 3}
    assert report["L3"] == {"hits": 0, "misses": 1, "hit_rate": 0.0}


def test_unknown_level_rejected(tmp_path):
    import pytest

    store = _store(tmp_path)
    with pytest.raises(ValueError, match="unknown artifact level"):
        store.get_or_compute("L99", {}, lambda: None)

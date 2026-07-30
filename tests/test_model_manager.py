from qautollm.core.model_manager import ModelManager


def test_load_large_constructs_once():
    manager = ModelManager()
    calls = []

    def factory():
        calls.append(1)
        return object()

    a = manager.load_large("gen", factory)
    b = manager.load_large("gen", factory)
    assert a is b
    assert len(calls) == 1


def test_load_large_unloads_previous_different_model():
    manager = ModelManager()
    manager.load_large("embedder", lambda: "embedder-instance")
    assert manager.resident_large == "embedder"

    manager.load_large("generator", lambda: "generator-instance")
    assert manager.resident_large == "generator"  # D1: only one large model at a time


def test_unload_large_clears_state():
    manager = ModelManager()
    manager.load_large("gen", lambda: object())
    manager.unload_large()
    assert manager.resident_large is None


def test_load_small_persists_across_large_model_swaps():
    manager = ModelManager()
    reranker = manager.load_small("reranker", lambda: "reranker-instance")
    manager.load_large("embedder", lambda: "embedder-instance")
    manager.load_large("generator", lambda: "generator-instance")
    assert manager.load_small("reranker", lambda: "should-not-be-called") is reranker

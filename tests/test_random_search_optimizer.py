from pathlib import Path

from qautollm.optimizers.random_search import RandomSearchOptimizer
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"
_SPACE = SearchSpace.from_yaml(RAG_V2_PATH)


def _init_design(n=3, seed=1):
    return [encode(raw, _SPACE) for raw in _SPACE.sobol_design(n, seed=seed)]


def test_suggest_drains_init_design_first_and_in_order():
    init = _init_design(n=3)
    opt = RandomSearchOptimizer(_SPACE, seed=42, init_design=init)
    for expected in init:
        assert opt.suggest() == expected


def test_suggest_after_init_design_is_valid():
    opt = RandomSearchOptimizer(_SPACE, seed=42, init_design=[])
    for _ in range(20):
        cfg = opt.suggest()
        _SPACE.validate(cfg.raw)


def test_resume_reproduces_identical_continuation():
    init = _init_design(n=2)
    opt_a = RandomSearchOptimizer(_SPACE, seed=42, init_design=init)
    for _ in range(2):
        opt_a.suggest()  # drain init design
    first_five_a = [opt_a.suggest() for _ in range(5)]

    opt_b = RandomSearchOptimizer(_SPACE, seed=42, init_design=init)
    for _ in range(2):
        opt_b.suggest()
    checkpoint = opt_b.state_dict()

    opt_c = RandomSearchOptimizer(_SPACE, seed=42, init_design=init)
    opt_c.load_state_dict(checkpoint)
    first_five_c = [opt_c.suggest() for _ in range(5)]

    assert first_five_a == first_five_c

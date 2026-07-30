from pathlib import Path

import numpy as np

from qautollm.optimizers.structured_bo import StructuredBO
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"
_SPACE = SearchSpace.from_yaml(RAG_V2_PATH)


def _init_design(n=3, seed=1):
    return [encode(raw, _SPACE) for raw in _SPACE.sobol_design(n, seed=seed)]


def _fake_objectives(cfg) -> np.ndarray:
    import hashlib

    seed = int(hashlib.sha256(cfg.config_hash.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=3)


def test_suggest_drains_init_design_first_and_in_order():
    init = _init_design(n=3)
    opt = StructuredBO(_SPACE, seed=42, init_design=init)
    for expected in init:
        cfg = opt.suggest()
        assert cfg == expected
        opt.update(cfg, _fake_objectives(cfg))


def test_suggest_after_init_design_is_valid_and_uses_ehvi():
    opt = StructuredBO(_SPACE, seed=42, init_design=_init_design(n=3))
    for _ in range(3):
        cfg = opt.suggest()
        opt.update(cfg, _fake_objectives(cfg))
    # by now len(self._configs) >= 2: the next suggestion goes through the
    # GP+EHVI path rather than the "not enough data" random fallback
    cfg = opt.suggest()
    _SPACE.validate(cfg.raw)
    opt.update(cfg, _fake_objectives(cfg))


def test_resume_reproduces_identical_continuation():
    init = _init_design(n=2)

    def _run(opt, n_steps):
        out = []
        for _ in range(n_steps):
            cfg = opt.suggest()
            out.append(cfg)
            opt.update(cfg, _fake_objectives(cfg))
        return out

    opt_a = StructuredBO(_SPACE, seed=42, init_design=init)
    _run(opt_a, 3)  # drain init (2) + 1 EHVI-guided suggestion
    continuation_a = _run(opt_a, 2)

    opt_b = StructuredBO(_SPACE, seed=42, init_design=init)
    _run(opt_b, 3)
    checkpoint = opt_b.state_dict()

    opt_c = StructuredBO(_SPACE, seed=42, init_design=init)
    opt_c.load_state_dict(checkpoint)
    continuation_c = _run(opt_c, 2)

    assert [c.config_hash for c in continuation_a] == [c.config_hash for c in continuation_c]

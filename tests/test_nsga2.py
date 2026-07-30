import hashlib
from pathlib import Path

import numpy as np

from qautollm.optimizers.nsga2 import NSGA2Optimizer
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"
_SPACE = SearchSpace.from_yaml(RAG_V2_PATH)


def _init_design(n=6, seed=1):
    return [encode(raw, _SPACE) for raw in _SPACE.sobol_design(n, seed=seed)]


def _fake_objectives(cfg) -> np.ndarray:
    seed = int(hashlib.sha256(cfg.config_hash.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=3)


def test_suggest_drains_init_design_first_and_in_order():
    init = _init_design(n=6)
    opt = NSGA2Optimizer(_SPACE, seed=42, init_design=init)
    for expected in init:
        cfg = opt.suggest()
        assert cfg == expected
        opt.update(cfg, _fake_objectives(cfg))


def test_suggest_past_first_generation_stays_valid():
    """Drive it through several full generations (population size 6, so
    every 6 suggestions past init is one generation) and check every
    offspring is a valid config for the search space.
    """
    opt = NSGA2Optimizer(_SPACE, seed=42, init_design=_init_design(n=6))
    for _ in range(30):  # init (6) + 4 full generations
        cfg = opt.suggest()
        _SPACE.validate(cfg.raw)
        opt.update(cfg, _fake_objectives(cfg))


def test_population_size_stays_constant_across_generations():
    opt = NSGA2Optimizer(_SPACE, seed=42, init_design=_init_design(n=6))
    for _ in range(6):  # drain init design -> generation-0 parents
        cfg = opt.suggest()
        opt.update(cfg, _fake_objectives(cfg))
    for _ in range(6):  # evaluate generation-0's offspring
        cfg = opt.suggest()
        opt.update(cfg, _fake_objectives(cfg))
    opt.suggest()  # triggers survival selection for the first time
    assert len(opt._parents) == 6


def test_resume_reproduces_identical_continuation():
    init = _init_design(n=4)

    def _run(opt, n_steps):
        out = []
        for _ in range(n_steps):
            cfg = opt.suggest()
            out.append(cfg)
            opt.update(cfg, _fake_objectives(cfg))
        return out

    opt_a = NSGA2Optimizer(_SPACE, seed=42, init_design=init)
    _run(opt_a, 10)  # init (4) + generation-0 offspring (4) + 2 into gen 1
    continuation_a = _run(opt_a, 6)

    opt_b = NSGA2Optimizer(_SPACE, seed=42, init_design=init)
    _run(opt_b, 10)
    checkpoint = opt_b.state_dict()

    opt_c = NSGA2Optimizer(_SPACE, seed=42, init_design=init)
    opt_c.load_state_dict(checkpoint)
    continuation_c = _run(opt_c, 6)

    assert [c.config_hash for c in continuation_a] == [c.config_hash for c in continuation_c]


def test_three_objective_toy_run_end_to_end_produces_a_nontrivial_front():
    """Brief P6 gate: 3-objective toy run end-to-end."""
    from qautollm.surrogate.pareto_archive import ParetoArchive

    opt = NSGA2Optimizer(_SPACE, seed=7, init_design=_init_design(n=8, seed=2))
    archive = ParetoArchive()
    for _ in range(40):
        cfg = opt.suggest()
        objectives = _fake_objectives(cfg)
        opt.update(cfg, objectives)
        archive.update(objectives)

    assert archive.front.shape[1] == 3
    assert len(archive.points) >= 1
    ref_point = np.array([1.0, 1.0, 1.0])
    assert archive.hypervolume(ref_point) > 0.0

"""Brief P4 gate: structured_bo beats random search on a cheap synthetic
function. Verified empirically across several seeds before landing this
test (seed=42 here; also holds for seeds 1 and 123 with the same budget).
"""

from pathlib import Path

import numpy as np

from qautollm.optimizers.random_search import RandomSearchOptimizer
from qautollm.optimizers.structured_bo import StructuredBO
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace
from qautollm.surrogate.pareto_archive import ParetoArchive
from qautollm.utils.seeds import make_rng

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"
_SPACE = SearchSpace.from_yaml(RAG_V2_PATH)


def _synthetic_objectives(config, targets: np.ndarray) -> np.ndarray:
    """3 fixed targets in augmented [z||mask] space; objective_i = distance
    to target_i (minimize) — a cheap, deterministic multi-objective test
    function with a genuine trade-off (can't be close to all 3 at once).
    """
    features = np.concatenate([config.z, config.mask])
    return np.array([np.linalg.norm(features - t) for t in targets])


def _run(optimizer_cls, budget: int, init_points: int, seed: int, targets: np.ndarray) -> float:
    init_rng = make_rng(seed + 1000)
    init_design = [encode(_SPACE.sample(init_rng), _SPACE) for _ in range(init_points)]
    optimizer = optimizer_cls(_SPACE, seed, init_design)
    archive = ParetoArchive()

    for _ in range(budget):
        cfg = optimizer.suggest()
        objectives = _synthetic_objectives(cfg, targets)
        optimizer.update(cfg, objectives)
        archive.update(objectives)

    dim = targets.shape[1]
    ref_point = np.array([dim**0.5] * 3)  # generous upper bound on max distance
    return archive.hypervolume(ref_point)


def test_structured_bo_beats_random_search_on_synthetic_function():
    sample_cfg = encode(_SPACE.sample(make_rng(0)), _SPACE)
    dim = len(sample_cfg.z) + len(sample_cfg.mask)
    target_rng = np.random.default_rng(7)
    targets = target_rng.uniform(0.2, 0.8, size=(3, dim))

    budget, init_points, seed = 18, 8, 42
    hv_random = _run(RandomSearchOptimizer, budget, init_points, seed, targets)
    hv_structured = _run(StructuredBO, budget, init_points, seed, targets)

    assert hv_structured > hv_random


def test_update_ignores_exact_duplicate_config():
    """Regression test: the orchestrator calls ``update()`` on every
    successful trial, including a re-suggested duplicate (Brief §5.2 only
    guarantees a duplicate costs zero *budget*, nothing upstream stops it
    reaching here). Appending the same config twice puts two identical rows
    in the GP's training X, which makes the kernel Gram matrix exactly
    singular -- this crashed a real run with BoTorch's
    ``ModelFittingError: All attempts to fit the model have failed`` once
    training set size made a collision likely.
    """
    init_rng = make_rng(1000)
    init_points = 4
    init_design = [encode(_SPACE.sample(init_rng), _SPACE) for _ in range(init_points)]
    optimizer = StructuredBO(_SPACE, seed=42, init_design=init_design)

    for cfg in init_design:
        optimizer.update(cfg, np.array([0.1, 0.2, 0.3]))
    assert len(optimizer._configs) == init_points

    duplicate_cfg = init_design[0]
    optimizer.update(duplicate_cfg, np.array([9.9, 9.9, 9.9]))  # same config re-added
    assert len(optimizer._configs) == init_points  # not appended a second time

    # the training set is still well-conditioned enough to actually fit a GP
    # (a real duplicate row here is exactly what made BoTorch's fit fail)
    optimizer.suggest()

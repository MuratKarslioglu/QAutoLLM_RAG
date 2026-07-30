from pathlib import Path

import pytest

from qautollm.space.search_space import SearchSpace

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"


@pytest.fixture(scope="module")
def space() -> SearchSpace:
    return SearchSpace.from_yaml(RAG_V2_PATH)


def test_sobol_design_produces_valid_configs(space: SearchSpace):
    design = space.sobol_design(10, seed=42)
    assert len(design) == 10
    for raw in design:
        space.validate(raw)  # must not raise


def test_sobol_design_is_deterministic_given_seed(space: SearchSpace):
    a = space.sobol_design(10, seed=42)
    b = space.sobol_design(10, seed=42)
    assert a == b


def test_sobol_design_differs_across_seeds(space: SearchSpace):
    a = space.sobol_design(10, seed=42)
    b = space.sobol_design(10, seed=7)
    assert a != b

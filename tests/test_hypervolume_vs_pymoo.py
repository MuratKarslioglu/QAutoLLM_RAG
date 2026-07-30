"""Brief P6 gate: hypervolume validated against pymoo reference values.

``ParetoArchive.hypervolume`` delegates to BoTorch's ``Hypervolume`` utility
(Brief §5.4's surrogate/pareto_archive.py) — this cross-checks that
implementation against pymoo's independent ``HV`` indicator (both in
minimization convention) across hand-picked and randomized point sets.
"""

import numpy as np
import pytest
from pymoo.indicators.hv import HV

from qautollm.surrogate.pareto_archive import ParetoArchive


def _hv_via_archive(points: np.ndarray, ref_point: np.ndarray) -> float:
    archive = ParetoArchive()
    for p in points:
        archive.update(p)
    return archive.hypervolume(ref_point)


def _hv_via_pymoo(points: np.ndarray, ref_point: np.ndarray) -> float:
    return float(HV(ref_point=ref_point)(points))


def test_single_point_2d():
    points = np.array([[1.0, 1.0]])
    ref = np.array([2.0, 2.0])
    assert _hv_via_archive(points, ref) == pytest.approx(_hv_via_pymoo(points, ref))


def test_two_mutually_nondominated_points_2d():
    points = np.array([[0.0, 1.0], [1.0, 0.0]])
    ref = np.array([2.0, 2.0])
    assert _hv_via_archive(points, ref) == pytest.approx(_hv_via_pymoo(points, ref))


def test_dominated_point_contributes_nothing():
    points = np.array([[0.5, 0.5], [1.0, 1.0]])  # second point dominated by first
    ref = np.array([2.0, 2.0])
    assert _hv_via_archive(points, ref) == pytest.approx(_hv_via_pymoo(points, ref))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_2d_fronts_match_pymoo(seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.1, 0.9, size=(15, 2))
    ref = np.array([1.0, 1.0])
    assert _hv_via_archive(points, ref) == pytest.approx(_hv_via_pymoo(points, ref), rel=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_3d_fronts_match_pymoo(seed):
    """3-objective case is the one that actually matters for this project
    (Brief §5.5's fixed 3-objective vector)."""
    rng = np.random.default_rng(seed + 100)
    points = rng.uniform(0.1, 0.9, size=(20, 3))
    ref = np.array([1.0, 1.0, 1.0])
    assert _hv_via_archive(points, ref) == pytest.approx(_hv_via_pymoo(points, ref), rel=1e-6)


def test_only_the_nondominated_front_matters():
    """ParetoArchive already prunes dominated points on insertion; feeding
    pymoo the *same* dominated-inclusive point set must still agree, since
    pymoo's HV also filters to the non-dominated front internally (nds=True
    by default).
    """
    rng = np.random.default_rng(42)
    points = rng.uniform(0.1, 0.9, size=(30, 3))
    ref = np.array([1.0, 1.0, 1.0])
    assert _hv_via_archive(points, ref) == pytest.approx(_hv_via_pymoo(points, ref), rel=1e-6)

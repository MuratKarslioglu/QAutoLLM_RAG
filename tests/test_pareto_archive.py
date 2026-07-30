import numpy as np
import pytest

from qautollm.surrogate.pareto_archive import ParetoArchive, dominates


def test_dominates_strict_improvement_in_one_objective():
    assert dominates(np.array([0.0, 0.0]), np.array([0.0, 1.0])) is True
    assert dominates(np.array([1.0, 1.0]), np.array([0.0, 0.0])) is False


def test_dominates_requires_no_worse_in_any_objective():
    assert dominates(np.array([0.0, 1.0]), np.array([1.0, 0.0])) is False


def test_dominates_identical_points_false():
    assert dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0])) is False


def test_update_adds_first_point():
    archive = ParetoArchive()
    assert archive.update(np.array([1.0, 1.0])) is True
    assert archive.front.shape == (1, 2)


def test_update_rejects_dominated_point():
    archive = ParetoArchive()
    archive.update(np.array([0.0, 0.0]))
    assert archive.update(np.array([1.0, 1.0])) is False
    assert archive.front.shape == (1, 2)


def test_update_prunes_points_it_dominates():
    archive = ParetoArchive()
    archive.update(np.array([1.0, 3.0]))
    archive.update(np.array([3.0, 1.0]))
    assert archive.front.shape == (2, 2)  # mutually non-dominated so far
    assert archive.update(np.array([0.5, 0.5])) is True  # dominates both
    assert archive.front.shape == (1, 2)
    assert np.allclose(archive.front[0], [0.5, 0.5])


def test_update_keeps_mutually_non_dominated_points():
    archive = ParetoArchive()
    archive.update(np.array([0.0, 1.0]))
    archive.update(np.array([1.0, 0.0]))
    assert archive.front.shape == (2, 2)


def test_hypervolume_empty_archive_is_zero():
    assert ParetoArchive().hypervolume(np.array([10.0, 10.0])) == 0.0


def test_hypervolume_single_point():
    archive = ParetoArchive()
    archive.update(np.array([1.0, 1.0]))
    # minimize convention: ref point (2,2), point (1,1) -> dominated volume = 1*1 = 1
    assert archive.hypervolume(np.array([2.0, 2.0])) == pytest.approx(1.0)


def test_hypervolume_increases_as_front_improves():
    archive = ParetoArchive()
    ref = np.array([10.0, 10.0])
    archive.update(np.array([5.0, 5.0]))
    hv1 = archive.hypervolume(ref)
    archive.update(np.array([3.0, 3.0]))
    hv2 = archive.hypervolume(ref)
    assert hv2 > hv1

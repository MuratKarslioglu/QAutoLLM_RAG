import pytest

from qautollm.space.encoding import encode
from qautollm.synth.benchmark import build_synthetic_objectives, make_qubit_sweep_space
from qautollm.utils.seeds import make_rng


def test_qubit_count_one_uses_binary_not_categorical():
    space = make_qubit_sweep_space(n_groups=2, qubits_per_group=1)
    for v in space.variables:
        assert v.kind == "binary"
        assert len(v.values) == 2  # a binary IS the 1-qubit 2-way choice


@pytest.mark.parametrize("qubits", [2, 3, 4, 5])
def test_qubit_count_n_uses_categorical_width_n(qubits):
    space = make_qubit_sweep_space(n_groups=2, qubits_per_group=qubits)
    for v in space.variables:
        assert v.kind == "categorical"
        assert len(v.values) == qubits


def test_group_z_slice_width_matches_qubit_count():
    for qubits in (1, 2, 3, 4, 5):
        space = make_qubit_sweep_space(n_groups=3, qubits_per_group=qubits)
        cfg = encode(space.sample(make_rng(0)), space)
        for g in space.group_order:
            sl = cfg.group_slices[g]
            assert sl.stop - sl.start == qubits


def test_rejects_qubit_count_out_of_range():
    with pytest.raises(ValueError, match="qubits_per_group"):
        make_qubit_sweep_space(n_groups=2, qubits_per_group=0)
    with pytest.raises(ValueError, match="qubits_per_group"):
        make_qubit_sweep_space(n_groups=2, qubits_per_group=6)


def test_rejects_invalid_n_groups():
    with pytest.raises(ValueError, match="n_groups"):
        make_qubit_sweep_space(n_groups=0, qubits_per_group=3)


def test_sampled_configs_are_valid_at_every_qubit_count():
    rng = make_rng(1)
    for qubits in (1, 2, 3, 4, 5):
        space = make_qubit_sweep_space(n_groups=3, qubits_per_group=qubits)
        for _ in range(20):
            space.validate(space.sample(rng))


def test_objectives_work_on_qubit_sweep_space():
    space = make_qubit_sweep_space(n_groups=3, qubits_per_group=4)
    fn = build_synthetic_objectives(space, interaction=0.5, seed=0)
    cfg = encode(space.sample(make_rng(0)), space)
    assert fn(cfg).shape == (3,)


def test_max_qubits_override_allows_beyond_brief_d3_cap():
    # scripts/run_qubit_sweep_extended.py's escape hatch; default (5) is
    # unaffected -- still rejects 6 with no override.
    with pytest.raises(ValueError, match="qubits_per_group"):
        make_qubit_sweep_space(n_groups=2, qubits_per_group=10)
    space = make_qubit_sweep_space(n_groups=2, qubits_per_group=10, max_qubits=20)
    for v in space.variables:
        assert v.kind == "categorical"
        assert len(v.values) == 10

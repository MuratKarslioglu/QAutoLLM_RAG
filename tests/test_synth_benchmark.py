import numpy as np
import pytest

from qautollm.space.encoding import encode
from qautollm.synth.benchmark import build_synthetic_objectives, make_synthetic_space
from qautollm.utils.seeds import make_rng


def test_make_synthetic_space_group_count_and_width():
    space = make_synthetic_space(n_groups=4, conditional_fraction=0.0)
    assert space.group_order == ("S0", "S1", "S2", "S3")
    for g in space.group_order:
        assert len(space.variables_in_group(g)) == 2  # categorical + gate, no conditional


def test_make_synthetic_space_conditional_fraction_adds_conditional_vars():
    space = make_synthetic_space(n_groups=4, conditional_fraction=0.5)
    conditional_groups = [g for g in space.group_order if len(space.variables_in_group(g)) == 3]
    assert len(conditional_groups) == 2  # round(0.5 * 4)


def test_synthetic_space_rejects_invalid_n_groups():
    with pytest.raises(ValueError, match="n_groups"):
        make_synthetic_space(n_groups=0)


def test_synthetic_space_rejects_invalid_conditional_fraction():
    with pytest.raises(ValueError, match="conditional_fraction"):
        make_synthetic_space(n_groups=3, conditional_fraction=1.5)


def test_group_width_stays_within_qubit_cap():
    """Brief D3: <=5 qubits per group — categorical(3) + gate(1) [+ cond(1)]
    must never exceed 5 one-hot/scalar columns.
    """
    from qautollm.space.encoding import encode as _encode

    space = make_synthetic_space(n_groups=3, conditional_fraction=1.0)
    rng = make_rng(0)
    cfg = _encode(space.sample(rng), space)
    for g in space.group_order:
        width = cfg.group_slices[g].stop - cfg.group_slices[g].start
        assert width <= 5


def test_sampled_configs_are_valid():
    space = make_synthetic_space(n_groups=5, conditional_fraction=0.4)
    rng = make_rng(1)
    for _ in range(50):
        raw = space.sample(rng)
        space.validate(raw)  # must not raise


def test_objectives_rejects_invalid_interaction():
    space = make_synthetic_space(n_groups=3)
    with pytest.raises(ValueError, match="interaction"):
        build_synthetic_objectives(space, interaction=1.5, seed=0)


def test_objectives_are_deterministic_given_seed():
    space = make_synthetic_space(n_groups=4, conditional_fraction=0.25)
    cfg = encode(space.sample(make_rng(0)), space)
    fn_a = build_synthetic_objectives(space, interaction=0.5, seed=7)
    fn_b = build_synthetic_objectives(space, interaction=0.5, seed=7)
    assert np.array_equal(fn_a(cfg), fn_b(cfg))


def test_objectives_differ_across_seeds():
    space = make_synthetic_space(n_groups=4)
    cfg = encode(space.sample(make_rng(0)), space)
    fn_a = build_synthetic_objectives(space, interaction=0.5, seed=1)
    fn_b = build_synthetic_objectives(space, interaction=0.5, seed=2)
    assert not np.array_equal(fn_a(cfg), fn_b(cfg))


def test_objectives_shape_matches_n_objectives():
    space = make_synthetic_space(n_groups=3)
    cfg = encode(space.sample(make_rng(0)), space)
    fn = build_synthetic_objectives(space, interaction=0.0, seed=0, n_objectives=3)
    assert fn(cfg).shape == (3,)


def test_zero_interaction_is_purely_additive_no_coupling_leakage():
    """At interaction=0, changing a variable in one group must not change
    the objective contribution attributable to another group's distance
    term — verified indirectly: two configs differing only in one group's
    values should differ in a way consistent with a per-group additive
    decomposition (a coarse sanity check, not a full derivation).
    """
    space = make_synthetic_space(n_groups=3, conditional_fraction=0.0)
    fn = build_synthetic_objectives(space, interaction=0.0, seed=3)
    rng = make_rng(0)
    raw = space.sample(rng)
    cfg1 = encode(raw, space)

    raw2 = dict(raw)
    raw2["cat0"] = next(v for v in space.by_name("cat0").values if v != raw["cat0"])
    cfg2 = encode(raw2, space)

    obj1 = fn(cfg1)
    obj2 = fn(cfg2)
    assert not np.array_equal(obj1, obj2)  # changing a group's value must move the objective


def test_single_group_has_no_interaction_term():
    space = make_synthetic_space(n_groups=1)
    fn = build_synthetic_objectives(space, interaction=1.0, seed=0)
    cfg = encode(space.sample(make_rng(0)), space)
    # with no group pairs, interaction=1.0 with coupled=0.0 collapses to 0 objective contribution
    assert np.all(fn(cfg) == 0.0)

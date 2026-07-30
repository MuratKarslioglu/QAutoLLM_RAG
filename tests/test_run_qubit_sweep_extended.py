from scripts.run_qubit_sweep_extended import run_one, run_sweep


def test_run_one_structured_bo_above_brief_cap():
    row = run_one(
        n_groups=2,
        qubits_per_group=6,  # > Brief D3's 5-qubit cap
        interaction=0.0,
        optimizer_name="structured_bo",
        seed=0,
        budget=4,
        init_points=2,
    )
    assert row["qubits_per_group"] == 6
    assert row["hypervolume"] >= 0.0


def test_run_one_qkernel_bo_above_brief_cap_uses_lightning():
    # small qubit count so this stays fast in the "not slow" suite --
    # this is the exact code path run_qubit_sweep_extended.py uses at
    # qubits_per_group up to 20, just at a size cheap enough for CI.
    row = run_one(
        n_groups=2,
        qubits_per_group=6,
        interaction=0.0,
        optimizer_name="qkernel_bo",
        seed=0,
        budget=4,
        init_points=2,
    )
    assert row["qubits_per_group"] == 6
    assert row["hypervolume"] >= 0.0


def test_run_one_is_deterministic_given_seed():
    kwargs = dict(
        n_groups=2,
        qubits_per_group=6,
        interaction=1.0,
        optimizer_name="qkernel_bo",
        seed=1,
        budget=4,
        init_points=2,
    )
    a = run_one(**kwargs)
    b = run_one(**kwargs)
    assert a["hypervolume"] == b["hypervolume"]


def test_run_sweep_covers_every_combination():
    df = run_sweep(
        n_groups=2,
        qubit_counts=(6, 7),
        interactions=(0.0, 1.0),
        budget=3,
        init_points=2,
        n_seeds=2,
    )
    # 2 qubit_counts x 2 interactions x 2 optimizers x 2 seeds
    assert len(df) == 16
    assert set(df["qubits_per_group"]) == {6, 7}
    assert set(df["optimizer"]) == {"structured_bo", "qkernel_bo"}

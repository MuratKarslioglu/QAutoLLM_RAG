from pathlib import Path

from scripts.run_depth_sweep import make_plots, run_one, run_sweep


def test_run_one_returns_expected_fields():
    row = run_one(
        n_groups=3,
        qubits_per_group=3,
        interaction=0.0,
        depth=2,
        optimizer_name="qkernel_bo",
        seed=0,
        budget=5,
        init_points=3,
    )
    assert row["qubits_per_group"] == 3
    assert row["depth"] == 2
    assert row["optimizer"] == "qkernel_bo"
    assert row["hypervolume"] >= 0.0


def test_run_one_is_deterministic_given_seed():
    kwargs = dict(
        n_groups=3,
        qubits_per_group=3,
        interaction=1.0,
        depth=3,
        optimizer_name="qkernel_bo",
        seed=1,
        budget=5,
        init_points=3,
    )
    a = run_one(**kwargs)
    b = run_one(**kwargs)
    assert a["hypervolume"] == b["hypervolume"]


def test_run_one_covers_depths_1_to_4():
    for depth in (1, 2, 3, 4):
        row = run_one(
            n_groups=2,
            qubits_per_group=3,
            interaction=0.0,
            depth=depth,
            optimizer_name="qkernel_bo",
            seed=0,
            budget=4,
            init_points=2,
        )
        assert row["hypervolume"] >= 0.0


def test_run_one_structured_bo_ignores_depth_value():
    # structured_bo has no feature_map_depth analog -- same seed/space
    # must give the same result regardless of which depth it's paired with.
    kwargs = dict(
        n_groups=2,
        qubits_per_group=3,
        interaction=0.0,
        optimizer_name="structured_bo",
        seed=0,
        budget=4,
        init_points=2,
    )
    a = run_one(depth=1, **kwargs)
    b = run_one(depth=4, **kwargs)
    assert a["hypervolume"] == b["hypervolume"]


def test_run_sweep_covers_every_combination():
    df = run_sweep(
        n_groups=2,
        qubit_counts=(3, 5),
        interactions=(0.0,),
        depths=(1, 2),
        optimizer_names=("structured_bo", "qkernel_bo"),
        budget=4,
        init_points=2,
        n_seeds=2,
    )
    # 2 qubit_counts x 1 interaction x 2 depths x 2 optimizers x 2 seeds
    assert len(df) == 16
    assert set(df["depth"]) == {1, 2}
    assert set(df["optimizer"]) == {"structured_bo", "qkernel_bo"}


def test_make_plots_writes_png(tmp_path: Path):
    df = run_sweep(
        n_groups=2,
        qubit_counts=(3,),
        interactions=(0.0,),
        depths=(1, 2),
        optimizer_names=("qkernel_bo",),
        budget=3,
        init_points=2,
        n_seeds=1,
    )
    make_plots(df, tmp_path)
    assert (tmp_path / "depth_sweep.png").exists()

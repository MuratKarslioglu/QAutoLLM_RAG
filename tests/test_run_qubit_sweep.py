from pathlib import Path

from scripts.run_qubit_sweep import make_plots, run_one, run_sweep


def test_run_one_returns_expected_fields():
    row = run_one(
        n_groups=3,
        qubits_per_group=2,
        interaction=0.0,
        optimizer_name="structured_bo",
        seed=0,
        budget=5,
        init_points=3,
    )
    assert row["qubits_per_group"] == 2
    assert row["optimizer"] == "structured_bo"
    assert row["hypervolume"] >= 0.0


def test_run_one_is_deterministic_given_seed():
    kwargs = dict(
        n_groups=3,
        qubits_per_group=1,
        interaction=1.0,
        optimizer_name="qkernel_bo",
        seed=1,
        budget=5,
        init_points=3,
    )
    a = run_one(**kwargs)
    b = run_one(**kwargs)
    assert a["hypervolume"] == b["hypervolume"]


def test_run_one_covers_every_qubit_count_1_to_5():
    for qubits in (1, 2, 3, 4, 5):
        row = run_one(
            n_groups=2,
            qubits_per_group=qubits,
            interaction=0.0,
            optimizer_name="structured_bo",
            seed=0,
            budget=4,
            init_points=2,
        )
        assert row["hypervolume"] >= 0.0


def test_run_sweep_covers_every_combination():
    df = run_sweep(
        n_groups=3,
        qubit_counts=(1, 3),
        interactions=(0.0, 1.0),
        optimizer_names=("structured_bo", "qkernel_bo"),
        budget=4,
        init_points=2,
        n_seeds=2,
    )
    # 2 qubit_counts x 2 interactions x 2 optimizers x 2 seeds
    assert len(df) == 16
    assert set(df["qubits_per_group"]) == {1, 3}
    assert set(df["optimizer"]) == {"structured_bo", "qkernel_bo"}


def test_make_plots_writes_png(tmp_path: Path):
    df = run_sweep(
        n_groups=3,
        qubit_counts=(1, 3),
        interactions=(0.0, 1.0),
        optimizer_names=("structured_bo",),
        budget=3,
        init_points=2,
        n_seeds=1,
    )
    make_plots(df, tmp_path)
    assert (tmp_path / "qubit_sweep.png").exists()

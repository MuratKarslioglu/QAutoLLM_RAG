from pathlib import Path

from scripts.run_depth_qubit_grid import make_plots, run_one, run_sweep


def test_run_one_returns_expected_fields():
    row = run_one(
        n_groups=2,
        qubits_per_group=6,  # > Brief D3's 5-qubit cap
        interaction=0.0,
        depth=2,
        optimizer_name="qkernel_bo",
        seed=0,
        budget=4,
        init_points=2,
        n_candidates=8,
        mc_samples=4,
    )
    assert row["qubits_per_group"] == 6
    assert row["depth"] == 2
    assert row["hypervolume"] >= 0.0
    assert row["failed"] is False


def test_run_one_depth_zero_is_degenerate_but_runs():
    # depth=0: no encoding layers, the circuit's output never depends on
    # the input -- a deliberate control point, not expected to be useful,
    # but must not crash.
    row = run_one(
        n_groups=2,
        qubits_per_group=6,
        interaction=0.0,
        depth=0,
        optimizer_name="qkernel_bo",
        seed=0,
        budget=3,
        init_points=2,
        n_candidates=8,
        mc_samples=4,
    )
    assert row["hypervolume"] >= 0.0


def test_run_one_is_deterministic_given_seed():
    kwargs = dict(
        n_groups=2,
        qubits_per_group=6,
        interaction=1.0,
        depth=3,
        optimizer_name="qkernel_bo",
        seed=1,
        budget=4,
        init_points=2,
        n_candidates=8,
        mc_samples=4,
    )
    a = run_one(**kwargs)
    b = run_one(**kwargs)
    assert a["hypervolume"] == b["hypervolume"]


def test_run_sweep_covers_every_combination():
    df = run_sweep(
        n_groups=2,
        qubit_counts=(6, 7),
        depths=(0, 2),
        interactions=(0.0,),
        budget=3,
        init_points=2,
        n_seeds=1,
        n_candidates=8,
        mc_samples=4,
    )
    # 2 qubit_counts x 2 depths x 1 interaction x 2 optimizers x 1 seed
    assert len(df) == 8
    assert set(df["depth"]) == {0, 2}
    assert set(df["optimizer"]) == {"structured_bo", "qkernel_bo"}
    assert not df["failed"].any()


def test_make_plots_writes_png(tmp_path: Path):
    df = run_sweep(
        n_groups=2,
        qubit_counts=(6,),
        depths=(0, 2),
        interactions=(0.0,),
        budget=3,
        init_points=2,
        n_seeds=1,
        n_candidates=8,
        mc_samples=4,
    )
    make_plots(df, tmp_path)
    assert (tmp_path / "depth_qubit_grid.png").exists()

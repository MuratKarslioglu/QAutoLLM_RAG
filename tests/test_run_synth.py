from pathlib import Path

import pandas as pd

from scripts.run_synth import make_plots, run_one, run_sweep


def test_run_one_returns_expected_fields():
    row = run_one(
        n_groups=3,
        conditional_fraction=0.0,
        interaction=0.0,
        optimizer_name="random",
        seed=0,
        budget=5,
        init_points=3,
    )
    assert row["n_groups"] == 3
    assert row["optimizer"] == "random"
    assert row["hypervolume"] >= 0.0
    assert row["wall_clock_s"] >= 0.0


def test_run_one_is_deterministic_given_seed():
    kwargs = dict(
        n_groups=3,
        conditional_fraction=0.0,
        interaction=0.5,
        optimizer_name="tpe",
        seed=1,
        budget=5,
        init_points=3,
    )
    a = run_one(**kwargs)
    b = run_one(**kwargs)
    assert a["hypervolume"] == b["hypervolume"]


def test_run_one_with_kernel_bo_optimizers_on_synthetic_space():
    """Exercises StructuralKernel/GP-fitting against the synthetic space's
    own group structure (different widths/dims than rag_v2) — a genuinely
    different code path from every other kernel_bo test in this suite.
    """
    for name in ("structured_bo", "qkernel_bo"):
        row = run_one(
            n_groups=3,
            conditional_fraction=0.0,
            interaction=0.0,
            optimizer_name=name,
            seed=0,
            budget=6,
            init_points=3,
        )
        assert row["hypervolume"] >= 0.0


def test_run_sweep_covers_every_combination():
    df = run_sweep(
        dims=(3,),
        conditionalities=(0.0,),
        interactions=(0.0, 1.0),
        optimizer_names=("random", "nsga2"),
        budget=4,
        init_points=2,
        n_seeds=2,
    )
    # 1 dim x 1 conditionality x 2 interactions x 2 optimizers x 2 seeds
    assert len(df) == 8
    assert set(df["optimizer"]) == {"random", "nsga2"}
    assert set(df["interaction"]) == {0.0, 1.0}


def test_make_plots_writes_png(tmp_path: Path):
    df = run_sweep(
        dims=(3, 4),
        conditionalities=(0.0,),
        interactions=(0.0, 1.0),
        optimizer_names=("random",),
        budget=3,
        init_points=2,
        n_seeds=1,
    )
    make_plots(df, tmp_path)
    assert (tmp_path / "layer_a_hypervolume_sweep.png").exists()


def test_sweep_result_is_csv_serializable_and_roundtrips(tmp_path: Path):
    df = run_sweep(
        dims=(3,),
        conditionalities=(0.0,),
        interactions=(0.0,),
        optimizer_names=("random",),
        budget=3,
        init_points=2,
        n_seeds=1,
    )
    path = tmp_path / "sweep.csv"
    df.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(df)

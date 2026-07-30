from pathlib import Path

from scripts.run_ablation import run_ablation


def test_run_ablation_end_to_end_small_scale(tmp_path: Path):
    """Brief P8 gate (scaled down for test speed): one call runs every
    requested optimizer on the toy dataset and produces the main table +
    convergence plot.
    """
    table = run_ablation(
        budget=6,
        init_points=3,
        seed=1,
        output_dir=tmp_path,
        optimizer_names=("random", "nsga2"),
    )

    assert set(table["optimizer"]) == {"random", "nsga2"}
    assert (table["n_ok"] >= 6).all()

    assert (tmp_path / "main_table.md").exists()
    assert (tmp_path / "main_table.csv").exists()
    assert (tmp_path / "convergence.png").exists()
    assert (tmp_path / "ablation_random" / "db.sqlite").exists()
    assert (tmp_path / "ablation_nsga2" / "db.sqlite").exists()

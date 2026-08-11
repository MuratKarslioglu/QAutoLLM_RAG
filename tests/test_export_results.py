from pathlib import Path

import pandas as pd
import pytest

from qautollm.store.db import ArtifactDB
from scripts.export_results import (
    build_main_table,
    build_seed_aggregated_table,
    compare_optimizers_paired,
    export_table,
    load_trials,
    make_convergence_plot,
    make_seed_variance_plot,
)


def _row(experiment_id, optimizer_name, trial_index, f1, faithfulness, latency, status="ok"):
    return {
        "experiment_id": experiment_id,
        "optimizer_name": optimizer_name,
        "seed": 0,
        "dataset_id": "toy",
        "trial_index": trial_index,
        "config_hash": f"hash{trial_index}",
        "config_json": "{}",
        "objective_f1": -f1,
        "objective_faithfulness": -faithfulness,
        "objective_latency_s": latency,
        "em": f1,
        "recall_at_k": f1,
        "ndcg": f1,
        "tokens_prompt": 10,
        "tokens_generated": 5,
        "peak_vram_mb": 0.0,
        "cache_hits_json": "{}",
        "duplicate_flag": 0,
        "wall_clock_s": 0.01,
        "status": status,
        "timestamp": "2026-01-01T00:00:00",
    }


def _sample_trials() -> pd.DataFrame:
    rows = [
        _row("exp_a", "random", 0, 0.5, 0.5, 0.1),
        _row("exp_a", "random", 1, 0.6, 0.6, 0.09),
        _row("exp_a", "random", 2, 0.4, 0.4, 0.2, status="failed"),
        _row("exp_b", "structured_bo", 0, 0.7, 0.7, 0.08),
        _row("exp_b", "structured_bo", 1, 0.8, 0.8, 0.07),
    ]
    return pd.DataFrame(rows)


def test_build_main_table_groups_by_optimizer():
    table = build_main_table(_sample_trials())
    assert set(table["optimizer"]) == {"random", "structured_bo"}
    random_row = table[table["optimizer"] == "random"].iloc[0]
    assert random_row["n_trials"] == 3
    assert random_row["n_ok"] == 2  # the failed trial is excluded from means
    assert random_row["mean_f1"] == 0.55  # mean of 0.5, 0.6 (failed row excluded)


def test_build_main_table_uses_shared_reference_point_across_optimizers():
    table = build_main_table(_sample_trials())
    # structured_bo's points are strictly better (higher f1/faithfulness);
    # against a shared reference point it must have >= hypervolume than random.
    hv_random = table.loc[table["optimizer"] == "random", "hypervolume"].iloc[0]
    hv_structured = table.loc[table["optimizer"] == "structured_bo", "hypervolume"].iloc[0]
    assert hv_structured >= hv_random


def test_build_main_table_empty_input():
    table = build_main_table(pd.DataFrame())
    assert table.empty
    assert "optimizer" in table.columns


def test_export_table_markdown(tmp_path: Path):
    table = build_main_table(_sample_trials())
    path = tmp_path / "table.md"
    export_table(table, path, fmt="markdown")
    text = path.read_text()
    assert text.startswith("| optimizer")
    assert "random" in text
    assert "structured_bo" in text


def test_export_table_csv(tmp_path: Path):
    table = build_main_table(_sample_trials())
    path = tmp_path / "table.csv"
    export_table(table, path, fmt="csv")
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(table)


def test_export_table_rejects_unknown_format(tmp_path: Path):
    import pytest

    with pytest.raises(ValueError, match="unknown format"):
        export_table(build_main_table(_sample_trials()), tmp_path / "x", fmt="yaml")


def test_make_convergence_plot_writes_png(tmp_path: Path):
    path = tmp_path / "convergence.png"
    make_convergence_plot(_sample_trials(), path)
    assert path.exists()


def test_make_convergence_plot_handles_empty_trials(tmp_path: Path):
    path = tmp_path / "convergence.png"
    make_convergence_plot(pd.DataFrame(), path)
    assert not path.exists()  # nothing to plot, function returns early


def test_load_trials_concatenates_multiple_dbs(tmp_path: Path):
    db_a_path = tmp_path / "a.sqlite"
    db_b_path = tmp_path / "b.sqlite"
    db_a = ArtifactDB(db_a_path)
    db_b = ArtifactDB(db_b_path)
    db_a.write_trial(_row("exp_a", "random", 0, 0.5, 0.5, 0.1))
    db_b.write_trial(_row("exp_b", "structured_bo", 0, 0.7, 0.7, 0.08))

    trials = load_trials([db_a_path, db_b_path])
    assert len(trials) == 2
    assert set(trials["experiment_id"]) == {"exp_a", "exp_b"}


def test_load_trials_handles_no_dbs_gracefully(tmp_path: Path):
    empty_db_path = tmp_path / "empty.sqlite"
    ArtifactDB(empty_db_path)
    trials = load_trials([empty_db_path])
    assert trials.empty


def test_build_seed_aggregated_table_groups_by_optimizer_and_seed():
    rows = [
        _row("exp_a_s0", "random", 0, 0.5, 0.5, 0.1),
        _row("exp_a_s1", "random", 0, 0.7, 0.7, 0.1),
    ]
    rows[1]["seed"] = 1
    table = build_seed_aggregated_table(pd.DataFrame(rows))
    assert list(table["optimizer"]) == ["random"]
    assert table.iloc[0]["n_seeds"] == 2
    assert table.iloc[0]["mean_f1_mean"] == pytest.approx(0.6)
    assert table.iloc[0]["mean_f1_std"] > 0


def test_build_seed_aggregated_table_single_seed_has_zero_std():
    table = build_seed_aggregated_table(_sample_trials())  # all rows seed=0
    assert (table["hypervolume_std"] == 0.0).all()
    assert (table["n_seeds"] == 1).all()


def test_build_seed_aggregated_table_empty_input():
    table = build_seed_aggregated_table(pd.DataFrame())
    assert table.empty
    assert "optimizer" in table.columns


def _multi_seed_rows(optimizer_name: str, f1_by_seed: dict[int, float]) -> list[dict]:
    rows = []
    for seed, f1 in f1_by_seed.items():
        row = _row(f"exp_{optimizer_name}_s{seed}", optimizer_name, 0, f1, f1, 0.1)
        row["seed"] = seed
        rows.append(row)
    return rows


def test_compare_optimizers_paired_needs_at_least_three_seeds():
    rows = _multi_seed_rows("qkernel_bo", {0: 0.5, 1: 0.6}) + _multi_seed_rows(
        "structured_bo", {0: 0.4, 1: 0.5}
    )
    result = compare_optimizers_paired(pd.DataFrame(rows), "qkernel_bo", "structured_bo")
    assert result["n_seeds"] == 2
    assert result["statistic"] is None
    assert result["p_value"] is None


def test_compare_optimizers_paired_reports_a_p_value_with_enough_seeds():
    rows = _multi_seed_rows("qkernel_bo", {0: 0.9, 1: 0.85, 2: 0.95}) + _multi_seed_rows(
        "structured_bo", {0: 0.5, 1: 0.4, 2: 0.45}
    )
    result = compare_optimizers_paired(pd.DataFrame(rows), "qkernel_bo", "structured_bo")
    assert result["n_seeds"] == 3
    assert result["p_value"] is not None
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["mean_diff_a_minus_b"] > 0  # qkernel_bo strictly better here


def test_compare_optimizers_paired_handles_identical_values_without_crashing():
    rows = _multi_seed_rows("qkernel_bo", {0: 0.5, 1: 0.5, 2: 0.5}) + _multi_seed_rows(
        "structured_bo", {0: 0.5, 1: 0.5, 2: 0.5}
    )
    result = compare_optimizers_paired(pd.DataFrame(rows), "qkernel_bo", "structured_bo")
    assert result["p_value"] == 1.0
    assert result["mean_diff_a_minus_b"] == 0.0


def test_compare_optimizers_paired_empty_input():
    result = compare_optimizers_paired(pd.DataFrame(), "qkernel_bo", "structured_bo")
    assert result["n_seeds"] == 0


def test_compare_optimizers_paired_supports_non_hypervolume_metrics():
    # mean_f1 favors qkernel_bo here while hypervolume (driven by the shared
    # 0.1 latency/0.1 faithfulness columns _row() fixes) would not
    # necessarily agree -- the 2026-08-09 real campaign hit exactly this
    # disagreement (qkernel_bo ahead on mean_f1, behind on hypervolume), so
    # the significance test needs to be checkable per metric, not just on
    # hypervolume.
    rows = _multi_seed_rows("qkernel_bo", {0: 0.9, 1: 0.85, 2: 0.95}) + _multi_seed_rows(
        "structured_bo", {0: 0.5, 1: 0.4, 2: 0.45}
    )
    result = compare_optimizers_paired(
        pd.DataFrame(rows), "qkernel_bo", "structured_bo", metric="mean_f1"
    )
    assert result["metric"] == "mean_f1"
    assert result["n_seeds"] == 3
    assert result["mean_diff_a_minus_b"] > 0


def test_compare_optimizers_paired_rejects_unknown_metric():
    rows = _multi_seed_rows("qkernel_bo", {0: 0.9, 1: 0.85, 2: 0.95}) + _multi_seed_rows(
        "structured_bo", {0: 0.5, 1: 0.4, 2: 0.45}
    )
    with pytest.raises(ValueError, match="unsupported metric"):
        compare_optimizers_paired(
            pd.DataFrame(rows), "qkernel_bo", "structured_bo", metric="not_a_metric"
        )


def test_make_seed_variance_plot_writes_png(tmp_path: Path):
    rows = _multi_seed_rows("qkernel_bo", {0: 0.9, 1: 0.85, 2: 0.95}) + _multi_seed_rows(
        "structured_bo", {0: 0.5, 1: 0.4, 2: 0.45}
    )
    path = tmp_path / "hypervolume_by_seed.png"
    make_seed_variance_plot(pd.DataFrame(rows), path)
    assert path.exists()


def test_make_seed_variance_plot_handles_empty_trials(tmp_path: Path):
    path = tmp_path / "hypervolume_by_seed.png"
    make_seed_variance_plot(pd.DataFrame(), path)
    assert not path.exists()

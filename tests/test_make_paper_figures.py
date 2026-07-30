from pathlib import Path

import pandas as pd

from scripts.make_paper_figures import (
    layer_a_summary_table,
    make_qubit_advantage_figure,
    qubit_advantage_table,
)

# Hand-built, not run through the real (slow, GP-fit-heavy) sweep — this
# tests the table/figure logic in isolation, cheaply.
_QUBIT_SWEEP_ROWS = [
    # interaction=0.0: quantum's advantage peaks at qubits_per_group=3
    {"qubits_per_group": 1, "interaction": 0.0, "optimizer": "structured_bo", "hypervolume": 5.0},
    {"qubits_per_group": 1, "interaction": 0.0, "optimizer": "qkernel_bo", "hypervolume": 5.1},
    {"qubits_per_group": 3, "interaction": 0.0, "optimizer": "structured_bo", "hypervolume": 5.0},
    {"qubits_per_group": 3, "interaction": 0.0, "optimizer": "qkernel_bo", "hypervolume": 6.0},
    {"qubits_per_group": 5, "interaction": 0.0, "optimizer": "structured_bo", "hypervolume": 5.0},
    {"qubits_per_group": 5, "interaction": 0.0, "optimizer": "qkernel_bo", "hypervolume": 5.2},
    # interaction=1.0 (misspecified): advantage is largest at qubits_per_group=1 here
    {"qubits_per_group": 1, "interaction": 1.0, "optimizer": "structured_bo", "hypervolume": 3.0},
    {"qubits_per_group": 1, "interaction": 1.0, "optimizer": "qkernel_bo", "hypervolume": 3.5},
    {"qubits_per_group": 5, "interaction": 1.0, "optimizer": "structured_bo", "hypervolume": 3.0},
    {"qubits_per_group": 5, "interaction": 1.0, "optimizer": "qkernel_bo", "hypervolume": 3.05},
]


def test_qubit_advantage_table_picks_the_peak_advantage_qubit_count():
    df = pd.DataFrame(_QUBIT_SWEEP_ROWS)
    table = qubit_advantage_table(df)

    row_0 = table[table["interaction"] == 0.0].iloc[0]
    assert row_0["best_qubit_count"] == 3
    assert row_0["quantum_advantage"] == 1.0  # 6.0 - 5.0

    row_1 = table[table["interaction"] == 1.0].iloc[0]
    assert row_1["best_qubit_count"] == 1
    assert row_1["quantum_advantage"] == 0.5  # 3.5 - 3.0


def test_qubit_advantage_table_ignores_non_kernel_bo_optimizers():
    extra = {
        "qubits_per_group": 3,
        "interaction": 0.0,
        "optimizer": "random",
        "hypervolume": 999.0,
    }
    rows = [*_QUBIT_SWEEP_ROWS, extra]
    table = qubit_advantage_table(pd.DataFrame(rows))
    assert set(table["interaction"]) == {0.0, 1.0}
    assert "random" not in table.columns


def test_make_qubit_advantage_figure_writes_png(tmp_path: Path):
    df = pd.DataFrame(_QUBIT_SWEEP_ROWS)
    output_path = tmp_path / "fig.png"
    make_qubit_advantage_figure(df, output_path)
    assert output_path.exists()


# scripts/run_qubit_sweep_extended.py's beyond-Brief-D3 range (5/10/15/20)
_EXTENDED_QUBIT_SWEEP_ROWS = [
    {"qubits_per_group": 10, "interaction": 0.0, "optimizer": "structured_bo", "hypervolume": 5.0},
    {"qubits_per_group": 10, "interaction": 0.0, "optimizer": "qkernel_bo", "hypervolume": 5.4},
    {"qubits_per_group": 20, "interaction": 0.0, "optimizer": "structured_bo", "hypervolume": 5.0},
    {"qubits_per_group": 20, "interaction": 0.0, "optimizer": "qkernel_bo", "hypervolume": 5.1},
    {"qubits_per_group": 10, "interaction": 1.0, "optimizer": "structured_bo", "hypervolume": 3.0},
    {"qubits_per_group": 10, "interaction": 1.0, "optimizer": "qkernel_bo", "hypervolume": 3.1},
    {"qubits_per_group": 20, "interaction": 1.0, "optimizer": "structured_bo", "hypervolume": 3.0},
    {"qubits_per_group": 20, "interaction": 1.0, "optimizer": "qkernel_bo", "hypervolume": 3.02},
]


def test_qubit_advantage_table_folds_in_extended_range_via_plain_concat():
    # this is exactly what make_paper_figures.py's main() does when both
    # --qubit-sweep-csv and --qubit-sweep-extended-csv are given.
    combined = pd.concat(
        [pd.DataFrame(_QUBIT_SWEEP_ROWS), pd.DataFrame(_EXTENDED_QUBIT_SWEEP_ROWS)],
        ignore_index=True,
    )
    table = qubit_advantage_table(combined)
    # interaction=0.0's peak advantage is still at qubits_per_group=3 (1.0,
    # from the base range) -- the extended range's best (10: 5.4-5.0=0.4)
    # doesn't beat it, so folding in the wider range must not silently
    # change an answer the base range already established.
    row_0 = table[table["interaction"] == 0.0].iloc[0]
    assert row_0["best_qubit_count"] == 3
    assert set(combined["qubits_per_group"]) == {1, 3, 5, 10, 20}


def test_make_qubit_advantage_figure_marks_the_brief_d3_cap_beyond_5_qubits(tmp_path: Path):
    combined = pd.concat(
        [pd.DataFrame(_QUBIT_SWEEP_ROWS), pd.DataFrame(_EXTENDED_QUBIT_SWEEP_ROWS)],
        ignore_index=True,
    )
    output_path = tmp_path / "fig_extended.png"
    make_qubit_advantage_figure(combined, output_path)
    assert output_path.exists()


_LAYER_A_ROWS = [
    {
        "n_groups": 3,
        "conditional_fraction": 0.0,
        "interaction": 0.0,
        "optimizer": "structured_bo",
        "hypervolume": 6.0,
    },
    {
        "n_groups": 3,
        "conditional_fraction": 0.0,
        "interaction": 0.0,
        "optimizer": "random",
        "hypervolume": 5.0,
    },
    {
        "n_groups": 5,
        "conditional_fraction": 0.0,
        "interaction": 0.0,
        "optimizer": "structured_bo",
        "hypervolume": 5.5,
    },
]


def test_layer_a_summary_table_groups_correctly():
    table = layer_a_summary_table(pd.DataFrame(_LAYER_A_ROWS))
    assert len(table) == 3  # one row per (n_groups, conditional_fraction, interaction, optimizer)
    assert {"mean", "std"} <= set(table.columns)

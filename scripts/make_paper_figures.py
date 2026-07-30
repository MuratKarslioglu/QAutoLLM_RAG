#!/usr/bin/env python3
"""Paper-ready tables and figures from the cheap synthetic sweeps.

Consumes the CSVs already produced by ``scripts/run_qubit_sweep.py`` and
``scripts/run_synth.py``. Real (HotpotQA) ablation tables/plots are
produced separately by ``scripts/export_results.py`` — this script is
specifically about the synthetic sweeps, in particular the question of
*at which qubit count the quantum fidelity kernel's advantage over the
classical Matérn kernel is largest*, and whether that depends on the
additive/misspecified (``interaction``) axis.

Also folds in ``scripts/run_qubit_sweep_extended.py``'s beyond-Brief-D3
qubit range (5/10/15/20) when given, via ``--qubit-sweep-extended-csv`` --
same schema as the base sweep, so it's a plain concat before the existing
table/figure code (no separate code path needed).

Usage:
    python -m scripts.make_paper_figures \\
        --qubit-sweep-csv results/qubit_sweep/qubit_sweep_results.csv \\
        --qubit-sweep-extended-csv results/qubit_sweep_extended/qubit_sweep_extended_results.csv \\
        --synth-csv results/synth/sweep_results.csv \\
        --output-dir results/paper
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from qautollm.utils.logging import get_logger

_logger = get_logger(__name__)


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def qubit_advantage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per interaction level: mean hypervolume of structured_bo/qkernel_bo
    at every qubit count, the quantum advantage
    (``qkernel_bo - structured_bo``), and which qubit count maximizes it —
    the direct answer to "at how many qubits does quantum help most".
    """
    both = df[df["optimizer"].isin(["structured_bo", "qkernel_bo"])]
    summary = (
        both.groupby(["interaction", "qubits_per_group", "optimizer"])["hypervolume"]
        .mean()
        .unstack("optimizer")
        .reset_index()
    )
    summary["quantum_advantage"] = summary["qkernel_bo"] - summary["structured_bo"]

    rows = []
    for interaction, group in summary.groupby("interaction"):
        best = group.loc[group["quantum_advantage"].idxmax()]
        rows.append(
            {
                "interaction": interaction,
                "best_qubit_count": int(best["qubits_per_group"]),
                "structured_bo_hv": float(best["structured_bo"]),
                "qkernel_bo_hv": float(best["qkernel_bo"]),
                "quantum_advantage": float(best["quantum_advantage"]),
            }
        )
    return pd.DataFrame(rows).sort_values("interaction").reset_index(drop=True)


def make_qubit_advantage_figure(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    summary = (
        df[df["optimizer"].isin(["structured_bo", "qkernel_bo"])]
        .groupby(["interaction", "qubits_per_group", "optimizer"])["hypervolume"]
        .agg(["mean", "std"])
        .reset_index()
    )
    interactions = sorted(summary["interaction"].unique())

    fig, axes = plt.subplots(
        1, len(interactions), figsize=(5 * len(interactions), 4), squeeze=False
    )
    labels = {"structured_bo": "Classical (Matérn)", "qkernel_bo": "Quantum (fidelity)"}
    styles = {"structured_bo": "o-", "qkernel_bo": "s--"}
    for col, inter in enumerate(interactions):
        ax = axes[0][col]
        cell = summary[summary["interaction"] == inter]
        for opt_name in ("structured_bo", "qkernel_bo"):
            sub = cell[cell["optimizer"] == opt_name].sort_values("qubits_per_group")
            if sub.empty:
                continue
            ax.errorbar(
                sub["qubits_per_group"],
                sub["mean"],
                yerr=sub["std"],
                fmt=styles[opt_name],
                capsize=3,
                label=labels[opt_name],
            )
        ax.set_title(f"interaction = {inter}")
        ax.set_xlabel("Qubits per group")
        ax.set_ylabel("Hypervolume")
        ax.set_xticks(sorted(summary["qubits_per_group"].unique()))
        ax.grid(alpha=0.3)
        if summary["qubits_per_group"].max() > 5:
            ax.axvline(5, color="grey", linestyle=":", linewidth=1)
            ax.text(
                5,
                ax.get_ylim()[1],
                " Brief D3 cap",
                fontsize=7,
                color="grey",
                va="top",
                ha="left",
            )
    axes[0][0].legend(fontsize="small")
    fig.suptitle("Classical vs. quantum kernel BO: hypervolume vs. qubit count")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def layer_a_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["n_groups", "conditional_fraction", "interaction", "optimizer"])[
            "hypervolume"
        ]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["interaction", "conditional_fraction", "n_groups", "optimizer"])
        .reset_index(drop=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qubit-sweep-csv", default="results/qubit_sweep/qubit_sweep_results.csv"
    )
    parser.add_argument(
        "--qubit-sweep-extended-csv",
        default="results/qubit_sweep_extended/qubit_sweep_extended_results.csv",
        help="optional 5/10/15/20-qubit sweep (scripts/run_qubit_sweep_extended.py); "
        "folded into the same table/figure as the base 1-5 sweep if present",
    )
    parser.add_argument("--synth-csv", default="results/synth/sweep_results.csv")
    parser.add_argument("--output-dir", default="results/paper")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qubit_csv = Path(args.qubit_sweep_csv)
    extended_csv = Path(args.qubit_sweep_extended_csv)
    if qubit_csv.exists() or extended_csv.exists():
        frames = []
        if qubit_csv.exists():
            frames.append(pd.read_csv(qubit_csv))
        else:
            _logger.warning("%s not found — figure will only cover the extended range", qubit_csv)
        if extended_csv.exists():
            frames.append(pd.read_csv(extended_csv))
            _logger.info("folding in extended qubit range from %s", extended_csv)
        df = pd.concat(frames, ignore_index=True)
        table = qubit_advantage_table(df)
        (output_dir / "qubit_advantage_table.md").write_text(_markdown_table(table))
        table.to_csv(output_dir / "qubit_advantage_table.csv", index=False)
        make_qubit_advantage_figure(df, output_dir / "qubit_advantage_figure.png")
        _logger.info("wrote qubit-count advantage table + figure to %s", output_dir)
        print(table.to_string(index=False))
    else:
        _logger.warning(
            "neither %s nor %s found — run `python -m scripts.run_qubit_sweep` "
            "(and optionally `run_qubit_sweep_extended`) first",
            qubit_csv,
            extended_csv,
        )

    synth_csv = Path(args.synth_csv)
    if synth_csv.exists():
        df = pd.read_csv(synth_csv)
        table = layer_a_summary_table(df)
        (output_dir / "layer_a_summary_table.md").write_text(_markdown_table(table))
        table.to_csv(output_dir / "layer_a_summary_table.csv", index=False)
        _logger.info("wrote Layer-A summary table to %s", output_dir)
    else:
        _logger.warning("%s not found — run `python -m scripts.run_synth` first", synth_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

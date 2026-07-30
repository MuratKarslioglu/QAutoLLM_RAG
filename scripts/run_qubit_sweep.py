#!/usr/bin/env python3
"""Qubit-count architecture comparison (extends the Brief §9 P7 Layer-A
benchmark): sweeps classical-vs-quantum kernel BO performance as a function
of per-group qubit count, crossed with the additive/misspecified
(``interaction``) axis, holding ``n_groups`` and conditionality fixed so the
qubit axis isn't confounded with the others (see
:func:`qautollm.synth.benchmark.make_qubit_sweep_space`).

Answers: at how many qubits does the quantum fidelity kernel start to help
(or hurt) relative to the classical Matérn+categorical-overlap kernel, and
does that depend on whether the true objective is well-specified (additive)
or not?

Usage:
    python -m scripts.run_qubit_sweep --quick   # smoke-scale, seconds
    python -m scripts.run_qubit_sweep           # full scale — GP fits dominate
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import qautollm.optimizers  # noqa: F401 - populates OPTIMIZERS registry
from qautollm.core.registry import OPTIMIZERS
from qautollm.space.encoding import encode
from qautollm.surrogate.pareto_archive import ParetoArchive
from qautollm.synth.benchmark import build_synthetic_objectives, make_qubit_sweep_space
from qautollm.utils.logging import get_logger
from qautollm.utils.profiling import progress

_logger = get_logger(__name__)

FULL_SCALE = {
    "n_groups": 3,
    "qubit_counts": (1, 2, 3, 4, 5),
    "interactions": (0.0, 1.0),
    "optimizer_names": ("structured_bo", "qkernel_bo"),
    # budget=30 (pre-2026-07-18) was too small to detect any classical/
    # quantum divergence at all: acquisition-value differences between the
    # two kernels (~1e-4, confirmed live) were 3-4 orders of magnitude
    # smaller than the between-candidate acquisition spread in that
    # exploration-dominated small-n regime, so the argmax never flipped --
    # all 50 (qubit_count x interaction x seed) settings produced
    # bit-identical hypervolume, telling us nothing. Bumped to match the
    # real main-campaign budget (100), where genuine (if small) divergence
    # was actually observed between structured_bo and qkernel_bo.
    "budget": 100,
    "init_points": 10,
    "n_seeds": 5,
}

QUICK_SCALE = {
    "n_groups": 3,
    "qubit_counts": (1, 3, 5),
    "interactions": (0.0, 1.0),
    "optimizer_names": ("structured_bo", "qkernel_bo"),
    "budget": 15,
    "init_points": 6,
    "n_seeds": 2,
}


def run_one(
    n_groups: int,
    qubits_per_group: int,
    interaction: float,
    optimizer_name: str,
    seed: int,
    budget: int,
    init_points: int,
) -> dict:
    space = make_qubit_sweep_space(n_groups, qubits_per_group)
    objective_fn = build_synthetic_objectives(space, interaction, seed=seed)

    init_design = [encode(raw, space) for raw in space.sobol_design(init_points, seed=seed)]
    optimizer_cls = OPTIMIZERS.get(optimizer_name)
    optimizer = optimizer_cls(space, seed, init_design)

    archive = ParetoArchive()
    start = time.perf_counter()
    for _ in range(budget):
        cfg = optimizer.suggest()
        objectives = objective_fn(cfg)
        optimizer.update(cfg, objectives)
        archive.update(objectives)
    elapsed = time.perf_counter() - start

    ref_point = [2.0, 2.0, 2.0]  # generous fixed upper bound; objective_fn's range is O(1)
    return {
        "n_groups": n_groups,
        "qubits_per_group": qubits_per_group,
        "interaction": interaction,
        "optimizer": optimizer_name,
        "seed": seed,
        "budget": budget,
        "hypervolume": archive.hypervolume(ref_point),
        "wall_clock_s": elapsed,
    }


def run_sweep(
    n_groups: int,
    qubit_counts: tuple[int, ...],
    interactions: tuple[float, ...],
    optimizer_names: tuple[str, ...],
    budget: int,
    init_points: int,
    n_seeds: int,
) -> pd.DataFrame:
    combos = list(itertools.product(qubit_counts, interactions, optimizer_names, range(n_seeds)))
    rows = []
    for qubits_per_group, interaction, optimizer_name, seed in progress(
        combos, desc="qubit sweep", total=len(combos)
    ):
        rows.append(
            run_one(
                n_groups, qubits_per_group, interaction, optimizer_name, seed, budget, init_points
            )
        )
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        df.groupby(["qubits_per_group", "interaction", "optimizer"])["hypervolume"]
        .agg(["mean", "std"])
        .reset_index()
    )
    interactions = sorted(summary["interaction"].unique())
    optimizers = sorted(summary["optimizer"].unique())

    fig, axes = plt.subplots(
        1, len(interactions), figsize=(5 * len(interactions), 4), squeeze=False
    )
    for col, inter in enumerate(interactions):
        ax = axes[0][col]
        cell = summary[summary["interaction"] == inter]
        for opt_name in optimizers:
            sub = cell[cell["optimizer"] == opt_name].sort_values("qubits_per_group")
            if sub.empty:
                continue
            ax.errorbar(
                sub["qubits_per_group"], sub["mean"], yerr=sub["std"], marker="o", label=opt_name
            )
        ax.set_title(f"interaction={inter}")
        ax.set_xlabel("qubits per group")
        ax.set_ylabel("hypervolume")
        ax.set_xticks(sorted(summary["qubits_per_group"].unique()))
    axes[0][0].legend(fontsize="small")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "qubit_sweep.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke-scale sweep")
    parser.add_argument("--output-dir", default="results/qubit_sweep")
    args = parser.parse_args()

    scale = QUICK_SCALE if args.quick else FULL_SCALE
    _logger.info("running qubit sweep (%s scale): %s", "quick" if args.quick else "full", scale)

    df = run_sweep(
        n_groups=scale["n_groups"],
        qubit_counts=scale["qubit_counts"],
        interactions=scale["interactions"],
        optimizer_names=scale["optimizer_names"],
        budget=scale["budget"],
        init_points=scale["init_points"],
        n_seeds=scale["n_seeds"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "qubit_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    _logger.info("wrote %s (%d rows)", csv_path, len(df))

    make_plots(df, output_dir)
    _logger.info("wrote plots to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

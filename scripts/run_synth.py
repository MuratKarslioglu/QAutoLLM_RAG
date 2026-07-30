#!/usr/bin/env python3
"""Layer-A synthetic benchmark sweep (Brief §9 P7): every optimizer against
every ``(n_groups, conditional_fraction, interaction)`` combination, over
several seeds, tracking final hypervolume. Saves a CSV + summary plots.

Usage:
    python -m scripts.run_synth            # full sweep (Brief-scale; can take
                                            # a long time — GP fits for
                                            # structured_bo/qkernel_bo dominate)
    python -m scripts.run_synth --quick     # small smoke-scale sweep, seconds

The full-scale sweep is the one whose results actually belong in the paper;
``--quick`` exists only to verify the pipeline runs correctly end-to-end.
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
from qautollm.synth.benchmark import build_synthetic_objectives, make_synthetic_space
from qautollm.utils.logging import get_logger
from qautollm.utils.profiling import progress

_logger = get_logger(__name__)

FULL_SCALE = {
    "dims": (3, 5, 8),
    "conditionalities": (0.0, 0.5),
    "interactions": (0.0, 0.5, 1.0),
    "optimizer_names": ("random", "tpe", "structured_bo", "qkernel_bo", "nsga2"),
    "budget": 30,
    "init_points": 10,
    "n_seeds": 5,
}

QUICK_SCALE = {
    "dims": (3, 5),
    "conditionalities": (0.0,),
    "interactions": (0.0, 1.0),
    "optimizer_names": ("random", "tpe", "structured_bo", "qkernel_bo", "nsga2"),
    "budget": 15,
    "init_points": 6,
    "n_seeds": 2,
}


def run_one(
    n_groups: int,
    conditional_fraction: float,
    interaction: float,
    optimizer_name: str,
    seed: int,
    budget: int,
    init_points: int,
) -> dict:
    space = make_synthetic_space(n_groups, conditional_fraction)
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
        "conditional_fraction": conditional_fraction,
        "interaction": interaction,
        "optimizer": optimizer_name,
        "seed": seed,
        "budget": budget,
        "hypervolume": archive.hypervolume(ref_point),
        "wall_clock_s": elapsed,
    }


def run_sweep(
    dims: tuple[int, ...],
    conditionalities: tuple[float, ...],
    interactions: tuple[float, ...],
    optimizer_names: tuple[str, ...],
    budget: int,
    init_points: int,
    n_seeds: int,
) -> pd.DataFrame:
    combos = list(
        itertools.product(dims, conditionalities, interactions, optimizer_names, range(n_seeds))
    )
    rows = []
    for n_groups, conditional_fraction, interaction, optimizer_name, seed in progress(
        combos, desc="synth sweep", total=len(combos)
    ):
        rows.append(
            run_one(
                n_groups,
                conditional_fraction,
                interaction,
                optimizer_name,
                seed,
                budget,
                init_points,
            )
        )
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        df.groupby(["n_groups", "conditional_fraction", "interaction", "optimizer"])["hypervolume"]
        .agg(["mean", "std"])
        .reset_index()
    )
    conditionalities = sorted(summary["conditional_fraction"].unique())
    interactions = sorted(summary["interaction"].unique())
    optimizers = sorted(summary["optimizer"].unique())

    fig, axes = plt.subplots(
        len(conditionalities),
        len(interactions),
        figsize=(4 * len(interactions), 3.5 * len(conditionalities)),
        squeeze=False,
    )
    for row, cond in enumerate(conditionalities):
        for col, inter in enumerate(interactions):
            ax = axes[row][col]
            cell = summary[
                (summary["conditional_fraction"] == cond) & (summary["interaction"] == inter)
            ]
            for opt_name in optimizers:
                sub = cell[cell["optimizer"] == opt_name].sort_values("n_groups")
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["n_groups"], sub["mean"], yerr=sub["std"], marker="o", label=opt_name
                )
            ax.set_title(f"conditionality={cond}, interaction={inter}")
            ax.set_xlabel("n_groups (dim)")
            ax.set_ylabel("hypervolume")
    axes[0][0].legend(fontsize="small")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "layer_a_hypervolume_sweep.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke-scale sweep")
    parser.add_argument("--output-dir", default="results/synth")
    args = parser.parse_args()

    scale = QUICK_SCALE if args.quick else FULL_SCALE
    _logger.info("running Layer-A sweep (%s scale): %s", "quick" if args.quick else "full", scale)

    df = run_sweep(
        dims=scale["dims"],
        conditionalities=scale["conditionalities"],
        interactions=scale["interactions"],
        optimizer_names=scale["optimizer_names"],
        budget=scale["budget"],
        init_points=scale["init_points"],
        n_seeds=scale["n_seeds"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    _logger.info("wrote %s (%d rows)", csv_path, len(df))

    make_plots(df, output_dir)
    _logger.info("wrote plots to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

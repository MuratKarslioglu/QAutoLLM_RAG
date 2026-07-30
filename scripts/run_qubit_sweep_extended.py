#!/usr/bin/env python3
"""Exploratory qubit-count sweep BEYOND Brief D3's 5-qubit-per-group cap
(2026-07-18, requested after the 1-5 qubit sweep showed a flat "no
advantage anywhere" result and the user asked to push further).

This intentionally goes past Brief D3 ("Group circuits are <= 5 qubits").
It exists to characterize *scaling* behavior, not to replace or feed back
into the Brief-compliant real pipeline (``qkernel_bo``/rag_v2.yaml's groups
are untouched, still capped at 5). Kept as a fully separate script/output
directory so it can never be confused with the committed-scope results in
``results/qubit_sweep``.

Feasibility check performed before writing this (live timing, not a
guess): PennyLane's default ``default.qubit`` simulator takes ~0.75s for a
single 20-qubit circuit evaluation -- with ~256 candidate evals per EHVI
suggestion x ~90 suggestions x 5 seeds x 2 interactions, that's ~48h just
for qubit_count=20. Switching to ``lightning.qubit`` (PennyLane's own
optimized statevector backend, already a declared dependency, identical
physics -- not an approximation) measured at ~0.04s for the same circuit,
an ~18x speedup, bringing qubit_count=20 down to ~2.5h. Still the most
expensive single point in the grid by far.

Usage:
    python -m scripts.run_qubit_sweep_extended --quick   # smoke-scale
    python -m scripts.run_qubit_sweep_extended           # full scale
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
from qautollm.kernels.quantum import quantum_group_kernel_factory
from qautollm.optimizers.structured_bo import KernelBO
from qautollm.space.encoding import encode
from qautollm.surrogate.pareto_archive import ParetoArchive
from qautollm.synth.benchmark import build_synthetic_objectives, make_qubit_sweep_space
from qautollm.utils.logging import get_logger
from qautollm.utils.profiling import progress

_logger = get_logger(__name__)

MAX_QUBITS_OVERRIDE = 20
DEVICE_NAME = "lightning.qubit"

FULL_SCALE = {
    "n_groups": 3,
    "qubit_counts": (5, 10, 15, 20),
    "interactions": (0.0, 1.0),
    "budget": 100,
    "init_points": 10,
    "n_seeds": 5,
}

QUICK_SCALE = {
    "n_groups": 3,
    "qubit_counts": (5, 10),
    "interactions": (0.0, 1.0),
    "budget": 15,
    "init_points": 6,
    "n_seeds": 2,
}


class _ExtendedQKernelBO(KernelBO):
    """Same ``KernelBO`` machinery as the real ``qkernel_bo`` -- deliberately
    NOT registered in ``OPTIMIZERS`` (this must never be reachable from a
    real experiment config; it exists only for this script). The only
    difference from production ``QKernelBO`` is the two overrides passed
    into ``quantum_group_kernel_factory``.
    """

    def __init__(self, search_space, seed, init_design, max_qubits, device_name):
        super().__init__(search_space, seed, init_design)
        self.group_kernel_factory = quantum_group_kernel_factory(
            depth=1, shots=None, seed=seed, max_qubits=max_qubits, device_name=device_name
        )


def run_one(
    n_groups: int,
    qubits_per_group: int,
    interaction: float,
    optimizer_name: str,
    seed: int,
    budget: int,
    init_points: int,
) -> dict:
    space = make_qubit_sweep_space(n_groups, qubits_per_group, max_qubits=MAX_QUBITS_OVERRIDE)
    objective_fn = build_synthetic_objectives(space, interaction, seed=seed)
    init_design = [encode(raw, space) for raw in space.sobol_design(init_points, seed=seed)]

    if optimizer_name == "qkernel_bo":
        optimizer = _ExtendedQKernelBO(
            space, seed, init_design, max_qubits=MAX_QUBITS_OVERRIDE, device_name=DEVICE_NAME
        )
    else:
        optimizer = OPTIMIZERS.get(optimizer_name)(space, seed, init_design)

    archive = ParetoArchive()
    start = time.perf_counter()
    for _ in range(budget):
        cfg = optimizer.suggest()
        objectives = objective_fn(cfg)
        optimizer.update(cfg, objectives)
        archive.update(objectives)
    elapsed = time.perf_counter() - start

    ref_point = [2.0, 2.0, 2.0]
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
    budget: int,
    init_points: int,
    n_seeds: int,
) -> pd.DataFrame:
    optimizer_names = ("structured_bo", "qkernel_bo")
    combos = list(itertools.product(qubit_counts, interactions, optimizer_names, range(n_seeds)))
    # expensive (high qubit_count) combos first so a partial run still
    # covers the full qubit range at lower seed counts if interrupted.
    combos.sort(key=lambda c: -c[0])
    rows = []
    for qubits_per_group, interaction, optimizer_name, seed in progress(
        combos, desc="extended qubit sweep", total=len(combos)
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
        ax.set_xlabel("qubits per group (beyond Brief D3's 5-qubit cap)")
        ax.set_ylabel("hypervolume")
        ax.set_xticks(sorted(summary["qubits_per_group"].unique()))
    axes[0][0].legend(fontsize="small")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "qubit_sweep_extended.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke-scale sweep")
    parser.add_argument("--output-dir", default="results/qubit_sweep_extended")
    args = parser.parse_args()

    scale = QUICK_SCALE if args.quick else FULL_SCALE
    _logger.info(
        "running EXTENDED qubit sweep (beyond Brief D3's 5-qubit cap, %s scale): %s",
        "quick" if args.quick else "full",
        scale,
    )

    df = run_sweep(
        n_groups=scale["n_groups"],
        qubit_counts=scale["qubit_counts"],
        interactions=scale["interactions"],
        budget=scale["budget"],
        init_points=scale["init_points"],
        n_seeds=scale["n_seeds"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "qubit_sweep_extended_results.csv"
    df.to_csv(csv_path, index=False)
    _logger.info("wrote %s (%d rows)", csv_path, len(df))

    make_plots(df, output_dir)
    _logger.info("wrote plots to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

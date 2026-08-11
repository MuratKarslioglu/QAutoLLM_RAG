#!/usr/bin/env python3
"""Feature-map depth sweep (2026-08-06): does a more expressive quantum
feature map (``feature_map_depth`` > 1, i.e. more data-re-uploading layers
-- the circuit re-encodes ``z`` at every layer, see
``qautollm.kernels.quantum._build_circuit``) create a genuine quantum-kernel
advantage over the classical Matérn+categorical-overlap baseline, on the
*synthetic* benchmark?

Motivation: the main HotpotQA campaign and the qubit-count sweeps (both
Brief-compliant 1-5 and the exploratory 5-20 range) all used
``feature_map_depth=1`` (the config default) and found no advantage in any
tested regime (see results/paper/qubit_advantage_figure.png -- the two
curves are visually identical). Depth is the one quantum-expressivity knob
with no classical analog; this sweep tests it in isolation, pre-registered
before looking at any results: report every depth value tried, not just
whichever one happens to look best. Stays within Brief D3's 5-qubit cap and
uses ``default.qubit`` (CPU) -- this is a controlled, synthetic-only
hypothesis test, not a change to the real pipeline (qkernel_bo.py and
rag_v2.yaml are untouched).

2026-08-11: this script was never run at real (non-``--quick``) scale before
a Clifford-periodicity bug was found and fixed in
``qautollm.kernels.quantum._build_circuit`` (see
``scripts/run_depth_qubit_grid.py``'s docstring for the full story -- that
script's real run had ``depth=10``/``depth=20`` collapse to ``depth=0``
exactly). This sweep's ``depths={1,2,3,4}`` never confirmed hitting that
specific revival before the fix landed (the exact period wasn't determined,
only that it divides both 10 and 20 but not 5), but the fix is in place
regardless, so any run of this script from now on already uses the
corrected circuit -- no action needed here beyond this note.

Usage:
    python -m scripts.run_depth_sweep --quick   # smoke-scale
    python -m scripts.run_depth_sweep           # full scale
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

FULL_SCALE = {
    "n_groups": 3,
    "qubit_counts": (3, 5),
    "interactions": (0.0, 1.0),
    "depths": (1, 2, 3, 4),
    "optimizer_names": ("structured_bo", "qkernel_bo"),
    "budget": 100,
    "init_points": 10,
    # 5, not the main campaign's 3: this is exactly the underpowered-n_seeds
    # gap flagged when assessing publication readiness -- fixed here since
    # this is a fresh sweep, not retrofitted onto the real campaign (that
    # needs its own, separate, explicitly-confirmed re-run given its cost).
    "n_seeds": 5,
}

QUICK_SCALE = {
    "n_groups": 3,
    "qubit_counts": (3, 5),
    "interactions": (0.0, 1.0),
    "depths": (1, 2),
    "optimizer_names": ("structured_bo", "qkernel_bo"),
    "budget": 15,
    "init_points": 6,
    "n_seeds": 2,
}


class _DepthQKernelBO(KernelBO):
    """Same ``KernelBO`` machinery as production ``qkernel_bo`` -- the one
    and only difference is ``feature_map_depth`` (Brief §7's
    ``optimizer.quantum.feature_map_depth``), explicit here instead of
    coming from an experiment YAML. ``max_qubits``/``device_name`` are left
    at :mod:`qautollm.kernels.quantum`'s Brief-compliant defaults (5-qubit
    cap, ``default.qubit``) since qubit_counts here never exceeds 5.
    """

    def __init__(self, search_space, seed, init_design, depth, n_candidates=256, mc_samples=128):
        super().__init__(
            search_space, seed, init_design, n_candidates=n_candidates, mc_samples=mc_samples
        )
        self.group_kernel_factory = quantum_group_kernel_factory(depth=depth, shots=None, seed=seed)


def run_one(
    n_groups: int,
    qubits_per_group: int,
    interaction: float,
    depth: int,
    optimizer_name: str,
    seed: int,
    budget: int,
    init_points: int,
) -> dict:
    space = make_qubit_sweep_space(n_groups, qubits_per_group)
    objective_fn = build_synthetic_objectives(space, interaction, seed=seed)
    init_design = [encode(raw, space) for raw in space.sobol_design(init_points, seed=seed)]

    if optimizer_name == "qkernel_bo":
        optimizer = _DepthQKernelBO(space, seed, init_design, depth=depth)
    else:
        # structured_bo (classical) has no feature_map_depth analog -- its
        # result is identical for every depth value at fixed
        # (qubits_per_group, interaction, seed). Recomputed per depth
        # anyway (cheap, classical kernel) so every row in the output CSV
        # carries a depth value and the table stays flat/groupby-friendly,
        # matching this repo's other sweep scripts' convention.
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
        "depth": depth,
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
    depths: tuple[int, ...],
    optimizer_names: tuple[str, ...],
    budget: int,
    init_points: int,
    n_seeds: int,
) -> pd.DataFrame:
    combos = list(
        itertools.product(qubit_counts, interactions, depths, optimizer_names, range(n_seeds))
    )
    rows = [
        run_one(n_groups, qubits, interaction, depth, optimizer_name, seed, budget, init_points)
        for qubits, interaction, depth, optimizer_name, seed in progress(
            combos, desc="depth sweep", total=len(combos)
        )
    ]
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        df.groupby(["qubits_per_group", "interaction", "depth", "optimizer"])["hypervolume"]
        .agg(["mean", "std"])
        .reset_index()
    )
    qubit_counts = sorted(summary["qubits_per_group"].unique())
    interactions = sorted(summary["interaction"].unique())
    optimizers = sorted(summary["optimizer"].unique())

    fig, axes = plt.subplots(
        len(qubit_counts),
        len(interactions),
        figsize=(5 * len(interactions), 4 * len(qubit_counts)),
        squeeze=False,
    )
    for row, qubits in enumerate(qubit_counts):
        for col, inter in enumerate(interactions):
            ax = axes[row][col]
            cell = summary[
                (summary["qubits_per_group"] == qubits) & (summary["interaction"] == inter)
            ]
            for opt_name in optimizers:
                sub = cell[cell["optimizer"] == opt_name].sort_values("depth")
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["depth"], sub["mean"], yerr=sub["std"], marker="o", label=opt_name
                )
            ax.set_title(f"qubits={qubits}, interaction={inter}")
            ax.set_xlabel("feature_map_depth")
            ax.set_ylabel("hypervolume")
            ax.set_xticks(sorted(summary["depth"].unique()))
    axes[0][0].legend(fontsize="small")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "depth_sweep.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke-scale sweep")
    parser.add_argument("--output-dir", default="results/depth_sweep")
    args = parser.parse_args()

    scale = QUICK_SCALE if args.quick else FULL_SCALE
    _logger.info("running depth sweep (%s scale): %s", "quick" if args.quick else "full", scale)

    df = run_sweep(
        n_groups=scale["n_groups"],
        qubit_counts=scale["qubit_counts"],
        interactions=scale["interactions"],
        depths=scale["depths"],
        optimizer_names=scale["optimizer_names"],
        budget=scale["budget"],
        init_points=scale["init_points"],
        n_seeds=scale["n_seeds"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "depth_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    _logger.info("wrote %s (%d rows)", csv_path, len(df))

    make_plots(df, output_dir)
    _logger.info("wrote plots to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

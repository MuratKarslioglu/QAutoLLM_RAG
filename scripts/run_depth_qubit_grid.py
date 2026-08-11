#!/usr/bin/env python3
"""2D grid: feature_map_depth x qubits_per_group (2026-08-06), both beyond
Brief D3's defaults, crossed together to look for a quantum-kernel
advantage regime that neither axis alone (see scripts.run_depth_sweep,
scripts.run_qubit_sweep_extended -- both flat/no-advantage) revealed.

Reuses scripts.run_qubit_sweep_extended's OOM hardening wholesale (RLIMIT_AS
worker cap, periodic malloc_trim, MALLOC_ARENA_MAX, lightning.qubit for
speed) rather than re-deriving it: qubits_per_group up to 20 here carries
the *exact* memory-growth risk already characterized there (worse, since
depth up to 20 also multiplies wall-clock per circuit call -- deeper
circuits are slower, not more memory-hungry: a statevector is still
2**n_qubits regardless of depth).

qubit_count=0 is not meaningful (dropped); depth=0 is kept as a deliberate
degenerate control point (zero encoding layers -> the circuit's output
never depends on the input, so the "kernel" is a constant -- expect near-
-zero hypervolume there, which is itself a useful correctness sanity check,
not a bug).

2026-08-11: the first real run of this grid (results/depth_qubit_grid_BROKEN_
pre_periodicity_fix/, kept only as the artifact that surfaced the bug -- do
not use it) had depth=10 and depth=20 collapse to bit-identical results with
depth=0 for every qkernel_bo cell. Root cause: this project's search spaces
are entirely one-hot categorical, so RY(pi*z) is Clifford for both z=0 and
z=1, and re-uploading the same angle every layer made the whole circuit a
fixed Clifford unitary to the depth-th power -- which has a small finite
period, not a real depth-expressivity signal. Fixed in
qautollm/kernels/quantum.py (_build_circuit's per-layer frequency
multiplier; depth=1, the only depth any production run ever uses, is
untouched). Any results collected before this fix must be re-run.

Usage:
    python -m scripts.run_depth_qubit_grid --quick   # smoke-scale
    python -m scripts.run_depth_qubit_grid           # full scale (see
        module docstring in run_qubit_sweep_extended.py for why this can
        run for hours: qubits=20 x depth=20 is the single most expensive
        cell in the grid by a wide margin)
"""

from __future__ import annotations

import os

# Must be set before torch gets imported anywhere below (directly or via
# qautollm.optimizers, imported a few lines down) -- see
# run_qubit_sweep_extended.py's identical setting for the full rationale
# (torch's lazy CUDA probe fails noisily on this host's driver setup even
# though DEVICE_NAME below is CPU-only and never touches the GPU). Every
# spawned worker re-executes this whole module fresh, so this must be a
# module-level statement, not something done once in main().
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import logging

logging.getLogger("torch.utils.cpp_extension").setLevel(logging.ERROR)

import argparse
import itertools
import multiprocessing as mp
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import qautollm.optimizers  # noqa: F401 - populates OPTIMIZERS registry
import scripts.run_qubit_sweep_extended as rqse
from qautollm.core.registry import OPTIMIZERS
from qautollm.kernels.quantum import quantum_group_kernel_factory
from qautollm.optimizers.structured_bo import KernelBO
from qautollm.space.encoding import encode
from qautollm.surrogate.pareto_archive import ParetoArchive
from qautollm.synth.benchmark import build_synthetic_objectives, make_qubit_sweep_space
from qautollm.utils.logging import get_logger
from qautollm.utils.profiling import progress

_logger = get_logger(__name__)

DEVICE_NAME = "lightning.qubit"

# Kept intentionally conservative -- matches HARDWARE_SCALE's validated-safe
# budget/n_candidates in run_qubit_sweep_extended.py (36/36 real combos,
# zero failures there), not the heavier FULL_SCALE numbers, since this grid
# already multiplies cost across two axes at once. Even so: qubits=20 x
# depth=20 is meaningfully slower per circuit call than qubits=20 x depth=1
# was (more gates -> more wall-clock, not more memory) -- expect this to be
# the long pole. Time it on --quick first and extrapolate before committing
# to a multi-hour --full run.
FULL_SCALE = {
    "n_groups": 3,
    "qubit_counts": (5, 10, 15, 20),
    "depths": (0, 5, 10, 20),
    "interactions": (0.0, 1.0),
    "budget": 40,
    "init_points": 8,
    "n_seeds": 3,
    "n_candidates": 24,
    "mc_samples": 16,
}

QUICK_SCALE = {
    "n_groups": 3,
    "qubit_counts": (5, 20),
    "depths": (0, 20),
    "interactions": (0.0, 1.0),
    "budget": 10,
    "init_points": 6,
    "n_seeds": 1,
    "n_candidates": 16,
    "mc_samples": 8,
}

MAX_QUBITS_OVERRIDE = rqse.MAX_QUBITS_OVERRIDE  # 25, just a validation ceiling


class _DepthQubitQKernelBO(KernelBO):
    """Same ``KernelBO`` machinery as production ``qkernel_bo`` --
    ``depth``, ``max_qubits``, and ``device_name`` are all explicit
    overrides here (union of run_depth_sweep.py's ``_DepthQKernelBO`` and
    run_qubit_sweep_extended.py's ``_ExtendedQKernelBO``, since this script
    varies both axes at once).
    """

    def __init__(
        self, search_space, seed, init_design, depth, max_qubits, device_name,
        n_candidates, mc_samples,
    ):
        super().__init__(
            search_space, seed, init_design, n_candidates=n_candidates, mc_samples=mc_samples
        )
        self.group_kernel_factory = quantum_group_kernel_factory(
            depth=depth, shots=None, seed=seed, max_qubits=max_qubits, device_name=device_name
        )


def _apply_with_gpu_retries(pool, args: tuple) -> dict:
    """Same retry rationale as ``rqse._apply_with_gpu_retries`` (lightning.gpu's
    tagged RuntimeError is transient driver flakiness, not a deterministic
    limit -- see that function's docstring), reimplemented locally because
    the shared one is hardcoded to call *its own module's* ``run_one``,
    whose positional signature (no ``depth``) doesn't match this module's.
    """
    last_exc: RuntimeError | None = None
    for attempt in range(1, rqse._GPU_ERROR_MAX_RETRIES + 1):
        try:
            return pool.apply(run_one, args)
        except RuntimeError as exc:
            if rqse._LIGHTNING_GPU_ERROR_TAG not in str(exc).lower():
                raise
            last_exc = exc
            if attempt < rqse._GPU_ERROR_MAX_RETRIES:
                _logger.warning(
                    "lightning.gpu error on attempt %d/%d (retrying in a fresh worker): %s",
                    attempt,
                    rqse._GPU_ERROR_MAX_RETRIES,
                    str(exc)[:200],
                )
    assert last_exc is not None
    raise last_exc


def run_one(
    n_groups: int,
    qubits_per_group: int,
    interaction: float,
    depth: int,
    optimizer_name: str,
    seed: int,
    budget: int,
    init_points: int,
    n_candidates: int,
    mc_samples: int,
) -> dict:
    space = make_qubit_sweep_space(n_groups, qubits_per_group, max_qubits=MAX_QUBITS_OVERRIDE)
    objective_fn = build_synthetic_objectives(space, interaction, seed=seed)
    init_design = [encode(raw, space) for raw in space.sobol_design(init_points, seed=seed)]

    if optimizer_name == "qkernel_bo":
        optimizer = _DepthQubitQKernelBO(
            space,
            seed,
            init_design,
            depth=depth,
            max_qubits=MAX_QUBITS_OVERRIDE,
            device_name=DEVICE_NAME,
            n_candidates=n_candidates,
            mc_samples=mc_samples,
        )
    else:
        # structured_bo has no depth analog; recomputed per depth anyway
        # (cheap, classical kernel) so the output table stays flat -- see
        # run_depth_sweep.py's identical convention/rationale.
        optimizer = OPTIMIZERS.get(optimizer_name)(
            space, seed, init_design, n_candidates=n_candidates, mc_samples=mc_samples
        )

    archive = ParetoArchive()
    start = time.perf_counter()
    for i in range(budget):
        cfg = optimizer.suggest()
        objectives = objective_fn(cfg)
        optimizer.update(cfg, objectives)
        archive.update(objectives)
        if (i + 1) % rqse._TRIM_EVERY == 0:
            rqse._malloc_trim()
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
        "failed": False,
    }


def run_sweep(
    n_groups: int,
    qubit_counts: tuple[int, ...],
    depths: tuple[int, ...],
    interactions: tuple[float, ...],
    budget: int,
    init_points: int,
    n_seeds: int,
    n_candidates: int,
    mc_samples: int,
) -> pd.DataFrame:
    optimizer_names = ("structured_bo", "qkernel_bo")
    combos = list(
        itertools.product(qubit_counts, depths, interactions, optimizer_names, range(n_seeds))
    )
    # most expensive (high qubit_count, high depth) first -- same rationale
    # as run_qubit_sweep_extended.py: a partial/interrupted run still covers
    # the full range at lower seed counts.
    combos.sort(key=lambda c: (-c[0], -c[1]))
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=1,
        maxtasksperchild=1,
        initializer=rqse._limit_worker_address_space,
        initargs=(DEVICE_NAME,),
    ) as pool:
        for qubits_per_group, depth, interaction, optimizer_name, seed in progress(
            combos, desc="depth x qubit grid", total=len(combos)
        ):
            args = (
                n_groups,
                qubits_per_group,
                interaction,
                depth,
                optimizer_name,
                seed,
                budget,
                init_points,
                n_candidates,
                mc_samples,
            )
            try:
                rows.append(_apply_with_gpu_retries(pool, args))
            except (MemoryError, RuntimeError) as exc:
                # Same reasoning as run_qubit_sweep_extended.py's run_sweep():
                # RLIMIT_AS tripping (MemoryError, CPU) is deterministic given
                # the combo's data size, so not retried. lightning.gpu's own
                # tagged RuntimeError (_LIGHTNING_GPU_ERROR_TAG) already went
                # through _GPU_ERROR_MAX_RETRIES fresh-worker retries in
                # _apply_with_gpu_retries and still failed -- infeasible on
                # this host/run, not a bug. Any *other* RuntimeError is a real
                # bug in our own code and must not be swallowed.
                if isinstance(exc, RuntimeError) and rqse._LIGHTNING_GPU_ERROR_TAG not in str(
                    exc
                ).lower():
                    raise
                _logger.warning(
                    "combo qubits=%d depth=%d interaction=%s optimizer=%s seed=%d exceeded "
                    "the worker memory cap (%s) -- marking failed and continuing",
                    qubits_per_group,
                    depth,
                    interaction,
                    optimizer_name,
                    seed,
                    type(exc).__name__,
                )
                rows.append(
                    {
                        "n_groups": n_groups,
                        "qubits_per_group": qubits_per_group,
                        "interaction": interaction,
                        "depth": depth,
                        "optimizer": optimizer_name,
                        "seed": seed,
                        "budget": budget,
                        "hypervolume": float("nan"),
                        "wall_clock_s": float("nan"),
                        "failed": True,
                    }
                )
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
    fig.savefig(output_dir / "depth_qubit_grid.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke-scale grid")
    parser.add_argument("--output-dir", default="results/depth_qubit_grid")
    args = parser.parse_args()

    scale = QUICK_SCALE if args.quick else FULL_SCALE
    _logger.info(
        "running depth x qubit grid (%s scale): %s", "quick" if args.quick else "full", scale
    )

    df = run_sweep(
        n_groups=scale["n_groups"],
        qubit_counts=scale["qubit_counts"],
        depths=scale["depths"],
        interactions=scale["interactions"],
        budget=scale["budget"],
        init_points=scale["init_points"],
        n_seeds=scale["n_seeds"],
        n_candidates=scale["n_candidates"],
        mc_samples=scale["mc_samples"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "depth_qubit_grid_results.csv"
    df.to_csv(csv_path, index=False)
    _logger.info("wrote %s (%d rows)", csv_path, len(df))

    make_plots(df, output_dir)
    _logger.info("wrote plots to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

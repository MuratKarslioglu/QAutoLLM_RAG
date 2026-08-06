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

import ctypes
import gc
import os

# Must be set before numpy/PennyLane get imported anywhere below (directly or
# via qautollm.kernels.quantum) -- glibc decides its arena count at first
# malloc, not reconfigurable afterwards. mp.get_context("spawn") launches a
# fresh interpreter that inherits this from os.environ, so the worker gets it
# too. Caps the per-thread arena proliferation that otherwise compounds the
# fragmentation described below (2026-08-04, this machine's 31GB-RAM/48GB-
# Brief-target mismatch made the existing 512MB kernel cache + per-combo
# worker recycling insufficient -- see HARDWARE_SCALE and _TRIM_EVERY below).
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import argparse
import itertools
import multiprocessing as mp
import resource
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
from qautollm.utils.profiling import lightning_gpu_available, progress, ram_probe

_logger = get_logger(__name__)

# 25 rather than 20: GPU_SCALE's ceiling (see below). Just a validation cap
# passed to make_qubit_sweep_space -- doesn't force any qubit_counts value,
# each scale's own qubit_counts tuple still controls what actually runs.
MAX_QUBITS_OVERRIDE = 25

# Every _TRIM_EVERY BO iterations inside run_one(), force glibc to hand
# fragmented arenas back to the OS (see the malloc_trim call in run_one()).
# The existing maxtasksperchild=1 worker recycling only reclaims memory
# *between* combos; the 2026-08-04 OOM kill happened 57 minutes into combo
# 6/80, i.e. inside a single qubit_count=20 combo's 100-iteration loop, which
# that recycling never touches.
_TRIM_EVERY = 10

# Fraction of total host RAM a single sweep worker is allowed to claim
# (resource.RLIMIT_AS, set in run_one() since that's what actually runs
# inside the spawned worker process). Deliberately conservative: this
# session's other apps (VS Code, browser, NVIDIA tooling) need the rest, and
# a worker hitting this cap should fail fast with a catchable MemoryError
# instead of the kernel's OOM killer picking an unrelated victim (that's what
# was killing VS Code, not this process itself).
_WORKER_RAM_FRACTION = 0.6

FULL_SCALE = {
    "n_groups": 3,
    "qubit_counts": (5, 10, 15, 20),
    "interactions": (0.0, 1.0),
    "budget": 100,
    "init_points": 10,
    "n_seeds": 5,
    "n_candidates": 256,
    "mc_samples": 128,
    "device_name": "lightning.qubit",
}

# Brief's target machine (QAutoLLM_Implementation_Brief.md:23) is a 48GB-RAM
# laptop; FULL_SCALE's 100-iteration x 256-candidate BO loop at
# qubit_count=20 was sized for that headroom. On a host with materially less
# available RAM, halve the two knobs that directly multiply statevector
# churn per iteration (n_candidates, mc_samples) and trim the seed/budget
# grid.
#
# qubit_count=20 is dropped from this tier entirely, not just shrunk: even
# after fixing the 20 GiB single-allocation bug in quantum.py (GPyTorch
# tiling the training set across a candidate batch), a controlled
# micro-benchmark on the 31GB-RAM/8GB-VRAM desktop this was tuned against
# (2026-08-04/05) still showed worker RSS climbing steadily to ~13 GiB over
# a single ~15-iteration combo *independent of n_candidates/mc_samples*
# (32/16 grew just as fast as 128/64) before the RLIMIT_AS safety net
# stepped in. That points at fit_independent_gps() building 9 fresh
# QuantumFidelityKernel instances per BO iteration (3 objectives x
# n_groups=3), each re-fit via up to 50 L-BFGS-B steps against a growing
# training set -- a cost that scales with iteration count and qubit count
# (2**20 complex128 per statevector), not with how many candidates are
# scored. Qubit_count=15 (2**15, 32x smaller per statevector) is still in
# this tier since the same growth pattern at that size is proportionally
# much smaller. Reaching 20 qubits on this class of hardware needs --full
# (explicit opt-in; the RLIMIT_AS/malloc_trim protections still apply, so a
# combo that doesn't fit fails on its own rather than taking the session
# down).
HARDWARE_SCALE = {
    "n_groups": 3,
    "qubit_counts": (5, 10, 15),
    "interactions": (0.0, 1.0),
    "budget": 60,
    "init_points": 10,
    "n_seeds": 3,
    "n_candidates": 128,
    "mc_samples": 64,
    "device_name": "lightning.qubit",
}

# GPU tier (2026-08-06): measured on this machine's RTX 5060 via
# pennylane-lightning-gpu (cuStateVec) against the *exact* circuit this
# script runs (depth=1, RY+CZ chain), sequential single-point calls -- the
# actual access pattern here, not a batch (PennyLane's parameter
# broadcasting was measured *slower* than sequential on both CPU and GPU
# for this circuit, so it isn't used):
#   qubits   statevector    lightning.qubit(CPU)   lightning.gpu    speedup
#   20       16 MiB         15.0 ms/call            6.3 ms/call      2.4x
#   22       64 MiB         150.5 ms/call          35.3 ms/call      4.3x
#   23       128 MiB        346.6 ms/call          67.0 ms/call      5.2x
#   25       512 MiB        1543.5 ms/call         275.8 ms/call     5.6x
# GPU's relative advantage grows with qubit count (cuStateVec's ~constant
# per-call overhead vs CPU's exponentially growing compute) -- but qubit_
# counts stops at 20, not 25, despite that trend: the result still lands
# back in host RAM as a plain numpy array either way (confirmed
# empirically: GPU VRAM returns to baseline right after each call), so
# lightning.gpu changes *speed*, not where memory pressure comes from. And
# at qubit=25 that memory requirement isn't just candidate-count-driven
# (which n_candidates/mc_samples below already address) -- it's the
# *training* Gram matrix itself: computing it needs every training point's
# statevector alive simultaneously (512 MiB each at 25 qubits), so even the
# 8-point initial design alone demanded a single 4 GiB array, before a
# single BO iteration ran (observed 2026-08-06, RLIMIT_AS caught it
# cleanly, but no n_candidates/budget tuning fixes this -- it scales with
# *training set size*, i.e. init_points + elapsed iterations, not
# candidates scored). 20 qubits keeps this safe (16 MiB/statevector, so
# even a 30-point training set is under 500 MiB) while still getting the
# full GPU speedup; init_points/budget/n_candidates below are still cut
# relative to HARDWARE_SCALE as a further margin. Validated empirically
# (controlled RSS micro-benchmark through the same Pool(initializer=
# _limit_worker_address_space) + periodic _malloc_trim() path FULL_SCALE/
# HARDWARE_SCALE use) before being trusted here -- the RLIMIT_AS safety net
# applies exactly the same way regardless of device, just with a higher
# floor for GPU (see _MIN_GPU_WORKER_RAM_LIMIT_GB below: cuStateVec's own
# context init needs virtual-address headroom independent of this).
#
# Even so, qubit_count=20 combos here can still hit the RLIMIT_AS cap and
# get marked failed (observed 2026-08-06: virtual address space -- not
# RSS, which stayed ~4 GiB -- crept up to the cap over ~20 iterations,
# most likely cuStateVec's own host-side pinned-memory pool, which
# _malloc_trim() can't touch since that's CUDA-driver-managed, not
# glibc's arena allocator). Accepted deliberately rather than shrinking
# n_candidates/budget further: a combo that doesn't fit fails safely and
# in isolation (see run_sweep()'s except clause) exactly like the CPU
# path, so this is a per-combo data-completeness trade-off, not a
# session-safety one.
GPU_SCALE = {
    "n_groups": 3,
    "qubit_counts": (5, 10, 15, 20),
    "interactions": (0.0, 1.0),
    "budget": 20,
    "init_points": 8,
    "n_seeds": 3,
    "n_candidates": 24,
    "mc_samples": 16,
    "device_name": "lightning.gpu",
}

# Below this available-RAM threshold, auto-select HARDWARE_SCALE over
# FULL_SCALE (see select_scale()). Chosen well below the Brief's 48GB target
# but above what a quick free-RAM blip should false-trigger on.
_LOW_RAM_THRESHOLD_GB = 40.0

QUICK_SCALE = {
    "n_groups": 3,
    "qubit_counts": (5, 10),
    "interactions": (0.0, 1.0),
    "budget": 15,
    "init_points": 6,
    "n_seeds": 2,
    "n_candidates": 64,
    "mc_samples": 32,
    "device_name": "lightning.qubit",
}


def select_scale(*, quick: bool, force_full: bool, force_gpu: bool = False) -> dict:
    """Pick QUICK_SCALE / GPU_SCALE / HARDWARE_SCALE / FULL_SCALE for this host.

    ``--full`` (``force_full=True``) always wins and reproduces the
    original, Brief-target-machine behavior verbatim (CPU, qubit_counts up
    to 20) -- an explicit opt-in for a host the operator already knows can
    take it.

    GPU_SCALE is opt-in only via ``--gpu`` (``force_gpu=True``), *not*
    auto-selected just because a working ``lightning.gpu`` device exists.
    It was auto-selected in an earlier version of this script; that was
    reverted (2026-08-06) after real GPU_SCALE runs (budget=20,
    n_candidates=24 -- FULL_SCALE-proportioned, not a toy) showed
    pennylane-lightning-gpu/cuStateVec failing on effectively every
    qkernel_bo combo regardless of qubit count, including the smallest (5
    qubits). Isolated reproduction (bare device construction/circuit
    execution up to 5000x, the exact real run_one() call path, retrying
    each combo up to 3x in a fresh worker, a fresh Pool per combo) never
    reproduced a single failure at *toy* scale (budget<=3) -- only the real
    workload volume triggers it, deterministically, every time. Not a
    tunable capacity limit; a genuine instability in this specific
    driver/library stack at production BO scale. A user who wants to
    retry GPU anyway (e.g. after a driver update) can still force it.
    """
    if quick:
        return QUICK_SCALE
    if force_full:
        return FULL_SCALE
    if force_gpu:
        if not lightning_gpu_available():
            raise RuntimeError("--gpu was passed but lightning.gpu is not available on this host")
        _logger.warning(
            "--gpu forced: GPU_SCALE is known to fail on effectively every qkernel_bo combo "
            "at this scale on at least one tested host (2026-08-06) -- proceeding anyway since "
            "it was explicitly requested"
        )
        return GPU_SCALE
    available_gb = ram_probe()["available_gb"]
    if available_gb < _LOW_RAM_THRESHOLD_GB:
        _logger.info(
            "detected %.1f GB available RAM (< %.0f GB threshold): using HARDWARE_SCALE "
            "instead of FULL_SCALE (pass --full to override)",
            available_gb,
            _LOW_RAM_THRESHOLD_GB,
        )
        return HARDWARE_SCALE
    return FULL_SCALE


class _ExtendedQKernelBO(KernelBO):
    """Same ``KernelBO`` machinery as the real ``qkernel_bo`` -- deliberately
    NOT registered in ``OPTIMIZERS`` (this must never be reachable from a
    real experiment config; it exists only for this script). The only
    difference from production ``QKernelBO`` is the two overrides passed
    into ``quantum_group_kernel_factory``.
    """

    def __init__(
        self, search_space, seed, init_design, max_qubits, device_name, n_candidates, mc_samples
    ):
        super().__init__(
            search_space, seed, init_design, n_candidates=n_candidates, mc_samples=mc_samples
        )
        self.group_kernel_factory = quantum_group_kernel_factory(
            depth=1, shots=None, seed=seed, max_qubits=max_qubits, device_name=device_name
        )


# CUDA context construction (cuStateVec's DevicePool init) reserves a large
# chunk of *virtual* address space up front for its own unified-addressing
# bookkeeping -- independent of actual VRAM/RSS used -- before any circuit
# runs. Measured empirically on this host (2026-08-06): the real worker
# import chain (torch/gpytorch/botorch/qautollm + pennylane-lightning-gpu)
# needs ~20 GiB of RLIMIT_AS headroom just to get past `qml.device(
# "lightning.gpu", ...)`, even though actual resident memory during a GPU
# combo stays small (VRAM returns to ~280 MiB baseline right after each
# circuit call -- see GPU_SCALE's comment). The plain RAM-fraction cap
# below (18.67 GiB on this 31GB host) is *below* that and made
# lightning.gpu fail at device construction, before running anything.
_MIN_GPU_WORKER_RAM_LIMIT_GB = 25.0

# pennylane-lightning-gpu tags every one of its own C++-level failures with
# this exact substring when surfacing them as a Python RuntimeError --
# confirmed (2026-08-06) across three unrelated internal failure sites with
# three different messages ("...Error in PennyLane Lightning: out of
# memory", "...: cuBLAS memory allocation failed", "...: custatevec not
# initialized" -- VRAM exhaustion, cuBLAS allocation failure, and a
# cuStateVec context/handle failure respectively, the last one *not*
# even memory-related). Matching on the library's own tag rather than an
# ever-growing list of specific phrasings: it reliably separates "the GPU
# library hit some internal capacity/driver limit" from "a real bug in our
# own Python code" (which would raise from *our* files, without this tag),
# without needing to special-case every new C++ exception message lightning-
# gpu happens to use.
_LIGHTNING_GPU_ERROR_TAG = "error in pennylane lightning"

# A combo hitting _LIGHTNING_GPU_ERROR_TAG has been observed to succeed on a
# plain retry (2026-08-06): the same combo, same code, same host, run in
# isolation always succeeded, while a full ~48-combo GPU_SCALE sweep hit it
# on effectively every qkernel_bo combo -- extensive isolated reproduction
# attempts (spawn+Pool alone, any RLIMIT_AS value, the full torch/gpytorch/
# botorch import chain, up to 200 device constructions and 5000 circuit
# executions in one process, the exact real run_one() call path both
# directly and through the real Pool) never reproduced a single failure,
# pointing at transient NVIDIA-driver/cuStateVec instability under sustained
# load rather than a deterministic capacity limit our own code controls.
# maxtasksperchild=1 already means a retry runs in a brand-new worker
# process (and thus a brand-new CUDA context) automatically, which is
# exactly the kind of fresh start this class of driver flakiness tends to
# recover from.
_GPU_ERROR_MAX_RETRIES = 3


def _apply_with_gpu_retries(pool, args: tuple) -> dict:
    """``pool.apply(run_one, args)``, retrying up to ``_GPU_ERROR_MAX_RETRIES``
    times if lightning.gpu raises its own tagged RuntimeError (see
    ``_LIGHTNING_GPU_ERROR_TAG``'s comment for why retrying is the right
    response to *this specific* failure, unlike the CPU-side MemoryError,
    which is deterministic given the combo's data size and reraised
    immediately by the caller instead).
    """
    last_exc: RuntimeError | None = None
    for attempt in range(1, _GPU_ERROR_MAX_RETRIES + 1):
        try:
            return pool.apply(run_one, args)
        except RuntimeError as exc:
            if _LIGHTNING_GPU_ERROR_TAG not in str(exc).lower():
                raise
            last_exc = exc
            if attempt < _GPU_ERROR_MAX_RETRIES:
                _logger.warning(
                    "lightning.gpu error on attempt %d/%d (retrying in a fresh worker): %s",
                    attempt,
                    _GPU_ERROR_MAX_RETRIES,
                    str(exc)[:200],
                )
    assert last_exc is not None
    raise last_exc


def _compute_worker_ram_limit_bytes(device_name: str) -> int:
    """``_WORKER_RAM_FRACTION`` of total host RAM, in bytes -- floored at
    ``_MIN_GPU_WORKER_RAM_LIMIT_GB`` for GPU devices (see above).

    Split out from :func:`_limit_worker_address_space` so the arithmetic is
    unit-testable without actually mutating any process's rlimits (which,
    once lowered, a non-root process can't raise back).
    """
    total_gb = ram_probe()["total_gb"]
    limit_gb = total_gb * _WORKER_RAM_FRACTION
    if device_name != "lightning.qubit":
        limit_gb = max(limit_gb, _MIN_GPU_WORKER_RAM_LIMIT_GB)
    return int(limit_gb * 1024**3)


def _limit_worker_address_space(device_name: str) -> None:
    """Cap this process's address space (see :func:`_compute_worker_ram_limit_bytes`).
    Passed as ``Pool(initializer=..., initargs=(device_name,))`` in
    run_sweep() -- it must run exactly once, in the disposable spawned
    worker process, and *never* in whatever process calls run_one()
    directly (e.g. a test, or a future caller): RLIMIT_AS can only be
    lowered, never raised back, by a non-root process, so applying this to
    a long-lived process such as the test runner would permanently wound
    it.

    A runaway combo then fails with a catchable ``MemoryError`` (CPU) or
    ``RuntimeError`` (GPU, lightning.gpu's own internal capacity/driver
    limits -- see ``_LIGHTNING_GPU_ERROR_TAG``) instead of growing until
    the kernel's OOM killer steps in and picks a victim from across the
    whole login session (that's what was killing VS Code, not this
    process) -- see run_sweep()'s try/except around pool.apply().
    """
    limit_bytes = _compute_worker_ram_limit_bytes(device_name)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, resource.RLIM_INFINITY))


def _malloc_trim() -> None:
    """Force glibc to hand fragmented-but-freed arenas back to the OS.

    gc.collect() only makes Python drop references; it doesn't touch what
    glibc's allocator decides to keep. malloc_trim(0) is the actual fix for
    the RSS-ratchets-up-and-never-comes-down behavior documented in
    run_sweep() -- called periodically here because that growth happens
    *within* a single combo's iteration loop, not just between combos.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # non-glibc platform (e.g. macOS) -- nothing to trim


def run_one(
    n_groups: int,
    qubits_per_group: int,
    interaction: float,
    optimizer_name: str,
    seed: int,
    budget: int,
    init_points: int,
    n_candidates: int,
    mc_samples: int,
    device_name: str,
) -> dict:
    space = make_qubit_sweep_space(n_groups, qubits_per_group, max_qubits=MAX_QUBITS_OVERRIDE)
    objective_fn = build_synthetic_objectives(space, interaction, seed=seed)
    init_design = [encode(raw, space) for raw in space.sobol_design(init_points, seed=seed)]

    if optimizer_name == "qkernel_bo":
        optimizer = _ExtendedQKernelBO(
            space,
            seed,
            init_design,
            max_qubits=MAX_QUBITS_OVERRIDE,
            device_name=device_name,
            n_candidates=n_candidates,
            mc_samples=mc_samples,
        )
    else:
        # Same n_candidates/mc_samples as the qkernel_bo branch above: the
        # sweep compares hypervolume *between* optimizers at a given scale,
        # so both must see the same BO budget per iteration or the
        # comparison stops being apples-to-apples.
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
        if (i + 1) % _TRIM_EVERY == 0:
            _malloc_trim()
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
        "failed": False,
    }


def run_sweep(
    n_groups: int,
    qubit_counts: tuple[int, ...],
    interactions: tuple[float, ...],
    budget: int,
    init_points: int,
    n_seeds: int,
    n_candidates: int,
    mc_samples: int,
    device_name: str,
) -> pd.DataFrame:
    optimizer_names = ("structured_bo", "qkernel_bo")
    combos = list(itertools.product(qubit_counts, interactions, optimizer_names, range(n_seeds)))
    # expensive (high qubit_count) combos first so a partial run still
    # covers the full qubit range at lower seed counts if interrupted.
    combos.sort(key=lambda c: -c[0])
    rows = []
    # Each combo builds fresh PennyLane devices/statevector caches (up to
    # 2**20 complex128 entries at qubit_count=20) in rapid, irregular-sized
    # bursts across its own 100-iteration BO budget. glibc's allocator
    # fragments under that pattern and never hands the freed arenas back to
    # the OS, so RSS ratchets up across combos even though every Python
    # object involved is correctly garbage-collected -- eventually large
    # enough to trip the system's OOM killer (observed: ~25GB after only
    # ~5 combos, taking down whatever cgroup was biggest, e.g. the
    # terminal). maxtasksperchild=1 recycles the worker process after every
    # combo so the OS reclaims the fragmented memory between combos; the
    # periodic _malloc_trim() inside run_one() additionally reclaims memory
    # *within* a combo (necessary since the 2026-08-04 OOM happened inside a
    # single combo's iteration loop, before this per-combo recycling had a
    # chance to matter).
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=1,
        maxtasksperchild=1,
        initializer=_limit_worker_address_space,
        initargs=(device_name,),
    ) as pool:
        for qubits_per_group, interaction, optimizer_name, seed in progress(
            combos, desc="extended qubit sweep", total=len(combos)
        ):
            args = (
                n_groups,
                qubits_per_group,
                interaction,
                optimizer_name,
                seed,
                budget,
                init_points,
                n_candidates,
                mc_samples,
                device_name,
            )
            try:
                rows.append(_apply_with_gpu_retries(pool, args))
            except (MemoryError, RuntimeError) as exc:
                # _limit_worker_address_space()'s RLIMIT_AS tripped inside the
                # worker (MemoryError, CPU) -- deterministic given this
                # combo's actual data size, so not retried (see
                # _apply_with_gpu_retries for the lightning.gpu case, which
                # *is* retried, being transient rather than data-size-driven).
                # Or lightning.gpu's underlying C++ library kept hitting some
                # internal capacity/driver limit even after retrying
                # (RuntimeError tagged with _LIGHTNING_GPU_ERROR_TAG). Either
                # way that combo alone is infeasible on this host/run, but it
                # must not take the rest of the sweep (or, worse, the whole
                # login session/VS Code) down with it. Any *other*
                # RuntimeError -- lacking the tag, so raised by our own code,
                # not lightning-gpu's -- is a real bug and must not be
                # swallowed.
                if isinstance(exc, RuntimeError) and _LIGHTNING_GPU_ERROR_TAG not in str(
                    exc
                ).lower():
                    raise
                _logger.warning(
                    "combo qubits=%d interaction=%s optimizer=%s seed=%d exceeded the "
                    "worker memory cap (%s) -- marking failed and continuing",
                    qubits_per_group,
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
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "force FULL_SCALE (the Brief's 48GB-RAM-target, CPU-only parameters, "
            "including qubit_count=20) instead of auto-selecting the reduced, "
            "20-qubit-excluded HARDWARE_SCALE"
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "force GPU_SCALE (lightning.gpu) instead of auto-selecting a CPU tier. "
            "WARNING: on at least one tested host (RTX 5060, driver 595.71, "
            "pennylane-lightning-gpu 0.45, 2026-08-06), GPU_SCALE at real BO scale "
            "(budget=20, n_candidates=24) failed on effectively every qkernel_bo "
            "combo regardless of qubit count -- not a tunable capacity limit, an "
            "apparent driver/library instability. Per-circuit speed alone (2.4x-5.6x, "
            "measured) was real, but no full sweep has completed successfully with it."
        ),
    )
    parser.add_argument("--output-dir", default="results/qubit_sweep_extended")
    args = parser.parse_args()

    scale = select_scale(quick=args.quick, force_full=args.full, force_gpu=args.gpu)
    _logger.info(
        "running EXTENDED qubit sweep (beyond Brief D3's 5-qubit cap, scale=%s): %s",
        "quick" if args.quick else ("full (forced)" if args.full else (
            "gpu (forced)" if args.gpu else "auto"
        )),
        scale,
    )

    df = run_sweep(
        n_groups=scale["n_groups"],
        qubit_counts=scale["qubit_counts"],
        interactions=scale["interactions"],
        budget=scale["budget"],
        init_points=scale["init_points"],
        n_seeds=scale["n_seeds"],
        n_candidates=scale["n_candidates"],
        mc_samples=scale["mc_samples"],
        device_name=scale["device_name"],
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

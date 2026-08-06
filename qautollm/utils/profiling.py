"""Timing context managers, VRAM probes, and the one progress-bar helper
used by every long-running loop in the codebase (Brief §4).

Rule: any loop expected to run more than a couple of seconds (corpus
indexing, the optimization budget loop, batched generation, kernel Gram
matrix construction, synthetic sweeps) must be wrapped in :func:`progress`
so there is always a visible ETA and current position. This is a display
concern only — it must never influence cached results or control flow
(Brief §D4: no wall-clock-dependent logic in results).
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from tqdm import tqdm

from qautollm.utils.logging import get_logger

_logger = get_logger(__name__)

T = TypeVar("T")


@contextmanager
def timer(name: str) -> Iterator[dict[str, float]]:
    """Time a block; ``result["seconds"]`` is filled in after the block exits.

    Usage:
        with timer("build_index") as t:
            ...
        t["seconds"]  # elapsed wall-clock time
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["seconds"] = elapsed
        _logger.info("%s took %.3fs", name, elapsed)


def progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
) -> Iterable[T]:
    """Wrap ``iterable`` with a progress bar showing elapsed time, ETA, and rate.

    Every training/evaluation/optimization loop in the codebase should
    iterate through this instead of a bare ``for`` loop, so a human watching
    the terminal always sees current position, total, and estimated time
    remaining.
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.5,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )


def vram_probe() -> dict[str, Any]:
    """Report current CUDA VRAM usage, or ``{"available": False}`` without CUDA.

    Values are MiB. ``allocated`` is live tensor memory; ``reserved`` is the
    cache PyTorch holds on to (D1: this is what must stay under the single
    resident model's budget).
    """
    try:
        import torch
    except ImportError:
        return {"available": False}

    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "total_mb": torch.cuda.get_device_properties(0).total_memory / (1024**2),
    }


def ram_probe() -> dict[str, float]:
    """Host RAM (GB), analogous to :func:`vram_probe` for the GPU side.

    ``available_gb`` (not ``total_gb``) is what callers should budget
    against: it already accounts for buffers/cache the kernel will
    reclaim under pressure, matching what ``free -h``'s "available" column
    reports.
    """
    import psutil

    vm = psutil.virtual_memory()
    return {
        "total_gb": vm.total / (1024**3),
        "available_gb": vm.available / (1024**3),
    }


def lightning_gpu_available() -> bool:
    """Whether PennyLane's ``lightning.gpu`` (cuStateVec-backed) device can
    actually be constructed on this host -- used by
    ``scripts.run_qubit_sweep_extended`` to auto-select a GPU-accelerated
    scale instead of failing at device-construction time inside a worker
    process. Cheap: constructing a 1-wire device just checks the CUDA/
    cuStateVec runtime is loadable, it doesn't allocate anything sized by
    the caller's real circuit.
    """
    try:
        import pennylane as qml

        qml.device("lightning.gpu", wires=1)
        return True
    except Exception:  # noqa: BLE001 - any import/runtime failure means "not available"
        return False


def disk_probe(path: str | Path = ".") -> dict[str, float]:
    """Free/used/total disk space (GB) for the filesystem holding ``path``.

    Model downloads fail deep inside huggingface_hub with a cryptic Rust-level
    IO error if the disk fills up mid-download (seen in practice: a 6GB+
    generator download crashing at 42% after a multi-GB-consuming
    build_index pass already ran) — callers about to download something
    large should check this first and fail fast with an actionable message
    instead.
    """
    usage = shutil.disk_usage(path)
    return {
        "free_gb": usage.free / (1024**3),
        "used_gb": usage.used / (1024**3),
        "total_gb": usage.total / (1024**3),
    }

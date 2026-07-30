#!/usr/bin/env python3
"""Print CUDA availability, VRAM, and key package versions (Brief §3).

Run this right after `pip install -r requirements.txt` and before anything
else — it is the first gate: if this doesn't report a CUDA build of torch
and >=8GB reported VRAM, nothing downstream will fit the hardware budget
(Brief §2, D1).
"""

from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "sentence_transformers",
    "faiss",
    "rank_bm25",
    "datasets",
    "optuna",
    "pymoo",
    "gpytorch",
    "botorch",
    "pennylane",
    "pydantic",
]


def _version_of(module_name: str) -> str:
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        return f"NOT INSTALLED ({e})"
    return getattr(mod, "__version__", "unknown")


def main() -> int:
    print(f"python: {sys.version.split()[0]} ({platform.python_version()})")
    print(f"platform: {platform.platform()}")
    print()

    print("--- package versions ---")
    for name in PACKAGES:
        print(f"{name:24s} {_version_of(name)}")
    print()

    print("--- CUDA / GPU ---")
    try:
        import torch
    except ImportError:
        print("torch not installed — cannot check CUDA.")
        return 1

    print(f"torch.__version__:     {torch.__version__}")
    print(f"torch cuda build:      {torch.version.cuda}")
    print(f"cuda available:        {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"device count:          {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            print(f"  [{i}] {props.name}: {total_gb:.2f} GB total VRAM")
        if torch.cuda.get_device_properties(0).total_memory / (1024**3) < 7.5:
            print(
                "WARNING: reported VRAM is below the 8GB target in Brief §2 — "
                "double check hardware.max_vram_gb budgets in experiment configs."
            )
    else:
        print(
            "WARNING: CUDA not available. If this is unexpected, torch was "
            "likely installed from PyPI's default (CPU-only) index — reinstall "
            "with e.g. `pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121` (see README)."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

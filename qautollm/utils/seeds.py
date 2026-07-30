"""Single entry point for determinism (Brief §2, §D4).

Every script under ``scripts/`` must call :func:`seed_everything` before
doing anything else. Nothing else in the codebase should call
``random.seed``, ``np.random.seed``, or ``torch.manual_seed`` directly.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed every RNG source used anywhere in the codebase, and force
    deterministic (non-benchmarked) backends where a choice exists.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_rng(seed: int) -> np.random.Generator:
    """Return an independent, seeded numpy Generator for a specific caller.

    Prefer this over the global ``np.random`` state inside library code so
    that two independent sampling streams (e.g. Sobol init design vs.
    optimizer-internal sampling) never perturb each other's sequence.
    """
    return np.random.default_rng(seed)

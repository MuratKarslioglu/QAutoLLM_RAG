"""Importing this package registers every optimizer into
``qautollm.core.registry.OPTIMIZERS`` — the ``@OPTIMIZERS.register(...)``
decorators only run when their module is actually imported, so anything
that resolves an optimizer by name (the orchestrator) must go through here
first rather than importing a specific optimizer module directly.
"""

from qautollm.optimizers.nsga2 import NSGA2Optimizer
from qautollm.optimizers.qkernel_bo import QKernelBO
from qautollm.optimizers.random_search import RandomSearchOptimizer
from qautollm.optimizers.structured_bo import KernelBO, StructuredBO
from qautollm.optimizers.tpe import TPEOptimizer

__all__ = [
    "KernelBO",
    "NSGA2Optimizer",
    "QKernelBO",
    "RandomSearchOptimizer",
    "StructuredBO",
    "TPEOptimizer",
]

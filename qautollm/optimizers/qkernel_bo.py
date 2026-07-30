"""Quantum-kernel BO (Brief §5.4, §9 P5).

Subclasses the same ``KernelBO`` that ``structured_bo.py`` does — every
line of GP-fitting/EHVI/archive/resume logic is inherited unchanged,
``__init__`` only swaps in the quantum ``group_kernel_factory`` (bound with
``optimizer.quantum.shots``/``feature_map_depth`` from the experiment YAML,
Brief §7). This is the enforcement of Brief §5.4's "structured_bo and
qkernel_bo must differ only in which GroupKernel factory they pass" — see
``tests/test_kernel_bo_parity.py`` for the assertion that checks it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qautollm.core.config import QuantumConfig
from qautollm.core.registry import OPTIMIZERS
from qautollm.kernels.quantum import quantum_group_kernel_factory
from qautollm.optimizers.structured_bo import KernelBO
from qautollm.space.encoding import CanonicalConfig
from qautollm.space.search_space import SearchSpace

if TYPE_CHECKING:
    from qautollm.core.config import OptimizerConfig


@OPTIMIZERS.register("qkernel_bo")
class QKernelBO(KernelBO):
    def __init__(
        self,
        search_space: SearchSpace,
        seed: int,
        init_design: list[CanonicalConfig],
        optimizer_config: OptimizerConfig | None = None,
        n_candidates: int = 256,
        mc_samples: int = 128,
    ) -> None:
        super().__init__(
            search_space, seed, init_design, optimizer_config, n_candidates, mc_samples
        )
        quantum_cfg = optimizer_config.quantum if optimizer_config is not None else QuantumConfig()
        self.group_kernel_factory = quantum_group_kernel_factory(
            depth=quantum_cfg.feature_map_depth, shots=quantum_cfg.shots, seed=seed
        )

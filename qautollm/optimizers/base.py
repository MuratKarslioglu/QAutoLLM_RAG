"""OptimizerInterface (Brief §5.2): every optimizer variant — random, tpe,
structured_bo, qkernel_bo — implements exactly this.

The shared ``__init__``/``_next_init_point`` here is what makes the "one
shared Sobol init design per seed, passed identically to every optimizer"
pairing guarantee (Brief §5.2) automatic: every subclass drains the same
``init_design`` in the same order before doing anything strategy-specific,
so every optimizer variant sees the exact same first N points at a given
seed.

``optimizer_config`` is an extension beyond Brief §5.2's literal 3-arg
signature: the orchestrator constructs every optimizer uniformly
(``cls(search_space, seed, init_design, optimizer_config=...)``), and only
``qkernel_bo`` actually reads anything from it (``optimizer.quantum.shots``/
``feature_map_depth``, Brief §7) — everyone else ignores it. Optional and
defaulted to ``None`` so it doesn't disturb random/tpe/structured_bo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from qautollm.space.encoding import CanonicalConfig
from qautollm.space.search_space import SearchSpace

if TYPE_CHECKING:
    from qautollm.core.config import OptimizerConfig


class OptimizerInterface(ABC):
    def __init__(
        self,
        search_space: SearchSpace,
        seed: int,
        init_design: list[CanonicalConfig],
        optimizer_config: OptimizerConfig | None = None,
    ) -> None:
        self.search_space = search_space
        self.seed = seed
        self._init_design = list(init_design)
        self._init_cursor = 0
        self.optimizer_config = optimizer_config

    def _next_init_point(self) -> CanonicalConfig | None:
        if self._init_cursor < len(self._init_design):
            cfg = self._init_design[self._init_cursor]
            self._init_cursor += 1
            return cfg
        return None

    @abstractmethod
    def suggest(self) -> CanonicalConfig: ...

    @abstractmethod
    def update(self, config: CanonicalConfig, objectives: np.ndarray) -> None: ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None: ...

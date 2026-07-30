"""Pure random baseline (Brief §7's ``optimizer.name: random``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qautollm.core.registry import OPTIMIZERS
from qautollm.optimizers.base import OptimizerInterface
from qautollm.space.encoding import CanonicalConfig, encode
from qautollm.space.search_space import SearchSpace
from qautollm.utils.seeds import make_rng

if TYPE_CHECKING:
    from qautollm.core.config import OptimizerConfig


@OPTIMIZERS.register("random")
class RandomSearchOptimizer(OptimizerInterface):
    def __init__(
        self,
        search_space: SearchSpace,
        seed: int,
        init_design: list[CanonicalConfig],
        optimizer_config: OptimizerConfig | None = None,
    ) -> None:
        super().__init__(search_space, seed, init_design, optimizer_config)
        self._rng = make_rng(seed)

    def suggest(self) -> CanonicalConfig:
        init = self._next_init_point()
        if init is not None:
            return init
        raw = self.search_space.sample(self._rng)
        return encode(raw, self.search_space)

    def update(self, config: CanonicalConfig, objectives: np.ndarray) -> None:
        pass  # memoryless

    def state_dict(self) -> dict[str, Any]:
        return {"init_cursor": self._init_cursor, "rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._init_cursor = state["init_cursor"]
        self._rng.bit_generator.state = state["rng_state"]

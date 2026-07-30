"""NSGA-II baseline (Brief §9 P6, ``optimizer.name: nsga2``): pymoo wrapper.

pymoo's own ``Algorithm``/``Problem`` machinery manages a whole population
per generation internally and doesn't fit ``OptimizerInterface``'s one-
point-at-a-time ``suggest()``/``update()`` contract, and it has no native
notion of *conditional* variables (Brief §5.1). So this wraps pymoo's core
NSGA-II building blocks — fast non-dominated sorting and crowding-distance
survival (``pymoo.util.nds``, both in the same minimization convention as
Brief §5.5) — around a streaming generational loop, with sampling/
crossover/mutation implemented directly against ``SearchSpace`` so
conditional validity is guaranteed by construction rather than bolted on.

Streaming state machine: the shared init design seeds generation 0's
parent population (Brief §5.2's paired-comparison guarantee applies here
too). Once a full population of parents is evaluated, offspring are
generated and queued; once those are evaluated, parents+offspring survive
via non-dominated-sort + crowding-distance truncation into the next
generation's parents, and the cycle repeats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from qautollm.core.registry import OPTIMIZERS
from qautollm.optimizers.base import OptimizerInterface
from qautollm.space.encoding import CanonicalConfig, encode
from qautollm.space.search_space import SearchSpace
from qautollm.surrogate.pareto_archive import dominates
from qautollm.utils.seeds import make_rng

if TYPE_CHECKING:
    from qautollm.core.config import OptimizerConfig

_MIN_POPULATION = 4  # NSGA-II needs at least a few individuals for meaningful selection


@OPTIMIZERS.register("nsga2")
class NSGA2Optimizer(OptimizerInterface):
    def __init__(
        self,
        search_space: SearchSpace,
        seed: int,
        init_design: list[CanonicalConfig],
        optimizer_config: OptimizerConfig | None = None,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
    ) -> None:
        super().__init__(search_space, seed, init_design, optimizer_config)
        self._rng = make_rng(seed)
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate

        self._all_evaluated: dict[str, tuple[CanonicalConfig, np.ndarray]] = {}
        self._parents: list[CanonicalConfig] = []
        self._offspring: list[CanonicalConfig] | None = None
        self._pending_queue: list[CanonicalConfig] = []

    def suggest(self) -> CanonicalConfig:
        init = self._next_init_point()
        if init is not None:
            self._parents.append(init)
            return init
        if len(self._parents) < _MIN_POPULATION:
            cfg = encode(self.search_space.sample(self._rng), self.search_space)
            self._parents.append(cfg)
            return cfg
        if not self._pending_queue:
            self._advance()
        return self._pending_queue.pop(0)

    def update(self, config: CanonicalConfig, objectives: np.ndarray) -> None:
        self._all_evaluated[config.config_hash] = (config, np.asarray(objectives, dtype=np.float64))

    def _advance(self) -> None:
        if self._offspring is None:
            self._offspring = self._make_offspring(self._parents)
        else:
            combined = self._parents + self._offspring
            self._parents = self._survive(combined, len(self._parents))
            self._offspring = self._make_offspring(self._parents)
        self._pending_queue = list(self._offspring)

    def _objectives_of(self, configs: list[CanonicalConfig]) -> np.ndarray:
        return np.stack([self._all_evaluated[c.config_hash][1] for c in configs])

    def _survive(self, combined: list[CanonicalConfig], target_size: int) -> list[CanonicalConfig]:
        objectives = self._objectives_of(combined)
        fronts = NonDominatedSorting().do(objectives)

        survivors: list[CanonicalConfig] = []
        for front in fronts:
            if len(survivors) + len(front) <= target_size:
                survivors.extend(combined[i] for i in front)
                continue
            remaining = target_size - len(survivors)
            crowding = calc_crowding_distance(objectives[front])
            order = np.argsort(-crowding)  # most spread-out first
            survivors.extend(combined[i] for i in front[order[:remaining]])
            break
        return survivors

    def _tournament_select(
        self, parents: list[CanonicalConfig], objectives: np.ndarray
    ) -> CanonicalConfig:
        i, j = self._rng.integers(0, len(parents), size=2)
        if dominates(objectives[i], objectives[j]):
            return parents[i]
        if dominates(objectives[j], objectives[i]):
            return parents[j]
        return parents[i] if self._rng.random() < 0.5 else parents[j]

    def _make_offspring(self, parents: list[CanonicalConfig]) -> list[CanonicalConfig]:
        objectives = self._objectives_of(parents)
        offspring = []
        for _ in range(len(parents)):
            parent_a = self._tournament_select(parents, objectives)
            parent_b = self._tournament_select(parents, objectives)
            child_raw = self._make_child_raw(parent_a.raw, parent_b.raw)
            offspring.append(encode(child_raw, self.search_space))
        return offspring

    def _make_child_raw(self, raw_a: dict[str, Any], raw_b: dict[str, Any]) -> dict[str, Any]:
        if self._rng.random() >= self._crossover_rate:
            raw_b = raw_a  # no crossover this time: child derives from parent_a alone

        child: dict[str, Any] = {}
        for v in self.search_space.variables:
            if v.is_conditional:
                continue
            value = raw_a[v.name] if self._rng.random() < 0.5 else raw_b[v.name]
            if self._rng.random() < self._mutation_rate:
                value = v.sample(self._rng)
            child[v.name] = value

        for v in self.search_space.variables:
            if not v.is_conditional or not v.condition.is_active(child):
                continue
            candidates = [raw[v.name] for raw in (raw_a, raw_b) if v.name in raw]
            value = (
                candidates[int(self._rng.integers(0, len(candidates)))]
                if candidates
                else v.sample(self._rng)
            )
            if self._rng.random() < self._mutation_rate:
                value = v.sample(self._rng)
            child[v.name] = value
        return child

    def state_dict(self) -> dict[str, Any]:
        return {
            "init_cursor": self._init_cursor,
            "rng_state": self._rng.bit_generator.state,
            "all_evaluated": {
                h: {"raw": cfg.raw, "objectives": obj.tolist()}
                for h, (cfg, obj) in self._all_evaluated.items()
            },
            "parents": [c.raw for c in self._parents],
            "offspring": [c.raw for c in self._offspring] if self._offspring is not None else None,
            "pending_queue": [c.raw for c in self._pending_queue],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._init_cursor = state["init_cursor"]
        self._rng.bit_generator.state = state["rng_state"]
        self._all_evaluated = {}
        for entry in state["all_evaluated"].values():
            cfg = encode(entry["raw"], self.search_space)
            self._all_evaluated[cfg.config_hash] = (cfg, np.array(entry["objectives"]))
        self._parents = [encode(r, self.search_space) for r in state["parents"]]
        self._offspring = (
            [encode(r, self.search_space) for r in state["offspring"]]
            if state["offspring"] is not None
            else None
        )
        self._pending_queue = [encode(r, self.search_space) for r in state["pending_queue"]]

"""Classical-kernel BO (Brief §5.4).

``KernelBO`` is the shared GP+EHVI machinery; ``StructuredBO`` is a *thin*
subclass that only sets ``group_kernel_factory`` to the classical one
(:mod:`qautollm.kernels.classical`). ``qkernel_bo.py`` subclasses this same
``KernelBO`` and swaps in the quantum factory instead — enforcing Brief
§5.4's "structured_bo and qkernel_bo must differ only in which
GroupKernel factory they pass."
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from qautollm.core.registry import OPTIMIZERS
from qautollm.kernels.classical import make_classical_group_kernel
from qautollm.kernels.structural import GroupKernelFactory, StructuralKernel
from qautollm.optimizers.base import OptimizerInterface
from qautollm.space.encoding import CanonicalConfig, encode
from qautollm.space.search_space import SearchSpace
from qautollm.surrogate.acquisition import build_ehvi, select_best_candidate
from qautollm.surrogate.gp import augmented_features, fit_independent_gps
from qautollm.surrogate.pareto_archive import ParetoArchive
from qautollm.utils.seeds import make_rng

if TYPE_CHECKING:
    from qautollm.core.config import OptimizerConfig

_MIN_TRAINING_POINTS = 2  # below this, an EHVI-guided GP suggestion is meaningless


class KernelBO(OptimizerInterface):
    group_kernel_factory: GroupKernelFactory  # set by subclass (classical vs. quantum)

    def __init__(
        self,
        search_space: SearchSpace,
        seed: int,
        init_design: list[CanonicalConfig],
        optimizer_config: OptimizerConfig | None = None,
        n_candidates: int = 256,
        mc_samples: int = 128,
    ) -> None:
        super().__init__(search_space, seed, init_design, optimizer_config)
        self._rng = make_rng(seed)
        self._n_candidates = n_candidates
        self._mc_samples = mc_samples
        self._configs: list[CanonicalConfig] = []
        self._objectives: list[np.ndarray] = []
        self._seen_config_hashes: set[str] = set()
        self._archive = ParetoArchive()
        # Brief §8 kernel_log: populated on every EHVI-guided suggestion
        # (None/0.0 while still in the init-design/random-fallback phase).
        # "Kernel time" here is GP-fit time as a whole (dominated by kernel
        # evaluation for our tiny n) — the same granularity the Brief §9 P4
        # GP-fit-time gate already measures at, not a StructuralKernel-
        # internal micro-timer.
        self.last_kernel_time_s: float | None = None
        self.last_psd_clipped_mass: float = 0.0

    def suggest(self) -> CanonicalConfig:
        init = self._next_init_point()
        if init is not None:
            return init
        if len(self._configs) < _MIN_TRAINING_POINTS:
            raw = self.search_space.sample(self._rng)
            return encode(raw, self.search_space)
        return self._suggest_via_ehvi()

    def _suggest_via_ehvi(self) -> CanonicalConfig:
        # BoTorch's fit fallback can reach for torch's *global* RNG (e.g. for
        # randomized restarts after a failed optimization attempt) — pin it
        # deterministically here so a resumed process (fresh global RNG
        # state) reproduces the exact same fit as an uninterrupted one
        # (Brief D4). Caught by test_kernel_bo_parity.py's qkernel_bo resume
        # test failing before this was added.
        torch.manual_seed(self.seed)
        train_x = augmented_features(self._configs)
        train_y_max = -torch.tensor(np.stack(self._objectives), dtype=torch.float64)

        def _make_kernel() -> StructuralKernel:
            return StructuralKernel.from_search_space(self.search_space, self.group_kernel_factory)

        start = time.perf_counter()
        model = fit_independent_gps(train_x, train_y_max, _make_kernel)
        self.last_kernel_time_s = time.perf_counter() - start
        self.last_psd_clipped_mass = _max_psd_clipped_mass(model)

        ref_point = train_y_max.min(dim=0).values - 1e-3
        acqf = build_ehvi(model, train_y_max, ref_point, self._mc_samples)

        candidate_configs = [
            encode(self.search_space.sample(self._rng), self.search_space)
            for _ in range(self._n_candidates)
        ]
        candidate_features = augmented_features(candidate_configs).unsqueeze(1)  # q=1
        best_idx = select_best_candidate(acqf, candidate_features)
        return candidate_configs[best_idx]

    def update(self, config: CanonicalConfig, objectives: np.ndarray) -> None:
        # The orchestrator calls update() on every successful trial, including
        # re-suggested duplicates (Brief §5.2: a duplicate costs zero *budget*,
        # but nothing upstream stops it reaching here). Appending the same
        # config twice puts two identical rows in the GP's training X, which
        # makes the kernel Gram matrix exactly singular -- observed in
        # practice as BoTorch's `ModelFittingError: All attempts to fit the
        # model have failed` once training set size made a collision likely.
        # A duplicate is zero new information for the surrogate anyway, so
        # skip it rather than let it corrupt the fit.
        if config.config_hash in self._seen_config_hashes:
            return
        self._seen_config_hashes.add(config.config_hash)
        self._configs.append(config)
        self._objectives.append(np.asarray(objectives, dtype=np.float64))
        self._archive.update(objectives, tag=config.config_hash)

    def state_dict(self) -> dict[str, Any]:
        return {
            "init_cursor": self._init_cursor,
            "rng_state": self._rng.bit_generator.state,
            "raws": [c.raw for c in self._configs],
            "objectives": [o.tolist() for o in self._objectives],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._init_cursor = state["init_cursor"]
        self._rng.bit_generator.state = state["rng_state"]
        # Rebuild through update()'s own dedup rather than a bare list-comp:
        # checkpoints written before the duplicate-training-point fix above
        # may already contain an exact-duplicate config_hash, and loading it
        # verbatim would reintroduce the same singular Gram matrix on the
        # very next EHVI-guided suggestion. Re-filtering here lets an
        # already-corrupted checkpoint self-heal on resume instead of
        # immediately re-failing.
        self._configs = []
        self._objectives = []
        self._seen_config_hashes = set()
        self._archive = ParetoArchive()
        raw_configs = [encode(r, self.search_space) for r in state["raws"]]
        raw_objectives = [np.array(o) for o in state["objectives"]]
        for cfg, obj in zip(raw_configs, raw_objectives, strict=True):
            self.update(cfg, obj)


def _max_psd_clipped_mass(model) -> float:
    """0.0 for classical kernels (they're already exactly PSD, no
    projection needed); the max across objectives/groups for quantum
    kernels with finite shots (Brief §8's ``kernel_log.psd_clipped_mass``).
    """
    masses = [
        getattr(group_kernel, "last_psd_clipped_mass", 0.0)
        for sub_model in model.models
        for group_kernel in sub_model.covar_module.base_kernel.group_kernels.values()
    ]
    return max(masses) if masses else 0.0


@OPTIMIZERS.register("structured_bo")
class StructuredBO(KernelBO):
    group_kernel_factory = staticmethod(make_classical_group_kernel)

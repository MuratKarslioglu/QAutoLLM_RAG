"""Brief P4 gate: GP fit time < 2s at n=100."""

import time
from pathlib import Path

import numpy as np
import torch

from qautollm.kernels.classical import make_classical_group_kernel
from qautollm.kernels.structural import StructuralKernel
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace
from qautollm.surrogate.gp import augmented_features, fit_independent_gps
from qautollm.utils.seeds import make_rng

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"


def test_gp_fit_time_under_2s_at_n_100():
    space = SearchSpace.from_yaml(RAG_V2_PATH)
    rng = make_rng(0)
    configs = [encode(space.sample(rng), space) for _ in range(100)]
    x = augmented_features(configs)

    target_rng = np.random.default_rng(1)
    targets = target_rng.uniform(0, 1, size=(3, x.shape[1]))
    y = torch.tensor(
        np.stack([np.linalg.norm(x.numpy() - t, axis=1) for t in targets], axis=1),
        dtype=torch.float64,
    )

    def make_kernel() -> StructuralKernel:
        return StructuralKernel.from_search_space(space, make_classical_group_kernel)

    # Warm up once: import/JIT overhead on the very first fit in a process
    # shouldn't count against the budget — an orchestrator fits GPs
    # repeatedly within one long-running process, so steady-state is the
    # realistic measurement.
    fit_independent_gps(x[:20], y[:20], make_kernel)

    start = time.perf_counter()
    fit_independent_gps(x, y, make_kernel)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"GP fit took {elapsed:.2f}s, expected < 2s (Brief P4 gate)"

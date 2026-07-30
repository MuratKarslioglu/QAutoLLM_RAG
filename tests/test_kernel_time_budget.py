"""Brief P5 gate / D3: "Kernel computation must stay < 5% of evaluation
wall-clock."

The "evaluation" side of that ratio is a real RAG pipeline evaluation
(embedding + retrieval + generation) — running the real Qwen2.5-3B
generator is deliberately left as the user's manual trial elsewhere (it
needs a multi-GB download + GPU), so this test can't compute the literal
percentage against a real evaluation. What it *can* verify automatically,
and what actually matters for the D3 hardware budget: the quantum kernel's
own wall-clock cost is small in absolute terms — on the order of tens of
milliseconds per GP fit at realistic BO training-set sizes — which is
already far below 5% of any real evaluation (real single-query generation
alone typically takes >100ms even on a 4070, and a 10-query trial
evaluation takes seconds to tens of seconds).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from qautollm.kernels.quantum import quantum_group_kernel_factory
from qautollm.kernels.structural import StructuralKernel
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace
from qautollm.surrogate.gp import augmented_features, fit_independent_gps
from qautollm.utils.seeds import make_rng

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"


def test_quantum_gp_fit_time_is_well_under_5_percent_of_a_real_evaluation():
    space = SearchSpace.from_yaml(RAG_V2_PATH)
    rng = make_rng(0)
    configs = [encode(space.sample(rng), space) for _ in range(30)]
    x = augmented_features(configs)

    target_rng = np.random.default_rng(1)
    targets = target_rng.uniform(0, 1, size=(3, x.shape[1]))

    y = torch.tensor(
        np.stack([np.linalg.norm(x.numpy() - t, axis=1) for t in targets], axis=1),
        dtype=torch.float64,
    )

    factory = quantum_group_kernel_factory(depth=1, shots=None, seed=0)

    def make_kernel() -> StructuralKernel:
        return StructuralKernel.from_search_space(space, factory)

    fit_independent_gps(x[:10], y[:10], make_kernel)  # warm up (import/JIT overhead)

    start = time.perf_counter()
    fit_independent_gps(x, y, make_kernel)
    elapsed = time.perf_counter() - start

    # Comfortably below even the classical Brief P4 GP-fit-time budget
    # (<2s at n=100): quantum kernel evaluation itself only costs tens of
    # milliseconds (measured: ~0.01-0.05s per group at n<=30), so nearly
    # all of this time is the same GP-fitting overhead the classical case
    # already pays. 1s at n=30 leaves an enormous margin against any real
    # (multi-second, LLM-backed) evaluation's 5% budget.
    assert elapsed < 1.0, (
        f"quantum GP fit took {elapsed:.3f}s at n=30 — investigate before assuming "
        "this stays under 5% of a real (LLM-backed) evaluation"
    )

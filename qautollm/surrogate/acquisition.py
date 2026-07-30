"""EHVI acquisition (q=1), shared by every kernel-BO variant (Brief §5.4).

Uses BoTorch's ``qLogExpectedHypervolumeImprovement`` rather than the plain
``qExpectedHypervolumeImprovement``: identical acquisition concept and API,
but BoTorch's own numerics warning on the plain version recommends the log
variant for optimization stability — a strict improvement, not a feature
addition, so it doesn't create a classical/quantum asymmetry.

Our search space is discrete/mixed/conditional, not a continuous box, so
candidates are scored over a finite batch of validly-sampled configs and the
argmax is taken, rather than BoTorch's usual continuous ``optimize_acqf``.
"""

from __future__ import annotations

import torch
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.models import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)


def build_ehvi(
    model: ModelListGP,
    train_y_max: torch.Tensor,
    ref_point: torch.Tensor,
    mc_samples: int = 128,
) -> qLogExpectedHypervolumeImprovement:
    """``train_y_max``/``ref_point`` must be in *maximization* convention —
    negate minimize-convention objectives (Brief §5.5) before calling.
    """
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y_max)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    return qLogExpectedHypervolumeImprovement(
        model=model, ref_point=ref_point, partitioning=partitioning, sampler=sampler
    )


def select_best_candidate(
    acqf: qLogExpectedHypervolumeImprovement, candidate_features: torch.Tensor
) -> int:
    """``candidate_features`` is ``(n, 1, d)`` (q=1 batches); returns the
    index of the highest-scoring candidate.
    """
    with torch.no_grad():
        values = acqf(candidate_features)
    return int(torch.argmax(values))

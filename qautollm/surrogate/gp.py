"""Multi-output surrogate = independent GPs per objective (Brief §5.4).

Each objective gets its own ``SingleTaskGP`` with its own structural-kernel
instance (its own learned ``alpha_g``/lengthscales) — no cross-objective
parameter sharing. Objectives only interact later, through EHVI's joint
acquisition over the resulting ``ModelListGP``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from qautollm.space.encoding import CanonicalConfig


def augmented_features(configs: list[CanonicalConfig]) -> torch.Tensor:
    """``[z || mask]`` concatenated per config — the GP's raw input tensor.

    The structural kernel needs both the z-vector and the group activity
    mask, but a GPyTorch kernel only ever sees one feature tensor per point,
    so the mask rides along concatenated onto the end of z.
    """
    rows = [np.concatenate([c.z, c.mask]) for c in configs]
    return torch.tensor(np.stack(rows), dtype=torch.float64)


def fit_independent_gps(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    structural_kernel_factory: Callable[[], Kernel],
    max_iter: int = 50,
) -> ModelListGP:
    """Fit one ``SingleTaskGP`` per column of ``train_y`` (shape ``(n, n_objectives)``).

    ``structural_kernel_factory`` is called once per objective so each GP
    gets its own fresh (unfitted) kernel instance with independent
    hyperparameters. ``max_iter`` caps L-BFGS-B iterations — measured
    empirically to keep 3-objective fits at n=100 comfortably under the
    Brief §9 P4 gate (<2s): uncapped fits ran 1.4-3.3s, ``max_iter=50``
    brings steady-state fits to ~1.4s without sacrificing convergence
    quality on this low-dimensional a hyperparameter space (~10-15 params).
    """
    models = []
    for i in range(train_y.shape[-1]):
        kernel = ScaleKernel(structural_kernel_factory())
        model = SingleTaskGP(
            train_x,
            train_y[:, i : i + 1],
            covar_module=kernel,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": max_iter}})
        models.append(model)
    return ModelListGP(*models)

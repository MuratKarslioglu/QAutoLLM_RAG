"""Group-decomposed kernel skeleton, shared by classical and quantum BO
(Brief §5.4): ``K(x,x') = Σ_g α_g · δ_g(mask,mask') · K_g(z_g, z'_g)``.

``classical.py`` and (P5) ``quantum.py`` only provide the per-group
``GroupKernel`` factory passed in here — this file must never special-case
either family, or the classical/quantum comparison stops being fair
(Brief §1: "the classical BO and quantum BO must share every line of code
except the inner kernel function").

Input convention: GPyTorch kernels see one feature tensor per point, but we
need both the z-vector and the group activity mask. So every tensor this
kernel receives is ``[z || mask]`` concatenated (see
:func:`qautollm.surrogate.gp.augmented_features`) — ``forward`` splits it
back apart using ``group_slices``/``group_order``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel

from qautollm.space.encoding import encode
from qautollm.utils.seeds import make_rng

GroupKernelFactory = Callable[[str, list], Kernel]


class StructuralKernel(Kernel):
    has_lengthscale = False

    def __init__(
        self,
        group_slices: dict[str, slice],
        group_order: tuple[str, ...],
        group_kernels: dict[str, Kernel],
        kappa: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.group_order = group_order
        self.group_slices = group_slices
        self.z_dim = sum(s.stop - s.start for s in group_slices.values())
        self.n_groups = len(group_order)
        self.kappa = kappa
        self.group_kernels = torch.nn.ModuleDict(group_kernels)

        self.register_parameter("raw_alpha", torch.nn.Parameter(torch.zeros(self.n_groups)))
        self.register_constraint("raw_alpha", Positive())

    @classmethod
    def from_search_space(
        cls, search_space, group_kernel_factory: GroupKernelFactory, kappa: float = 0.1
    ) -> StructuralKernel:
        """Build the kernel's group_slices from the search space itself
        (via one throwaway encoded config) rather than duplicating
        encoding.py's width-computation logic here.
        """
        sample_raw = search_space.sample(make_rng(0))
        sample_cfg = encode(sample_raw, search_space)
        group_kernels = {
            g: group_kernel_factory(g, search_space.variables_in_group(g))
            for g in search_space.group_order
        }
        return cls(
            group_slices=sample_cfg.group_slices,
            group_order=search_space.group_order,
            group_kernels=group_kernels,
            kappa=kappa,
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        z1, mask1 = x1[..., : self.z_dim], x1[..., self.z_dim :]
        z2, mask2 = x2[..., : self.z_dim], x2[..., self.z_dim :]

        total = None
        for gi, g in enumerate(self.group_order):
            sl = self.group_slices[g]
            k_g = self.group_kernels[g].forward(z1[..., sl], z2[..., sl], diag=diag, **params)
            # Composite (e.g. product) GroupKernels may call their own
            # sub-kernels through __call__ internally, which always wraps
            # results as a LinearOperator — densify so the where/mul below
            # can freely combine this with plain tensors.
            if hasattr(k_g, "to_dense"):
                k_g = k_g.to_dense()

            m1, m2 = mask1[..., gi], mask2[..., gi]
            if diag:
                delta = torch.where(m1 == m2, torch.ones_like(m1), torch.full_like(m1, self.kappa))
            else:
                match = m1.unsqueeze(-1) == m2.unsqueeze(-2)
                delta = torch.where(
                    match, torch.ones_like(k_g), torch.full_like(k_g, self.kappa)
                )

            term = self.alpha[gi] * delta * k_g
            total = term if total is None else total + term
        return total

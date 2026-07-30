"""Classical GroupKernel: Matérn-5/2 (numeric dims) × categorical-overlap
kernel (one-hot dims), per group (Brief §5.4).

Provides only the ``GroupKernel`` factory consumed by
:meth:`qautollm.kernels.structural.StructuralKernel.from_search_space` —
``quantum.py`` (P5) provides the same-shaped factory so
``structured_bo``/``qkernel_bo`` differ only in which one they pass.
"""

from __future__ import annotations

import torch
from gpytorch.kernels import Kernel, MaternKernel


class _SlicedKernel(Kernel):
    """Applies ``base`` to a fixed subset of the last dimension only."""

    has_lengthscale = False

    def __init__(self, base: Kernel, indices: list[int]) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("_indices", torch.tensor(indices, dtype=torch.long))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        x1s = x1.index_select(-1, self._indices)
        x2s = x2.index_select(-1, self._indices)
        out = self.base.forward(x1s, x2s, diag=diag, **params)
        # MaternKernel.forward can return a LinearOperator (not a plain
        # Tensor) for larger/batched inputs — densify so StructuralKernel
        # can freely combine this with plain-tensor arithmetic (where/mul).
        return out.to_dense() if hasattr(out, "to_dense") else out


class CategoricalOverlapKernel(Kernel):
    """``k(x1,x2)`` = fraction of matching one-hot columns (1.0 if identical
    category, 0.0 if completely disjoint one-hot blocks concatenated).
    """

    has_lengthscale = False

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        if diag:
            return (x1 == x2).to(x1.dtype).mean(dim=-1)
        eq = (x1.unsqueeze(-2) == x2.unsqueeze(-3)).to(x1.dtype)
        return eq.mean(dim=-1)


def make_classical_group_kernel(group_name: str, variables: list) -> Kernel:
    """Composite kernel for one group's z-slice: Matérn-5/2 over its
    numeric (binary/discrete_numeric) columns times a categorical-overlap
    kernel over its one-hot categorical columns. Groups with only one kind
    of column skip the product and return that sub-kernel directly.
    """
    numeric_idx: list[int] = []
    categorical_idx: list[int] = []
    cursor = 0
    for v in variables:
        width = len(v.values) if v.kind == "categorical" else 1
        if v.kind == "categorical":
            categorical_idx.extend(range(cursor, cursor + width))
        else:
            numeric_idx.append(cursor)
        cursor += width

    sub_kernels: list[Kernel] = []
    if numeric_idx:
        sub_kernels.append(_SlicedKernel(MaternKernel(nu=2.5), numeric_idx))
    if categorical_idx:
        sub_kernels.append(_SlicedKernel(CategoricalOverlapKernel(), categorical_idx))

    if not sub_kernels:
        raise ValueError(f"group {group_name!r} has no variables to build a kernel from")
    if len(sub_kernels) == 1:
        return sub_kernels[0]
    return sub_kernels[0] * sub_kernels[1]

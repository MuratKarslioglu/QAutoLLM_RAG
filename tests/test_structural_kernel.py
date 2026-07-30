from pathlib import Path

import pytest
import torch

from qautollm.kernels.classical import make_classical_group_kernel
from qautollm.kernels.structural import StructuralKernel
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace
from qautollm.surrogate.gp import augmented_features
from qautollm.utils.seeds import make_rng

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"
_SPACE = SearchSpace.from_yaml(RAG_V2_PATH)


def _features(n: int, seed: int) -> torch.Tensor:
    rng = make_rng(seed)
    configs = [encode(_SPACE.sample(rng), _SPACE) for _ in range(n)]
    return augmented_features(configs)


def _kernel() -> StructuralKernel:
    return StructuralKernel.from_search_space(_SPACE, make_classical_group_kernel)


def test_kernel_matrix_shape():
    x = _features(6, seed=0)
    k = _kernel()
    with torch.no_grad():
        mat = k.forward(x, x).to_dense()
    assert mat.shape == (6, 6)


def test_kernel_is_symmetric():
    x = _features(6, seed=1)
    k = _kernel()
    with torch.no_grad():
        mat = k.forward(x, x).to_dense()
    assert torch.allclose(mat, mat.T, atol=1e-8)


def test_kernel_diagonal_is_self_similarity_maximum():
    x = _features(10, seed=2)
    k = _kernel()
    with torch.no_grad():
        mat = k.forward(x, x).to_dense()
    diag = torch.diag(mat)
    # every off-diagonal entry in a row must not exceed that row's self-similarity
    for i in range(mat.shape[0]):
        assert torch.all(mat[i] <= diag[i] + 1e-6)


def test_asymmetric_shapes_supported():
    x1 = _features(4, seed=3)
    x2 = _features(7, seed=4)
    k = _kernel()
    with torch.no_grad():
        mat = k.forward(x1, x2).to_dense()
    assert mat.shape == (4, 7)


def test_mask_mismatch_uses_kappa_not_full_similarity():
    """Two configs identical except one has hybrid_on active (dense_weight
    present) and the other doesn't: their G3_retrieval group masks differ,
    so that group's contribution must be scaled by kappa, not 1.0.
    """
    raw_a = {
        "chunking_strategy": "fixed",
        "chunk_size": 256,
        "chunk_overlap": 0,
        "embedding_model": "bge-small",
        "top_k": 5,
        "hybrid_on": 0,
        "reranker_on": 0,
        "context_budget": 1024,
    }
    raw_b = {**raw_a, "hybrid_on": 1, "dense_weight": 0.5}

    cfg_a = encode(raw_a, _SPACE)
    cfg_b = encode(raw_b, _SPACE)
    x = augmented_features([cfg_a, cfg_b])

    k = StructuralKernel.from_search_space(_SPACE, make_classical_group_kernel, kappa=0.1)
    with torch.no_grad():
        mat = k.forward(x, x).to_dense()

    g3_idx = _SPACE.group_order.index("G3_retrieval")
    assert cfg_a.mask[g3_idx] != cfg_b.mask[g3_idx]
    # cross term must be strictly less than either self-similarity, since at
    # least the G3_retrieval group is downweighted by kappa < 1
    assert mat[0, 1] < mat[0, 0]
    assert mat[0, 1] < mat[1, 1]


class _OnesKernel(torch.nn.Module):
    """A trivial GroupKernel stand-in: constant similarity 1.0 everywhere,
    so the mask's kappa discount can be isolated from any real z-distance.
    """

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        if diag:
            return torch.ones(x1.shape[:-1], dtype=x1.dtype)
        return torch.ones(x1.shape[-2], x2.shape[-2], dtype=x1.dtype)


def test_delta_uses_kappa_exactly_when_masks_disagree():
    kernel = StructuralKernel(
        group_slices={"g0": slice(0, 1)},
        group_order=("g0",),
        group_kernels={"g0": _OnesKernel()},
        kappa=0.25,
    )
    alpha = float(kernel.alpha[0].detach())

    z = torch.zeros(2, 1, dtype=torch.float64)
    mask_same = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    mask_diff = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    x_same = torch.cat([z, mask_same], dim=-1)
    x_diff = torch.cat([z, mask_diff], dim=-1)

    with torch.no_grad():
        k_same = kernel.forward(x_same, x_same).to_dense()
        k_diff = kernel.forward(x_diff, x_diff).to_dense()

    assert k_same[0, 1] == pytest.approx(alpha * 1.0)
    assert k_diff[0, 1] == pytest.approx(alpha * 0.25)

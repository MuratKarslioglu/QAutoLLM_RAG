"""Group definitions G1-G5 for the group-decomposed kernel (Brief §5.4, §7).

The group order defined here is canonical: it fixes the order of the
``mask`` array in :class:`~qautollm.space.encoding.CanonicalConfig` and the
iteration order of ``group_slices``. Every module that touches groups must
import ``GROUP_ORDER`` from here rather than re-deriving it.
"""

from __future__ import annotations

GROUP_ORDER: tuple[str, ...] = (
    "G1_chunking",
    "G2_embedding",
    "G3_retrieval",
    "G4_reranking",
    "G5_context",
)

GROUP_VARIABLES: dict[str, tuple[str, ...]] = {
    "G1_chunking": ("chunking_strategy", "chunk_size", "chunk_overlap"),
    "G2_embedding": ("embedding_model",),
    "G3_retrieval": ("top_k", "hybrid_on", "dense_weight"),
    "G4_reranking": ("reranker_on", "reranker_top_n"),
    "G5_context": ("context_budget",),
}


def group_of(variable_name: str) -> str:
    """Return the group name that owns ``variable_name``.

    Raises:
        KeyError: if the variable is not assigned to any group.
    """
    for group_name, variables in GROUP_VARIABLES.items():
        if variable_name in variables:
            return group_name
    raise KeyError(f"variable {variable_name!r} is not assigned to any group")


def group_index(group_name: str) -> int:
    """Return the fixed position of ``group_name`` in ``GROUP_ORDER``."""
    return GROUP_ORDER.index(group_name)

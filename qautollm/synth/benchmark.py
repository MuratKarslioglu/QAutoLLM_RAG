"""Synthetic mixed-space test functions ("Layer A", Brief §4/§9 P7).

Builds parametrized synthetic ``(SearchSpace, objective_fn)`` pairs that
mirror the real ``rag_v2`` space's *structure* (groups, categorical +
binary + conditional variables, ≤5 qubits/group) without needing any
RAG/LLM evaluation — cheap enough to sweep across many
``(dim, conditionality, interaction)`` combinations and many seeds.

Sweep axes, matching Brief §1's core question (does a group-decomposed
quantum fidelity kernel outperform a classical one, and under what
conditions):
  - **dim**: number of groups (``n_groups``).
  - **conditionality**: fraction of groups that get an extra conditional
    variable (branching structure) vs. a fixed combinatorial-only group.
  - **interaction**: ``0.0`` = the true objective is exactly additive
    across groups (matches ``StructuralKernel``'s own ``K=Σ_g α_g K_g``
    assumption for *both* classical and quantum, Brief §5.4 — the "easy",
    well-specified case); ``1.0`` = strong pairwise cross-group coupling
    the additive kernel skeleton cannot represent at all (the "hard",
    misspecified case). This misspecification is shared by classical
    *and* quantum kernels (both sit inside the same additive
    ``StructuralKernel``) — this axis measures BO robustness to that
    shared limitation, not a classical/quantum architectural difference.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np

from qautollm.space.encoding import CanonicalConfig, encode
from qautollm.space.search_space import SearchSpace
from qautollm.utils.seeds import make_rng

_CATEGORICAL_WIDTH = 3  # + 1 binary gate (+1 conditional var if present) stays <=5 qubits/group


def make_synthetic_space(n_groups: int, conditional_fraction: float = 0.0) -> SearchSpace:
    """Build an ``n_groups``-group synthetic search space.

    Every group has one categorical variable (width ``_CATEGORICAL_WIDTH``)
    and one binary gate; a ``conditional_fraction`` share of groups (the
    first ``round(conditional_fraction * n_groups)`` of them) also get a
    conditional numeric variable gated on their own binary — group width
    stays <=5 either way (Brief D3's qubit cap).
    """
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")
    if not 0.0 <= conditional_fraction <= 1.0:
        raise ValueError("conditional_fraction must be in [0, 1]")
    n_conditional_groups = round(conditional_fraction * n_groups)

    variables = []
    groups = []
    for i in range(n_groups):
        group_name = f"S{i}"
        groups.append(group_name)
        variables.append(
            {
                "name": f"cat{i}",
                "kind": "categorical",
                "group": group_name,
                "values": [f"c{i}_{j}" for j in range(_CATEGORICAL_WIDTH)],
            }
        )
        variables.append(
            {"name": f"gate{i}", "kind": "binary", "group": group_name, "values": [0, 1]}
        )
        if i < n_conditional_groups:
            variables.append(
                {
                    "name": f"cond{i}",
                    "kind": "discrete_numeric",
                    "group": group_name,
                    "values": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "condition": {"depends_on": f"gate{i}", "active_when": 1},
                }
            )

    return SearchSpace.from_dict({"variables": variables, "groups": groups})


def make_qubit_sweep_space(
    n_groups: int, qubits_per_group: int, max_qubits: int = 5
) -> SearchSpace:
    """Every group is exactly one variable whose one-hot/scalar width equals
    ``qubits_per_group``: a binary 2-way choice at 1 qubit, an N-category
    choice at N>=2 qubits. No gates/conditionals — deliberately simpler
    than :func:`make_synthetic_space` so the qubit-count axis isn't
    confounded with the conditionality axis (a qubit-count sweep
    experiment, not the Layer-A dim/conditionality/interaction sweep).

    ``max_qubits`` defaults to Brief D3's 5-qubit-per-group cap. Pass a
    higher value only for the exploratory, beyond-Brief-D3 sweep in
    ``scripts/run_qubit_sweep_extended.py`` (2026-07-18) — the real
    pipeline (``configs/search_spaces/rag_v2.yaml``'s groups, and every
    other caller of this function) always uses the default.
    """
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")
    if not 1 <= qubits_per_group <= max_qubits:
        raise ValueError(f"qubits_per_group must be in [1, {max_qubits}]")

    if qubits_per_group == 1:
        variables = [
            {"name": f"cat{i}", "kind": "binary", "group": f"Q{i}", "values": [0, 1]}
            for i in range(n_groups)
        ]
    else:
        variables = [
            {
                "name": f"cat{i}",
                "kind": "categorical",
                "group": f"Q{i}",
                "values": [f"c{i}_{j}" for j in range(qubits_per_group)],
            }
            for i in range(n_groups)
        ]
    groups = [f"Q{i}" for i in range(n_groups)]
    return SearchSpace.from_dict({"variables": variables, "groups": groups})


def build_synthetic_objectives(
    space: SearchSpace, interaction: float, seed: int, n_objectives: int = 3
) -> Callable[[CanonicalConfig], np.ndarray]:
    """Return a deterministic ``config -> objectives`` function (minimize
    convention, Brief §5.5) for this benchmark instance.

    Per objective: an *additive* term (mean squared distance, per group, to
    a fixed random target) blended with a *coupling* term (a fixed random
    scalar weight per group-pair times the product of the two groups' mean
    z-values) via ``interaction``.
    """
    if not 0.0 <= interaction <= 1.0:
        raise ValueError("interaction must be in [0, 1]")

    rng = np.random.default_rng(seed)
    sample_cfg = encode(space.sample(make_rng(seed)), space)
    group_slices = sample_cfg.group_slices
    group_names = space.group_order
    group_pairs = list(combinations(group_names, 2))

    targets = {
        obj_idx: {
            g: rng.uniform(0.0, 1.0, size=group_slices[g].stop - group_slices[g].start)
            for g in group_names
        }
        for obj_idx in range(n_objectives)
    }
    coupling_weights = {
        obj_idx: dict(zip(group_pairs, rng.uniform(-1.0, 1.0, size=len(group_pairs)), strict=True))
        for obj_idx in range(n_objectives)
    }

    def objective_fn(config: CanonicalConfig) -> np.ndarray:
        z = config.z
        group_means = {g: float(np.mean(z[group_slices[g]])) for g in group_names}

        values = np.empty(n_objectives, dtype=np.float64)
        for obj_idx in range(n_objectives):
            additive = np.mean(
                [
                    np.mean((z[group_slices[g]] - targets[obj_idx][g]) ** 2)
                    for g in group_names
                ]
            )
            if group_pairs:
                coupled = np.mean(
                    [
                        coupling_weights[obj_idx][pair]
                        * group_means[pair[0]]
                        * group_means[pair[1]]
                        for pair in group_pairs
                    ]
                )
            else:
                coupled = 0.0
            values[obj_idx] = (1.0 - interaction) * additive + interaction * abs(coupled)
        return values

    return objective_fn

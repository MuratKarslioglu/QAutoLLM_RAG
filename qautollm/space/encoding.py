"""Canonical encoding between raw config dicts and fixed-length z-vectors
(Brief §5.1). This is the one place where the RAG parameter space is turned
into the numeric representation every optimizer, GP, and kernel consumes.

Assumption relied on by ``encode``/``decode``: at most one conditional
variable's gate needs to be tracked per group (true for ``rag_v2.yaml``: G3's
gate is ``hybrid_on``, G4's gate is ``reranker_on``). If a future search space
needs multiple independently-gated conditional variables in one group, the
group-level ``mask`` scalar must become a per-variable mask and every kernel
consumer of ``mask`` must be updated in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qautollm.space.search_space import SearchSpace, Variable
from qautollm.store.hashing import canonical_hash

_SENTINEL = 0.5


@dataclass(frozen=True)
class CanonicalConfig:
    """The canonical representation of one RAG pipeline configuration."""

    raw: dict[str, Any]
    z: np.ndarray
    mask: np.ndarray
    group_slices: dict[str, slice]
    config_hash: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CanonicalConfig):
            return NotImplemented
        return self.config_hash == other.config_hash

    def __hash__(self) -> int:
        # raw/z/mask are unhashable (dict, ndarray); identity is config_hash.
        return hash(self.config_hash)


def _variable_width(v: Variable) -> int:
    return len(v.values) if v.kind == "categorical" else 1


def _normalize(v: Variable, value: Any) -> np.ndarray:
    if v.kind == "categorical":
        onehot = np.zeros(len(v.values), dtype=np.float64)
        onehot[v.values.index(value)] = 1.0
        return onehot
    if v.kind == "binary":
        return np.array([float(value)], dtype=np.float64)
    lo, hi = min(v.values), max(v.values)
    norm = 0.0 if hi == lo else (value - lo) / (hi - lo)
    return np.array([float(norm)], dtype=np.float64)


def _denormalize(v: Variable, segment: np.ndarray) -> Any:
    if v.kind == "categorical":
        return v.values[int(np.argmax(segment))]
    if v.kind == "binary":
        return v.values[int(round(float(segment[0])))]
    lo, hi = min(v.values), max(v.values)
    approx = segment[0] * (hi - lo) + lo
    return min(v.values, key=lambda candidate: abs(candidate - approx))


def encode(raw: dict[str, Any], space: SearchSpace) -> CanonicalConfig:
    """Encode a raw config dict into its :class:`CanonicalConfig`.

    Raises:
        SearchSpaceError: if ``raw`` is not a valid, complete config for
            ``space`` (unknown keys, out-of-domain values, or a violated
            conditional rule) — validation happens at construction time.
    """
    space.validate(raw)

    z_parts: list[np.ndarray] = []
    mask = np.ones(len(space.group_order), dtype=np.float64)
    group_slices: dict[str, slice] = {}
    cursor = 0

    for gi, group_name in enumerate(space.group_order):
        group_vars = space.variables_in_group(group_name)
        group_start = cursor
        conditional_gate_active: bool | None = None

        for v in group_vars:
            width = _variable_width(v)
            if v.is_conditional:
                gate_active = v.condition.is_active(raw)
                conditional_gate_active = gate_active
                if gate_active:
                    z_parts.append(_normalize(v, raw[v.name]))
                else:
                    z_parts.append(np.full(width, _SENTINEL, dtype=np.float64))
            else:
                z_parts.append(_normalize(v, raw[v.name]))
            cursor += width

        group_slices[group_name] = slice(group_start, cursor)
        if conditional_gate_active is not None:
            mask[gi] = 1.0 if conditional_gate_active else 0.0

    z = np.concatenate(z_parts) if z_parts else np.zeros(0, dtype=np.float64)
    return CanonicalConfig(
        raw=dict(raw),
        z=z,
        mask=mask,
        group_slices=group_slices,
        config_hash=canonical_hash(raw),
    )


def decode(config: CanonicalConfig, space: SearchSpace) -> dict[str, Any]:
    """Reconstruct the raw config dict purely from ``config.z``/``mask``.

    Deliberately does not read ``config.raw`` — this is what makes
    ``decode(encode(x)) == x`` a real round-trip test of the numeric
    encoding, not a pass-through of the stored dict.
    """
    raw: dict[str, Any] = {}
    for gi, group_name in enumerate(space.group_order):
        group_vars = space.variables_in_group(group_name)
        cursor = config.group_slices[group_name].start
        for v in group_vars:
            width = _variable_width(v)
            segment = config.z[cursor : cursor + width]
            cursor += width
            if v.is_conditional:
                if config.mask[gi] >= 0.5:
                    raw[v.name] = _denormalize(v, segment)
            else:
                raw[v.name] = _denormalize(v, segment)
    return raw

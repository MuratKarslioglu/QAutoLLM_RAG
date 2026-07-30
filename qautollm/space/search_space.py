"""Variable definitions, sampling, and validity checking for the RAG search
space (Brief §5.1, §7). Loads from YAML files such as
``configs/search_spaces/rag_v2.yaml``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError
from scipy.stats import qmc

from qautollm.core.config import SearchSpaceSpec
from qautollm.space.constraints import (
    ConditionalRule,
    ConstraintViolation,
)
from qautollm.space.constraints import (
    validate as _validate_rules,
)

VariableKind = Literal["categorical", "discrete_numeric", "binary"]


@dataclass(frozen=True)
class Variable:
    """One decision variable of the search space."""

    name: str
    kind: VariableKind
    group: str
    values: tuple[Any, ...]
    condition: ConditionalRule | None = None

    @property
    def is_conditional(self) -> bool:
        return self.condition is not None

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw one value uniformly from ``self.values`` using ``rng``."""
        idx = int(rng.integers(0, len(self.values)))
        return self.values[idx]


class SearchSpaceError(ValueError):
    """Raised for malformed search-space YAML or domain violations."""


@dataclass(frozen=True)
class SearchSpace:
    """A complete, validated search space: variables + groups + conditional rules."""

    variables: tuple[Variable, ...]
    group_order: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> SearchSpace:
        try:
            spec = SearchSpaceSpec.from_yaml(path)
        except ValidationError as e:
            raise SearchSpaceError(f"invalid search space schema in {path}: {e}") from e
        return cls.from_spec(spec)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SearchSpace:
        try:
            spec = SearchSpaceSpec.model_validate(raw)
        except ValidationError as e:
            raise SearchSpaceError(f"invalid search space schema: {e}") from e
        return cls.from_spec(spec)

    @classmethod
    def from_spec(cls, spec: SearchSpaceSpec) -> SearchSpace:
        # Group names are whatever this spec declares — not restricted to
        # the real rag_v2 space's G1-G5 (Brief §7); Brief §9 P7's synthetic
        # benchmark spaces need arbitrary group counts/names.
        # _check_internal_consistency below still catches a variable
        # referencing a group missing from `groups:`.
        group_order = tuple(spec.groups)

        variables: list[Variable] = []
        for entry in spec.variables:
            condition = None
            if entry.condition is not None:
                condition = ConditionalRule(
                    variable=entry.name,
                    depends_on=entry.condition.depends_on,
                    active_when=entry.condition.active_when,
                )
            variables.append(
                Variable(
                    name=entry.name,
                    kind=entry.kind,
                    group=entry.group,
                    values=tuple(entry.values),
                    condition=condition,
                )
            )

        space = cls(variables=tuple(variables), group_order=group_order)
        space._check_internal_consistency()
        return space

    def _check_internal_consistency(self) -> None:
        names = [v.name for v in self.variables]
        if len(names) != len(set(names)):
            raise SearchSpaceError(f"duplicate variable names in search space: {names}")
        for v in self.variables:
            if v.group not in self.group_order:
                raise SearchSpaceError(f"variable {v.name!r} references unknown group {v.group!r}")
            if v.condition is not None and v.condition.depends_on not in names:
                raise SearchSpaceError(
                    f"variable {v.name!r} depends on undefined variable {v.condition.depends_on!r}"
                )

    @property
    def rules(self) -> list[ConditionalRule]:
        return [v.condition for v in self.variables if v.condition is not None]

    def by_name(self, name: str) -> Variable:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(f"unknown variable {name!r}")

    def variables_in_group(self, group_name: str) -> list[Variable]:
        return [v for v in self.variables if v.group == group_name]

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """Draw one uniformly-random *valid* raw config.

        Non-conditional variables are sampled first, then conditional
        variables are sampled only if their gate is active in the resulting
        dict — otherwise they are omitted entirely (never merely masked).
        """
        raw: dict[str, Any] = {}
        for v in self.variables:
            if not v.is_conditional:
                raw[v.name] = v.sample(rng)
        for v in self.variables:
            if v.is_conditional and v.condition.is_active(raw):
                raw[v.name] = v.sample(rng)
        return raw

    def sobol_design(self, n_points: int, seed: int) -> list[dict[str, Any]]:
        """Quasi-random Sobol init design (Brief §5.2): "one shared Sobol
        init design of 10 points per seed", passed identically to every
        optimizer for a paired comparison — this is the one place that
        design gets generated.

        Draws one Sobol coordinate per non-conditional variable, plus one
        per conditional variable (consumed only if that variable turns out
        active), so the same underlying draw always maps to the same value
        deterministically for a given seed regardless of optimizer.
        """
        non_conditional = [v for v in self.variables if not v.is_conditional]
        conditional = [v for v in self.variables if v.is_conditional]
        dim = len(non_conditional) + len(conditional)

        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        with warnings.catch_warnings():
            # Brief §5.2 fixes the init design at 10 points, not a power of
            # 2 — scipy's balance-property warning doesn't apply to us here.
            warnings.simplefilter("ignore", UserWarning)
            draws = sampler.random(n_points)

        designs: list[dict[str, Any]] = []
        for row in draws:
            raw: dict[str, Any] = {}
            cursor = 0
            for v in non_conditional:
                raw[v.name] = self._pick_from_unit(v, row[cursor])
                cursor += 1
            for v in conditional:
                u = row[cursor]
                cursor += 1
                if v.condition.is_active(raw):
                    raw[v.name] = self._pick_from_unit(v, u)
            designs.append(raw)
        return designs

    @staticmethod
    def _pick_from_unit(v: Variable, u: float) -> Any:
        position = min(int(u * len(v.values)), len(v.values) - 1)
        return v.values[position]

    def validate(self, raw: dict[str, Any]) -> None:
        """Raise if ``raw`` is not a valid, complete config for this space.

        Checks: no unknown keys, all non-conditional keys present, all
        values within domain, and all conditional rules satisfied.
        """
        known_names = {v.name for v in self.variables}
        unknown = set(raw) - known_names
        if unknown:
            raise SearchSpaceError(f"unknown parameter(s) in config: {unknown}")

        for v in self.variables:
            if not v.is_conditional and v.name not in raw:
                raise SearchSpaceError(f"missing required parameter {v.name!r}")
            if v.name in raw and raw[v.name] not in v.values:
                raise SearchSpaceError(
                    f"value {raw[v.name]!r} for {v.name!r} not in domain {v.values}"
                )

        try:
            _validate_rules(raw, self.rules)
        except ConstraintViolation as e:
            raise SearchSpaceError(str(e)) from e

"""Conditional rule enforcement for the RAG search space (Brief §5.1, §7).

A conditional rule ties one variable's presence to another variable's value
(e.g. ``dense_weight`` is only meaningful when ``hybrid_on == 1``). Rules are
data, not code, so they round-trip through the same YAML that defines the
search space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ConstraintViolation(ValueError):
    """Raised when a raw config dict violates a conditional rule."""


@dataclass(frozen=True)
class ConditionalRule:
    """``variable`` may only appear in ``raw`` when ``depends_on == active_when``."""

    variable: str
    depends_on: str
    active_when: Any

    def is_active(self, raw: dict[str, Any]) -> bool:
        """Whether ``self.variable`` is allowed to be present, given ``raw``."""
        if self.depends_on not in raw:
            raise ConstraintViolation(
                f"cannot evaluate condition for {self.variable!r}: "
                f"gating variable {self.depends_on!r} is missing from raw config"
            )
        return raw[self.depends_on] == self.active_when


def validate(raw: dict[str, Any], rules: list[ConditionalRule]) -> None:
    """Raise :class:`ConstraintViolation` if ``raw`` violates any rule.

    Two failure modes are checked:
      - ``variable`` is present in ``raw`` while its condition is inactive.
      - ``variable`` is active-by-condition but missing from ``raw``.
    """
    for rule in rules:
        active = rule.is_active(raw)
        present = rule.variable in raw
        if present and not active:
            raise ConstraintViolation(
                f"{rule.variable!r} is present but {rule.depends_on!r} != "
                f"{rule.active_when!r} in raw config {raw!r}"
            )
        if active and not present:
            raise ConstraintViolation(
                f"{rule.variable!r} is required (since {rule.depends_on!r} == "
                f"{rule.active_when!r}) but missing from raw config {raw!r}"
            )


def strip_inactive(raw: dict[str, Any], rules: list[ConditionalRule]) -> dict[str, Any]:
    """Return a copy of ``raw`` with any variable dropped whose condition is inactive.

    Used before hashing/validation so that two raw dicts that differ only in
    a leftover inactive key still canonicalize to the same config.
    """
    result = dict(raw)
    for rule in rules:
        if rule.variable in result and not rule.is_active(result):
            del result[rule.variable]
    return result

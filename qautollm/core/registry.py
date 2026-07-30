"""String -> class registries for optimizers, kernels, and metrics (Brief §4).

Registries are the one place in the codebase allowed to hold global mutable
state (Brief §10). Every optimizer/kernel/metric implementation registers
itself with a decorator; config-driven code (e.g. the orchestrator reading
``optimizer.name`` from YAML) looks classes up by name here instead of
importing every implementation directly.
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> class map with a `@register("name")` decorator."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str):
        def _decorator(cls: type[T]) -> type[T]:
            if name in self._entries and self._entries[name] is not cls:
                raise ValueError(
                    f"{self._kind} registry already has an entry for {name!r} "
                    f"({self._entries[name]!r}); refusing to overwrite with {cls!r}"
                )
            self._entries[name] = cls
            return cls

        return _decorator

    def get(self, name: str) -> type[T]:
        try:
            return self._entries[name]
        except KeyError:
            available = sorted(self._entries)
            raise KeyError(
                f"unknown {self._kind} {name!r}; registered: {available}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries


OPTIMIZERS: Registry = Registry("optimizer")
KERNELS: Registry = Registry("kernel")
METRICS: Registry = Registry("metric")

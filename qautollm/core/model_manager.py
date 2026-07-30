"""Enforces D1 (Brief §2): at most one *large* model resident in VRAM.

Nothing outside this module should call a model's ``.to("cuda")`` /
``device_map=...`` construction path directly for the embedder or the
generator — those go through :meth:`ModelManager.load_large`, which owns
deciding whether the previous large model must be unloaded first. Small
models (reranker, NLI, ~0.5GB each) may co-reside with the resident large
model via :meth:`ModelManager.load_small`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qautollm.utils.logging import get_logger
from qautollm.utils.profiling import vram_probe

_logger = get_logger(__name__)


class ModelManager:
    """Owns load/unload of the single resident large model, plus any number
    of co-resident small models.
    """

    def __init__(self) -> None:
        self._large_name: str | None = None
        self._large_model: Any = None
        self._small_models: dict[str, Any] = {}

    def load_large(self, name: str, factory: Callable[[], Any]) -> Any:
        """Return the resident large model named ``name``, constructing it
        via ``factory()`` if needed. If a *different* large model is
        currently resident, it is unloaded first (D1: only one at a time).
        """
        if self._large_name == name:
            return self._large_model

        if self._large_name is not None:
            self.unload_large()

        _logger.info("loading large model %r", name)
        self._large_model = factory()
        self._large_name = name
        _logger.info("loaded large model %r; vram=%s", name, vram_probe())
        return self._large_model

    def unload_large(self) -> None:
        if self._large_name is None:
            return
        _logger.info("unloading large model %r", self._large_name)
        self._large_model = None
        self._large_name = None
        self._free_cuda_cache()
        _logger.info("unloaded large model; vram=%s", vram_probe())

    def load_small(self, name: str, factory: Callable[[], Any]) -> Any:
        """Return the resident small model named ``name``, constructing it
        via ``factory()`` on first use. Small models are never unloaded by
        loading a different large model — they're assumed to fit alongside
        it (Brief D1: ~0.5GB each).
        """
        if name not in self._small_models:
            _logger.info("loading small model %r", name)
            self._small_models[name] = factory()
            _logger.info("loaded small model %r; vram=%s", name, vram_probe())
        return self._small_models[name]

    @property
    def resident_large(self) -> str | None:
        return self._large_name

    @staticmethod
    def _free_cuda_cache() -> None:
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

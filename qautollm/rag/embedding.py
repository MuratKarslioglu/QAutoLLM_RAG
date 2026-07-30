"""Embedding model wrapper for offline batch indexing (Brief §4).

Deliberately does not call ``.to("cuda")`` on its own initiative — callers
construct it through :meth:`qautollm.core.model_manager.ModelManager.load_large`
so the manager controls *when* construction (and therefore VRAM allocation)
happens (Brief D1).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from qautollm.core.config import EmbeddingModelSpec, load_model_catalog
from qautollm.utils.logging import get_logger

_CATALOG_PATH = Path(__file__).resolve().parents[2] / "configs" / "models" / "embedding.yaml"
_logger = get_logger(__name__)


def _catalog() -> dict[str, EmbeddingModelSpec]:
    return load_model_catalog(_CATALOG_PATH, EmbeddingModelSpec)


class EmbeddingModel:
    """A single sentence-transformers embedding model, used for both corpus
    chunks and queries so retrieval scores are comparable (same space).
    """

    def __init__(self, name: str, device: str = "cpu") -> None:
        spec = _catalog()[name]
        self.name = name
        self.dim = spec.dim
        self._model = SentenceTransformer(spec.hf_id, device=device)

    def encode(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """L2-normalized embeddings (so downstream inner-product == cosine).

        Logs one start/done line per call instead of a per-batch progress
        bar (``show_progress_bar`` to sentence-transformers is always off in
        practice) — a single call can take a while on a large chunk set, and
        with no per-batch bar the terminal would otherwise go silent for
        that whole stretch with no sign the process is still alive.
        """
        _logger.info("embedding %d texts (model=%s)...", len(texts), self.name)
        start = time.perf_counter()
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        elapsed = time.perf_counter() - start
        rate = len(texts) / elapsed if elapsed > 0 else float("inf")
        _logger.info("embedded %d texts in %.1fs (%.0f texts/s)", len(texts), elapsed, rate)
        return np.asarray(vectors, dtype=np.float32)


class SyntheticEmbeddingModel:
    """Deterministic fake embedder for smoke tests / toy-dataset dry runs —
    each text hashes to a fixed seeded unit vector instead of running a real
    model. Never use for real experiments; only for exercising pipeline
    mechanics (orchestrator, resume, caching) without a GPU or network.
    """

    def __init__(self, name: str, dim: int = 16, device: str = "cpu") -> None:
        # `device` is accepted (and ignored) only so this class is
        # interchangeable with the real `EmbeddingModel` behind
        # `build_index.build_all`'s single `embedder_cls(name, device=...)`
        # call site -- this fake never runs a real model, so there's
        # nothing to place on a device.
        self.name = name
        self.dim = dim

    def encode(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        return np.stack([self._vector_for(t) for t in texts])

    def _vector_for(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=self.dim).astype(np.float32)
        return v / np.linalg.norm(v)


def _query_embedding_dir(cache_root: Path, embedding_model_name: str, query_set_id: str) -> Path:
    """Query embeddings live outside the formal L1-L5 scheme (Brief §5.3's
    L2 key is chunking-specific; queries don't depend on chunking at all) —
    keyed only by embedding model + query set, computed once per experiment
    dataset during the offline build_index.py phase (Brief D1).
    """
    return Path(cache_root) / "query_embeddings" / embedding_model_name / query_set_id


def embed_queries_cached(
    embedder: EmbeddingModel,
    query_set_id: str,
    query_ids: list[str],
    questions: list[str],
    cache_root: Path,
) -> None:
    dest = _query_embedding_dir(cache_root, embedder.name, query_set_id)
    if (dest / "vectors.npy").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    vectors = embedder.encode(questions, show_progress=False)
    np.save(dest / "vectors.npy", vectors)
    (dest / "query_ids.json").write_text(json.dumps(query_ids))


def load_query_vectors(
    cache_root: Path, embedding_model_name: str, query_set_id: str
) -> dict[str, np.ndarray]:
    """Load query embeddings cached by :func:`embed_queries_cached` — no
    embedder needed, so ``evaluate()`` never touches it (Brief D1).
    """
    dest = _query_embedding_dir(cache_root, embedding_model_name, query_set_id)
    vectors = np.load(dest / "vectors.npy")
    query_ids = json.loads((dest / "query_ids.json").read_text())
    return dict(zip(query_ids, vectors, strict=True))

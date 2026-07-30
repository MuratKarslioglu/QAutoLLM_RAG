"""Cross-encoder rerank of the top-k retrieved chunks (Brief §7).

Small model (~0.5GB) — construct via
:meth:`qautollm.core.model_manager.ModelManager.load_small`, it may
co-reside with the resident generator (Brief D1).
"""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import CrossEncoder

from qautollm.core.config import RerankerModelSpec, load_model_catalog
from qautollm.rag.retrieval import RetrievedDoc

_CATALOG_PATH = Path(__file__).resolve().parents[2] / "configs" / "models" / "reranker.yaml"


def _catalog() -> dict[str, RerankerModelSpec]:
    return load_model_catalog(_CATALOG_PATH, RerankerModelSpec)


class Reranker:
    def __init__(self, name: str = "default", device: str = "cpu") -> None:
        spec = _catalog()[name]
        self.name = name
        self._model = CrossEncoder(spec.hf_id, device=device)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDoc],
        chunk_text_by_id: dict[str, str],
        top_n: int,
    ) -> list[RetrievedDoc]:
        if not candidates:
            return []
        pairs = [(query, chunk_text_by_id[c.chunk_id]) for c in candidates]
        # One rerank() call per query per trial — a per-call "Batches: 1/1"
        # bar here would flood the terminal under the *one* overall trial
        # progress bar (Orchestrator.run); the orchestrator's bar is the
        # only one that should be visible.
        scores = self._model.predict(pairs, show_progress_bar=False)
        reranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: -pair[1])
        return [RetrievedDoc(chunk_id=c.chunk_id, score=float(s)) for c, s in reranked[:top_n]]

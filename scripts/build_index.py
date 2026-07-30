#!/usr/bin/env python3
"""Pre-build every (chunking x embedding) index + cached query embeddings
for a dataset, so ``run_experiment.py`` never needs the embedder resident
(Brief D1: embed offline, unload embedder, load generator once).

Usage: called from a small driver script that constructs a
:class:`qautollm.evaluation.evaluator.Dataset` and calls :func:`build_all` —
kept as a library function (not just a __main__ block) since a Dataset
object, not a file path, is the natural input until dataset loading (Brief
§12 Q2) is decided.
"""

from __future__ import annotations

import json
from pathlib import Path

from qautollm.core.config import EmbeddingModelSpec, load_model_catalog
from qautollm.core.model_manager import ModelManager
from qautollm.evaluation.evaluator import Dataset
from qautollm.rag.chunking import chunk_document
from qautollm.rag.embedding import EmbeddingModel, embed_queries_cached
from qautollm.rag.indexing import (
    build_dense_index,
    build_sparse_index,
    save_dense_index,
    save_sparse_index,
)
from qautollm.store.artifact_store import ArtifactStore
from qautollm.store.db import ArtifactDB
from qautollm.utils.logging import get_logger
from qautollm.utils.profiling import progress
from qautollm.utils.seeds import seed_everything

_logger = get_logger(__name__)

EMBEDDING_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "models" / "embedding.yaml"
)

CHUNK_SIZES = (256, 384, 512, 768)
CHUNK_OVERLAPS = (0, 32, 64, 96)
CHUNKING_STRATEGIES = ("fixed", "sentence", "semantic")


def _chunking_combos() -> list[tuple[str, int, int]]:
    return [
        (strategy, size, overlap)
        for strategy in CHUNKING_STRATEGIES
        for size in CHUNK_SIZES
        for overlap in CHUNK_OVERLAPS
        if overlap < size
    ]


def build_all(
    dataset: Dataset,
    artifacts_root: Path,
    db_path: Path,
    query_set_id: str,
    embedder_cls: type = EmbeddingModel,
    device: str = "cpu",
) -> ArtifactStore:
    """``embedder_cls`` defaults to the real sentence-transformers wrapper;
    pass :class:`qautollm.rag.embedding.SyntheticEmbeddingModel` for toy-set
    dry runs (Brief P3 gate) that must not require a GPU or network.

    ``device`` defaults to CPU (safe for the toy path's fake embedder, which
    ignores it anyway). Real callers must pass ``device="cuda"`` when a GPU
    is available -- 2026-07-18: found live that this call site was
    constructing the real embedder with no device argument at all, silently
    falling back to CPU-only sentence-transformers encoding even with a GPU
    idle and free (observed: ~16 texts/s, turning the one-time build_index
    pass into hours instead of the expected tens of minutes). Brief D1's
    "one large model resident at a time" is unaffected either way -- the
    embedder is always fully unloaded (``manager.unload_large()`` below,
    which also frees the CUDA cache) before this function returns, strictly
    before the generator is ever loaded by the caller.
    """
    seed_everything(0)  # no randomness happens here; kept for the D4 entry-point convention
    store = ArtifactStore(artifacts_root, ArtifactDB(db_path))
    manager = ModelManager()
    embedding_catalog = load_model_catalog(EMBEDDING_CATALOG_PATH, EmbeddingModelSpec)
    combos = _chunking_combos()

    for embedding_name in embedding_catalog:
        embedder = manager.load_large(
            f"embedder:{embedding_name}",
            lambda n=embedding_name, cls=embedder_cls, d=device: cls(n, device=d),
        )

        for strategy, size, overlap in progress(combos, desc=f"{embedding_name}: chunk x embed"):
            l1_key = {
                "corpus_id": dataset.corpus_id,
                "chunking_strategy": strategy,
                "chunk_size": size,
                "chunk_overlap": overlap,
            }

            def _compute_l1(dest: Path, strategy=strategy, size=size, overlap=overlap) -> None:
                records = [
                    {"chunk_id": f"{doc.doc_id}::{i}", "text": text}
                    for doc in dataset.documents
                    for i, text in enumerate(chunk_document(doc.text, strategy, size, overlap))
                ]
                (dest / "chunks.json").write_text(json.dumps(records))

            l1 = store.get_or_compute("L1", l1_key, _compute_l1)
            records = json.loads((l1.value / "chunks.json").read_text())
            chunk_ids = [r["chunk_id"] for r in records]
            chunk_texts = [r["text"] for r in records]

            l2_key = {**l1_key, "embedding_model": embedding_name}

            def _compute_l2(
                dest: Path, chunk_ids=chunk_ids, chunk_texts=chunk_texts, embedder=embedder
            ) -> None:
                vectors = embedder.encode(chunk_texts, show_progress=False)
                save_dense_index(build_dense_index(vectors, chunk_ids), dest)
                save_sparse_index(build_sparse_index(chunk_texts, chunk_ids), dest)

            store.get_or_compute("L2", l2_key, _compute_l2)

        embed_queries_cached(
            embedder,
            query_set_id,
            [q.query_id for q in dataset.queries],
            [q.question for q in dataset.queries],
            artifacts_root,
        )

    manager.unload_large()
    _logger.info("build_index done: %s", store.report())
    return store

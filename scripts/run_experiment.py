#!/usr/bin/env python3
"""Main entry point: ``python -m scripts.run_experiment --config path.yaml
[--resume]`` (Brief §6 data flow).

Two dataset backends are wired up (Brief §12 Q2, resolved):
  - ``dataset.id: toy`` — fully-synthetic, fake generator/reranker/
    faithfulness components, no GPU/network (Brief P3 gate).
  - ``dataset.id: hotpotqa`` — real HotpotQA (prepared by
    ``scripts/prepare_hotpotqa.py``), real embedding/reranker/faithfulness
    models, and the real generator from ``config.generator.model``
    (Brief §12 Q1's default: Qwen2.5-3B-Instruct, 4-bit) via
    :class:`~qautollm.core.model_manager.ModelManager` (Brief D1). This
    path needs a GPU and downloads several GB on first use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import scripts.build_index as build_index
from qautollm.core.config import EmbeddingModelSpec, ExperimentConfig, load_model_catalog
from qautollm.core.model_manager import ModelManager
from qautollm.core.orchestrator import Orchestrator
from qautollm.evaluation.evaluator import Evaluator
from qautollm.evaluation.faithfulness import FaithfulnessScorer
from qautollm.evaluation.hotpotqa_dataset import load_hotpotqa_dataset, load_hotpotqa_splits
from qautollm.evaluation.toy_dataset import build_toy_dataset
from qautollm.rag.embedding import EmbeddingModel, SyntheticEmbeddingModel, load_query_vectors
from qautollm.rag.generation import GenerationResult, Generator
from qautollm.rag.reranking import Reranker
from qautollm.rag.retrieval import RetrievedDoc
from qautollm.space.search_space import SearchSpace
from qautollm.store.artifact_store import ArtifactStore
from qautollm.store.db import ArtifactDB
from qautollm.utils.logging import add_file_logging, get_logger
from qautollm.utils.profiling import disk_probe, vram_probe

_logger = get_logger(__name__)

_HOTPOTQA_DATA_DIR = Path("data/hotpotqa")
_HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"
# Qwen2.5-3B-Instruct's fp16 download (quantized to 4-bit only after loading
# into memory) is ~6-7GB; leave margin for the reranker/NLI models and
# in-flight reconstruction overhead. Checked *before* build_index runs, not
# just before the download itself, so a too-full disk fails in seconds
# instead of after a lengthy (real-embedding) build_index pass — this is
# exactly the failure seen in practice: build_index finished fine, then the
# generator download crashed 30+ minutes later with a bare
# "No space left on device" from deep inside huggingface_hub.
_MIN_FREE_DISK_GB_FOR_GENERATOR = 15.0

BuiltEvaluator = tuple[Evaluator, ArtifactDB, str | None, list[str] | None]


def _check_disk_space_for_generator_download() -> None:
    probe_path = _HF_CACHE_DIR if _HF_CACHE_DIR.exists() else Path.home()
    disk = disk_probe(probe_path)
    if disk["free_gb"] < _MIN_FREE_DISK_GB_FOR_GENERATOR:
        raise RuntimeError(
            f"only {disk['free_gb']:.1f}GB free on the filesystem holding {probe_path} — "
            f"the generator download needs ~{_MIN_FREE_DISK_GB_FOR_GENERATOR:.0f}GB+ free "
            "to be safe. Free up space before retrying (e.g. `sudo snap remove` old "
            "revisions, `huggingface-cli delete-cache` for unused cached models) — any "
            "already-built chunking/embedding indexes stay cached, so a retry skips "
            "straight to the generator download instead of redoing build_index."
        )


class _FakeGenerator:
    name = "fake-gen"
    prompt_template_id = "qwen_chat"

    def encode(self, text: str) -> list[int]:
        return text.split()

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        words = prompt.split()
        answer = " ".join(words[-3:]) if words else ""
        return GenerationResult(text=answer, prompt_tokens=len(words), generated_tokens=3)


class _FakeReranker:
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDoc],
        chunk_text_by_id: dict[str, str],
        top_n: int,
    ) -> list[RetrievedDoc]:
        return candidates[:top_n]


class _FakeFaithfulnessScorer:
    def score(self, answer: str, context: str) -> float:
        return 1.0 if answer and answer.split()[0] in context else 0.0


def build_evaluator(config: ExperimentConfig, run_dir: Path) -> BuiltEvaluator:
    """Returns ``(evaluator, db, query_set_id, split_query_ids)`` — the last
    two feed straight into ``Orchestrator(...)``. Public so
    ``scripts/run_ablation.py`` (Brief P8) can reuse this construction.
    """
    if config.dataset.id == "toy":
        return _build_toy_evaluator(config, run_dir)
    if config.dataset.id == "hotpotqa":
        return _build_hotpotqa_evaluator(config, run_dir)
    raise NotImplementedError(
        f"dataset {config.dataset.id!r} is not wired up yet — supported: 'toy', 'hotpotqa'."
    )


def _build_toy_evaluator(config: ExperimentConfig, run_dir: Path) -> BuiltEvaluator:
    artifacts_root = run_dir / "artifacts"
    db_path = run_dir / "db.sqlite"
    db = ArtifactDB(db_path)
    store = ArtifactStore(artifacts_root, db)

    dataset = build_toy_dataset()
    build_index.build_all(
        dataset,
        artifacts_root,
        db_path,
        config.dataset.optimization_split,
        SyntheticEmbeddingModel,
    )

    embedding_catalog = load_model_catalog(build_index.EMBEDDING_CATALOG_PATH, EmbeddingModelSpec)
    query_vectors = {
        name: load_query_vectors(artifacts_root, name, config.dataset.optimization_split)
        for name in embedding_catalog
    }

    evaluator = Evaluator(
        store=store,
        dataset=dataset,
        query_vectors=query_vectors,
        generator=_FakeGenerator(),
        reranker=_FakeReranker(),
        faithfulness_scorer=_FakeFaithfulnessScorer(),
        max_new_tokens=config.generator.max_new_tokens,
    )
    return evaluator, db, None, None


def _build_hotpotqa_evaluator(config: ExperimentConfig, run_dir: Path) -> BuiltEvaluator:
    if not (_HOTPOTQA_DATA_DIR / "dataset.json").exists():
        raise FileNotFoundError(
            f"{_HOTPOTQA_DATA_DIR} not prepared — run "
            "`python -m scripts.prepare_hotpotqa` first (Brief §12 Q2)."
        )
    _check_disk_space_for_generator_download()
    dataset = load_hotpotqa_dataset(_HOTPOTQA_DATA_DIR)
    splits = load_hotpotqa_splits(_HOTPOTQA_DATA_DIR)
    # all queries share one embedding cache; splits narrow at evaluate() time
    query_set_id = "hotpotqa_all"

    artifacts_root = run_dir / "artifacts"
    db_path = run_dir / "db.sqlite"
    db = ArtifactDB(db_path)
    store = ArtifactStore(artifacts_root, db)

    # computed once, up front, and reused for both the embedder (below) and
    # the generator (further down) -- one vram_probe() call, one source of
    # truth for "is a GPU actually available right now".
    device = "cuda" if vram_probe().get("available") else "cpu"
    build_index.build_all(
        dataset, artifacts_root, db_path, query_set_id, EmbeddingModel, device=device
    )

    embedding_catalog = load_model_catalog(build_index.EMBEDDING_CATALOG_PATH, EmbeddingModelSpec)
    query_vectors = {
        name: load_query_vectors(artifacts_root, name, query_set_id) for name in embedding_catalog
    }

    manager = ModelManager()
    generator = manager.load_large(
        f"generator:{config.generator.model}",
        lambda: Generator(config.generator.model, device=device),
    )
    reranker = manager.load_small("reranker", lambda: Reranker("default", device="cpu"))
    faithfulness_scorer = manager.load_small(
        "faithfulness", lambda: FaithfulnessScorer("default", device="cpu")
    )

    evaluator = Evaluator(
        store=store,
        dataset=dataset,
        query_vectors=query_vectors,
        generator=generator,
        reranker=reranker,
        faithfulness_scorer=faithfulness_scorer,
        max_new_tokens=config.generator.max_new_tokens,
    )
    split_query_ids = splits[config.dataset.optimization_split]
    return evaluator, db, query_set_id, split_query_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    search_space = SearchSpace.from_yaml(config.search_space)
    run_dir = Path("results") / config.experiment.name
    add_file_logging(run_dir / "run.log")

    evaluator, db, query_set_id, split_query_ids = build_evaluator(config, run_dir)
    orchestrator = Orchestrator(
        config,
        search_space,
        evaluator,
        db,
        run_dir,
        query_set_id=query_set_id,
        split_query_ids=split_query_ids,
    )
    orchestrator.run(resume=args.resume)
    _logger.info("done: %s", evaluator.cache_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

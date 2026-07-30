"""Runs one CanonicalConfig over a query set -> EvalResult (Brief §5.5).

Only L3 (retrieval) and L4 (generation) are computed lazily here, per
trial — they depend on config fields (top_k, rerank, context_budget,
generator) that vary every trial, so precomputing all combinations isn't
feasible. L1 (chunk lists) and L2 (dense+BM25 indexes) must already be
populated by ``scripts/build_index.py``: Brief D1 keeps the embedder out of
the optimization loop entirely, so a miss on L1/L2 here is a setup error,
not something this evaluator can recover by embedding on the fly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qautollm.evaluation.answer_metrics import exact_match, token_f1
from qautollm.evaluation.faithfulness import FaithfulnessScorer
from qautollm.evaluation.retrieval_metrics import mrr, ndcg_at_k, recall_at_k
from qautollm.evaluation.system_metrics import collect_system_metrics
from qautollm.rag.generation import Generator
from qautollm.rag.indexing import load_dense_index, load_sparse_index
from qautollm.rag.prompting import build_prompt, parse_structured_answer
from qautollm.rag.reranking import Reranker
from qautollm.rag.retrieval import retrieve
from qautollm.space.encoding import CanonicalConfig
from qautollm.store.artifact_store import ArtifactStore


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    question: str
    gold_answer: str
    # Document ids, not chunk ids: a document's chunk_ids depend on the
    # trial's chunking_strategy/chunk_size/chunk_overlap, so they can't be
    # fixed at dataset-build time. Resolved to actual chunk_ids per trial in
    # Evaluator._evaluate_one (Brief: real datasets like HotpotQA split a
    # document into a config-dependent number of chunks; the toy dataset's
    # 1-chunk-per-doc case made this distinction invisible until real data).
    gold_evidence_doc_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Dataset:
    corpus_id: str
    documents: tuple[Document, ...]
    queries: tuple[QueryExample, ...]


@dataclass(frozen=True)
class EvalResult:
    # minimization convention, documented once (Brief §5.5): lower is better
    # for every entry, so -F1 and -faithfulness sit alongside +latency.
    objectives: np.ndarray  # [-mean_f1, -mean_faithfulness, mean_latency_s]
    em: float
    recall_at_k: float
    ndcg: float
    mrr: float
    tokens_prompt: int
    tokens_generated: int
    peak_vram_mb: float
    per_query: list[dict[str, Any]]
    cache_report: dict[str, dict[str, float | int]]


def _l1_miss(_dest: Path) -> None:
    raise RuntimeError(
        "L1 cache miss during evaluation: run scripts/build_index.py to "
        "pre-build chunking artifacts before run_experiment.py (Brief D1)."
    )


def _l2_miss(_dest: Path) -> None:
    raise RuntimeError(
        "L2 cache miss during evaluation: run scripts/build_index.py to "
        "pre-build (chunking, embedding) indexes before run_experiment.py (Brief D1)."
    )


class Evaluator:
    def __init__(
        self,
        store: ArtifactStore,
        dataset: Dataset,
        query_vectors: dict[str, dict[str, np.ndarray]],  # embedding_model -> query_id -> vector
        generator: Generator,
        reranker: Reranker | None,
        faithfulness_scorer: FaithfulnessScorer,
        retrieval_metric_k: int = 10,
        max_new_tokens: int = 64,
    ) -> None:
        self._store = store
        self._dataset = dataset
        self._query_vectors = query_vectors
        self._generator = generator
        self._reranker = reranker
        self._faithfulness_scorer = faithfulness_scorer
        self._retrieval_metric_k = retrieval_metric_k
        self._max_new_tokens = max_new_tokens

    def cache_report(self) -> dict[str, dict[str, float | int]]:
        return self._store.report()

    def evaluate(
        self,
        config: CanonicalConfig,
        query_set_id: str,
        split_query_ids: list[str] | None = None,
    ) -> EvalResult:
        raw = config.raw
        queries = self._dataset.queries
        if split_query_ids is not None:
            wanted = set(split_query_ids)
            queries = tuple(q for q in queries if q.query_id in wanted)

        l1_key = {
            "corpus_id": self._dataset.corpus_id,
            "chunking_strategy": raw["chunking_strategy"],
            "chunk_size": raw["chunk_size"],
            "chunk_overlap": raw["chunk_overlap"],
        }
        l1 = self._store.get_or_compute("L1", l1_key, _l1_miss)
        chunk_records = json.loads((l1.value / "chunks.json").read_text())
        chunk_text_by_id = {c["chunk_id"]: c["text"] for c in chunk_records}

        l2_key = {**l1_key, "embedding_model": raw["embedding_model"]}
        l2 = self._store.get_or_compute("L2", l2_key, _l2_miss)
        dense_index = load_dense_index(l2.value)
        sparse_index = load_sparse_index(l2.value)
        query_vectors = self._query_vectors[raw["embedding_model"]]

        per_query: list[dict[str, Any]] = []
        for q in queries:
            result = self._evaluate_one(
                q,
                query_set_id,
                raw,
                l2_key,
                chunk_text_by_id,
                dense_index,
                sparse_index,
                query_vectors,
            )
            per_query.append(result)

        return self._aggregate(per_query)

    def _evaluate_one(
        self,
        q: QueryExample,
        query_set_id: str,
        raw: dict[str, Any],
        l2_key: dict[str, Any],
        chunk_text_by_id: dict[str, str],
        dense_index: Any,
        sparse_index: Any,
        query_vectors: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        l3_key = {
            **l2_key,
            "query_set_id": query_set_id,
            "query_id": q.query_id,
            "top_k": raw["top_k"],
            "hybrid_on": raw["hybrid_on"],
            "dense_weight": raw.get("dense_weight"),
            "reranker_on": raw["reranker_on"],
            "reranker_top_n": raw.get("reranker_top_n"),
        }

        def _compute_l3() -> list[dict[str, Any]]:
            candidates = retrieve(
                query_vector=query_vectors[q.query_id],
                query_text=q.question,
                dense_index=dense_index,
                sparse_index=sparse_index,
                top_k=raw["top_k"],
                hybrid_on=bool(raw["hybrid_on"]),
                dense_weight=raw.get("dense_weight", 0.5),
            )
            if raw["reranker_on"] and self._reranker is not None:
                candidates = self._reranker.rerank(
                    q.question, candidates, chunk_text_by_id, raw["reranker_top_n"]
                )
            return [{"chunk_id": c.chunk_id, "score": c.score} for c in candidates]

        l3 = self._store.get_or_compute("L3", l3_key, _compute_l3)
        retrieved_ids = [r["chunk_id"] for r in l3.value]
        ranked_chunks = [chunk_text_by_id[cid] for cid in retrieved_ids]

        l4_key = {
            **l3_key,
            "context_budget": raw["context_budget"],
            "prompt_template_id": self._generator.prompt_template_id,
            "generator_id": self._generator.name,
            # max_new_tokens affects the generated text (a tighter budget
            # can truncate it), so it must be part of the cache key -- it
            # was previously hardcoded to a single constant everywhere,
            # which is why this wasn't needed before (2026-07-18).
            "max_new_tokens": self._max_new_tokens,
        }

        def _compute_l4() -> dict[str, Any]:
            prompt = build_prompt(
                self._generator.prompt_template_id,
                q.question,
                ranked_chunks,
                raw["context_budget"],
                self._generator,
            )
            result = self._generator.generate(prompt, max_new_tokens=self._max_new_tokens)
            return {
                "text": result.text,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
            }

        l4 = self._store.get_or_compute("L4", l4_key, _compute_l4)
        # `qwen_chat_evidence`-style templates emit "ANSWER: .. / EVIDENCE:
        # .. "; other templates have no such markers and fall back to
        # (raw_text, "") -- see parse_structured_answer's docstring. EM/F1
        # are graded on the short answer; faithfulness is scored against
        # the full-sentence evidence, since bare short spans were found to
        # be un-scoreable by NLI entailment regardless of NLI model size
        # (2026-07-18 investigation).
        answer, evidence = parse_structured_answer(l4.value["text"])
        context_text = "\n\n".join(ranked_chunks)

        # gold_evidence_doc_ids -> the chunk_ids that actually exist for
        # *this* trial's chunking (a doc can split into any number of
        # chunks depending on chunking_strategy/chunk_size/chunk_overlap).
        gold_chunk_ids = frozenset(
            cid for cid in chunk_text_by_id if cid.rsplit("::", 1)[0] in q.gold_evidence_doc_ids
        )

        return {
            "query_id": q.query_id,
            "answer": answer,
            "f1": token_f1(answer, q.gold_answer),
            "em": exact_match(answer, q.gold_answer),
            "faithfulness": (
                self._faithfulness_scorer.score(evidence, context_text)
                if context_text and evidence
                else 0.0
            ),
            "recall_at_k": recall_at_k(retrieved_ids, gold_chunk_ids, self._retrieval_metric_k),
            "ndcg": ndcg_at_k(retrieved_ids, gold_chunk_ids, self._retrieval_metric_k),
            "mrr": mrr(retrieved_ids, gold_chunk_ids),
            "prompt_tokens": l4.value["prompt_tokens"],
            "generated_tokens": l4.value["generated_tokens"],
            "latency_s": l4.latency_s,
        }

    def _aggregate(self, per_query: list[dict[str, Any]]) -> EvalResult:
        if not per_query:
            objectives = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            return EvalResult(
                objectives=objectives,
                em=0.0,
                recall_at_k=0.0,
                ndcg=0.0,
                mrr=0.0,
                tokens_prompt=0,
                tokens_generated=0,
                peak_vram_mb=0.0,
                per_query=[],
                cache_report=self._store.report(),
            )

        def _mean(field_name: str) -> float:
            return float(np.mean([p[field_name] for p in per_query]))

        total_prompt_tokens = sum(p["prompt_tokens"] for p in per_query)
        total_generated_tokens = sum(p["generated_tokens"] for p in per_query)
        mean_latency = _mean("latency_s")
        system = collect_system_metrics(
            mean_latency, total_prompt_tokens, total_generated_tokens
        )
        objectives = np.array(
            [-_mean("f1"), -_mean("faithfulness"), mean_latency], dtype=np.float64
        )

        return EvalResult(
            objectives=objectives,
            em=_mean("em"),
            recall_at_k=_mean("recall_at_k"),
            ndcg=_mean("ndcg"),
            mrr=_mean("mrr"),
            tokens_prompt=total_prompt_tokens,
            tokens_generated=total_generated_tokens,
            peak_vram_mb=system.peak_vram_mb,
            per_query=per_query,
            cache_report=self._store.report(),
        )

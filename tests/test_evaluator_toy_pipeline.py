"""End-to-end Evaluator test on a 10-query toy set (Brief P2 gate: "one
config evaluates end-to-end on a 10-query toy set; re-running is a 100%
cache hit and bit-identical; VRAM stays < 7GB").

Generator and faithfulness scorer are fakes here — this test is about the
Evaluator's orchestration and the Artifact Store's caching contract, not
about real model quality (that's covered separately by the network-dependent
``test_small_model_pipeline.py``, and left as the user's manual trial for
the actual generator).
"""

from __future__ import annotations

import json

import numpy as np

from qautollm.evaluation.evaluator import Dataset, Document, Evaluator, QueryExample
from qautollm.rag.chunking import chunk_document
from qautollm.rag.generation import GenerationResult
from qautollm.rag.indexing import (
    build_dense_index,
    build_sparse_index,
    save_dense_index,
    save_sparse_index,
)
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace
from qautollm.store.artifact_store import ArtifactStore
from qautollm.store.db import ArtifactDB

_RAG_V2_PATH = "configs/search_spaces/rag_v2.yaml"

_DOCS = tuple(
    Document(
        doc_id=f"d{i}",
        text=f"Synthetic document number {i} talks about topic {i} in detail. " * 3,
    )
    for i in range(5)
)
_QUERIES = tuple(
    QueryExample(
        query_id=f"q{i}",
        question=f"What does document {i % 5} say about topic {i % 5}?",
        gold_answer=f"topic {i % 5}",
        gold_evidence_doc_ids=frozenset({f"d{i % 5}"}),
    )
    for i in range(10)
)
_DATASET = Dataset(corpus_id="toy_corpus", documents=_DOCS, queries=_QUERIES)

_RAW_CONFIG = {
    "chunking_strategy": "fixed",
    "chunk_size": 256,
    "chunk_overlap": 0,
    "embedding_model": "bge-small",
    "top_k": 3,
    "hybrid_on": 1,
    "dense_weight": 0.5,
    "reranker_on": 0,
    "context_budget": 1024,
}


class _FakeGenerator:
    name = "fake-gen"
    prompt_template_id = "qwen_chat"

    def encode(self, text: str) -> list[int]:
        return text.split()

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        words = prompt.split()
        answer = " ".join(words[-3:]) if words else ""
        return GenerationResult(text=answer, prompt_tokens=len(words), generated_tokens=3)


class _FakeFaithfulnessScorer:
    def score(self, answer: str, context: str) -> float:
        return 1.0 if answer and answer.split()[0] in context else 0.0


def _build_store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts", ArtifactDB(tmp_path / "db.sqlite"))


def _prepopulate_l1_l2(store: ArtifactStore, raw: dict) -> None:
    l1_key = {
        "corpus_id": _DATASET.corpus_id,
        "chunking_strategy": raw["chunking_strategy"],
        "chunk_size": raw["chunk_size"],
        "chunk_overlap": raw["chunk_overlap"],
    }

    def _compute_l1(dest) -> None:
        records = [
            {"chunk_id": f"{doc.doc_id}::{i}", "text": text}
            for doc in _DATASET.documents
            for i, text in enumerate(
                chunk_document(
                    doc.text, raw["chunking_strategy"], raw["chunk_size"], raw["chunk_overlap"]
                )
            )
        ]
        (dest / "chunks.json").write_text(json.dumps(records))

    l1 = store.get_or_compute("L1", l1_key, _compute_l1)
    records = json.loads((l1.value / "chunks.json").read_text())
    chunk_ids = [r["chunk_id"] for r in records]
    chunk_texts = [r["text"] for r in records]

    l2_key = {**l1_key, "embedding_model": raw["embedding_model"]}

    def _compute_l2(dest) -> None:
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(len(chunk_ids), 16)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        save_dense_index(build_dense_index(vectors, chunk_ids), dest)
        save_sparse_index(build_sparse_index(chunk_texts, chunk_ids), dest)

    store.get_or_compute("L2", l2_key, _compute_l2)


def _query_vectors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(len(_QUERIES), 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return {q.query_id: v for q, v in zip(_QUERIES, vectors, strict=True)}


def test_evaluate_end_to_end_on_toy_set_and_cache_is_bit_identical_on_rerun(tmp_path):
    space = SearchSpace.from_yaml(_RAG_V2_PATH)
    config = encode(_RAW_CONFIG, space)

    store = _build_store(tmp_path)
    _prepopulate_l1_l2(store, _RAW_CONFIG)

    evaluator = Evaluator(
        store=store,
        dataset=_DATASET,
        query_vectors={"bge-small": _query_vectors()},
        generator=_FakeGenerator(),
        reranker=None,
        faithfulness_scorer=_FakeFaithfulnessScorer(),
    )

    result1 = evaluator.evaluate(config, query_set_id="toy_10")
    assert len(result1.per_query) == 10
    assert result1.peak_vram_mb < 7000  # Brief P2 gate: VRAM stays < 7GB

    report_after_first = store.report()
    assert report_after_first["L3"]["misses"] == 10  # one retrieval per query, all cold
    assert report_after_first["L4"]["misses"] == 10  # one generation per query, all cold

    result2 = evaluator.evaluate(config, query_set_id="toy_10")
    report_after_second = store.report()

    # Brief P2 gate: re-running is a 100% cache hit for the re-run itself.
    assert report_after_second["L3"]["misses"] == report_after_first["L3"]["misses"]
    assert report_after_second["L4"]["misses"] == report_after_first["L4"]["misses"]
    assert report_after_second["L3"]["hits"] == report_after_first["L3"]["hits"] + 10
    assert report_after_second["L4"]["hits"] == report_after_first["L4"]["hits"] + 10

    # Brief P2 gate: bit-identical.
    assert np.array_equal(result1.objectives, result2.objectives)
    for p1, p2 in zip(result1.per_query, result2.per_query, strict=True):
        assert p1["answer"] == p2["answer"]
        # hit reuses the stored cold latency, never re-measured:
        assert p1["latency_s"] == p2["latency_s"]


class _EvidenceFakeGenerator:
    """Emits ``qwen_chat_evidence``-style structured output so the toy
    pipeline can exercise the 2026-07-18 ANSWER/EVIDENCE routing without a
    real model (Brief P2 gate scope: Evaluator orchestration, not model
    quality).
    """

    name = "fake-gen-evidence"
    prompt_template_id = "qwen_chat_evidence"

    def encode(self, text: str) -> list[int]:
        return text.split()

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        text = "ANSWER: topic 0\nEVIDENCE: the document is about topic 0 in detail."
        return GenerationResult(
            text=text, prompt_tokens=len(prompt.split()), generated_tokens=len(text.split())
        )


class _RecordingFaithfulnessScorer:
    """Records exactly what it was asked to score, so the test can assert
    faithfulness is computed against the parsed *evidence*, not the raw
    generated text or the short answer.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def score(self, answer: str, context: str) -> float:
        self.calls.append((answer, context))
        return 1.0 if answer else 0.0


def test_evaluate_routes_answer_to_f1_em_and_evidence_to_faithfulness(tmp_path):
    space = SearchSpace.from_yaml(_RAG_V2_PATH)
    config = encode(_RAW_CONFIG, space)

    store = _build_store(tmp_path)
    _prepopulate_l1_l2(store, _RAW_CONFIG)

    scorer = _RecordingFaithfulnessScorer()
    evaluator = Evaluator(
        store=store,
        dataset=_DATASET,
        query_vectors={"bge-small": _query_vectors()},
        generator=_EvidenceFakeGenerator(),
        reranker=None,
        faithfulness_scorer=scorer,
    )

    result = evaluator.evaluate(config, query_set_id="toy_10")

    assert len(scorer.calls) == 10
    for scored_text, _context in scorer.calls:
        # never the raw "ANSWER: .. \nEVIDENCE: .." blob, never the bare
        # short answer -- always the parsed evidence sentence.
        assert scored_text == "the document is about topic 0 in detail."
    for p in result.per_query:
        assert p["answer"] == "topic 0"  # F1/EM graded on the short answer


def test_evaluate_falls_back_to_zero_faithfulness_when_template_has_no_evidence_marker(tmp_path):
    space = SearchSpace.from_yaml(_RAG_V2_PATH)
    config = encode(_RAW_CONFIG, space)

    store = _build_store(tmp_path)
    _prepopulate_l1_l2(store, _RAW_CONFIG)

    scorer = _RecordingFaithfulnessScorer()
    evaluator = Evaluator(
        store=store,
        dataset=_DATASET,
        query_vectors={"bge-small": _query_vectors()},
        generator=_FakeGenerator(),  # old-style: prompt_template_id == "qwen_chat"
        reranker=None,
        faithfulness_scorer=scorer,
    )

    result = evaluator.evaluate(config, query_set_id="toy_10")

    assert scorer.calls == []  # never called -- no evidence marker means evidence == ""
    for p in result.per_query:
        assert p["faithfulness"] == 0.0


def test_max_new_tokens_is_part_of_the_l4_cache_key(tmp_path):
    # A tighter/looser generation budget can change the generated text, so
    # a run with a different max_new_tokens must be a fresh L4 miss, not a
    # false cache hit serving text generated under the old budget
    # (2026-07-18: max_new_tokens was previously hardcoded, so this couldn't
    # happen in practice -- now that it's config-driven, it must be keyed).
    space = SearchSpace.from_yaml(_RAG_V2_PATH)
    config = encode(_RAW_CONFIG, space)

    store = _build_store(tmp_path)
    _prepopulate_l1_l2(store, _RAW_CONFIG)

    evaluator_64 = Evaluator(
        store=store,
        dataset=_DATASET,
        query_vectors={"bge-small": _query_vectors()},
        generator=_FakeGenerator(),
        reranker=None,
        faithfulness_scorer=_FakeFaithfulnessScorer(),
        max_new_tokens=64,
    )
    evaluator_64.evaluate(config, query_set_id="toy_10")
    report_after_64 = store.report()
    assert report_after_64["L4"]["misses"] == 10

    evaluator_96 = Evaluator(
        store=store,
        dataset=_DATASET,
        query_vectors={"bge-small": _query_vectors()},
        generator=_FakeGenerator(),
        reranker=None,
        faithfulness_scorer=_FakeFaithfulnessScorer(),
        max_new_tokens=96,
    )
    evaluator_96.evaluate(config, query_set_id="toy_10")
    report_after_96 = store.report()
    assert report_after_96["L4"]["misses"] == report_after_64["L4"]["misses"] + 10

"""Real-model integration test for embedding + retrieval + rerank +
faithfulness (Brief P2 gate). Downloads three small (<500MB total) HF
models on first run. The heavy piece this deliberately does NOT cover is
the generator (Qwen2.5-3B) — that stays the user's manual trial (Brief
D1: only the embedder/reranker/NLI models are small enough to verify here).
"""

import pytest

from qautollm.evaluation.faithfulness import FaithfulnessScorer
from qautollm.rag.embedding import EmbeddingModel
from qautollm.rag.indexing import build_dense_index, build_sparse_index
from qautollm.rag.reranking import Reranker
from qautollm.rag.retrieval import retrieve

CORPUS = {
    "d0": "The Eiffel Tower is located in Paris, France.",
    "d1": "Mount Everest is the tallest mountain on Earth.",
    "d2": "Python is a popular programming language for data science.",
    "d3": "Quantum computers use qubits instead of classical bits.",
}


@pytest.mark.slow
def test_embedding_retrieval_rerank_faithfulness_real_models():
    embedder = EmbeddingModel("bge-small", device="cpu")
    chunk_ids = list(CORPUS)
    chunk_texts = list(CORPUS.values())
    vectors = embedder.encode(chunk_texts, show_progress=False)

    dense = build_dense_index(vectors, chunk_ids)
    sparse = build_sparse_index(chunk_texts, chunk_ids)

    query = "Where is the Eiffel Tower?"
    query_vector = embedder.encode([query], show_progress=False)[0]

    docs = retrieve(
        query_vector=query_vector,
        query_text=query,
        dense_index=dense,
        sparse_index=sparse,
        top_k=2,
        hybrid_on=True,
        dense_weight=0.7,
    )
    assert docs[0].chunk_id == "d0"

    reranker = Reranker("default", device="cpu")
    reranked = reranker.rerank(query, docs, CORPUS, top_n=1)
    assert reranked[0].chunk_id == "d0"

    scorer = FaithfulnessScorer("default", device="cpu")
    faithful = scorer.score("The Eiffel Tower is in Paris.", CORPUS["d0"])
    unfaithful = scorer.score("The Eiffel Tower is in Berlin.", CORPUS["d0"])
    assert faithful > unfaithful

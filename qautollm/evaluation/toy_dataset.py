"""A small, fully synthetic Dataset for smoke-testing the pipeline
end-to-end (Brief P2/P3 gates: "10-query toy set") without depending on the
still-open real-dataset decision (Brief §12 Q2).
"""

from __future__ import annotations

from qautollm.evaluation.evaluator import Dataset, Document, QueryExample


def build_toy_dataset(n_queries: int = 10, n_docs: int = 5) -> Dataset:
    documents = tuple(
        Document(
            doc_id=f"d{i}",
            text=f"Synthetic document number {i} talks about topic {i} in detail. " * 3,
        )
        for i in range(n_docs)
    )
    queries = tuple(
        QueryExample(
            query_id=f"q{i}",
            question=f"What does document {i % n_docs} say about topic {i % n_docs}?",
            gold_answer=f"topic {i % n_docs}",
            gold_evidence_doc_ids=frozenset({f"d{i % n_docs}"}),
        )
        for i in range(n_queries)
    )
    return Dataset(corpus_id="toy_corpus", documents=documents, queries=queries)

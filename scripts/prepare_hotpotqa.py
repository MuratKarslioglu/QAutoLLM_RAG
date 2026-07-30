#!/usr/bin/env python3
"""Download + prepare HotpotQA (distractor) into one shared-corpus Dataset
with disjoint opt/test query-id splits (Brief §12 Q2 — resolved: write
download/subsetting code rather than assuming a local copy).

One shared corpus (union of every paragraph across *both* splits, deduped
by title) rather than HotpotQA's own per-question 10-paragraph distractor
set: retrieval needs to actually search among many candidate documents to
be a meaningful RAG evaluation, not just pick 2 out of a curated 10. Also
matches the toy dataset's own pattern — one Dataset, `split_query_ids`
narrows to a split at evaluate() time (Brief §5.3: L1/L2 keyed by
corpus_id, so opt/test must share one corpus or the cache gets corrupted
by whichever split's documents were chunked first).

Usage:
    python -m scripts.prepare_hotpotqa --n-opt 100 --n-test 300 \\
        --output-dir data/hotpotqa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from qautollm.evaluation.evaluator import Dataset, Document, QueryExample
from qautollm.evaluation.hotpotqa_dataset import DATASET_FILENAME, SPLITS_FILENAME
from qautollm.utils.logging import get_logger

_logger = get_logger(__name__)

HF_REPO = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"
# has public answers; train/test(hidden) don't fit a "never touched" gold eval
HF_SPLIT = "validation"


def _doc_id(title: str) -> str:
    return f"hotpot::{title}"


def _build_dataset(examples: list[dict[str, Any]], corpus_id: str) -> Dataset:
    documents: dict[str, str] = {}
    queries = []
    for ex in examples:
        for title, sentences in zip(
            ex["context"]["title"], ex["context"]["sentences"], strict=True
        ):
            doc_id = _doc_id(title)
            if doc_id not in documents:
                documents[doc_id] = " ".join(sentences)

        gold_doc_ids = frozenset(_doc_id(t) for t in ex["supporting_facts"]["title"])
        queries.append(
            QueryExample(
                query_id=f"hotpot_{ex['id']}",
                question=ex["question"],
                gold_answer=ex["answer"],
                gold_evidence_doc_ids=gold_doc_ids,
            )
        )

    doc_objs = tuple(Document(doc_id=d, text=t) for d, t in documents.items())
    return Dataset(corpus_id=corpus_id, documents=doc_objs, queries=tuple(queries))


def _dataset_to_json(dataset: Dataset) -> dict[str, Any]:
    return {
        "corpus_id": dataset.corpus_id,
        "documents": [{"doc_id": d.doc_id, "text": d.text} for d in dataset.documents],
        "queries": [
            {
                "query_id": q.query_id,
                "question": q.question,
                "gold_answer": q.gold_answer,
                "gold_evidence_doc_ids": sorted(q.gold_evidence_doc_ids),
            }
            for q in dataset.queries
        ],
    }


def prepare(n_opt: int, n_test: int, output_dir: str | Path) -> Path:
    total = n_opt + n_test
    _logger.info(
        "downloading %s/%s split=%s[:%d]", HF_REPO, HF_CONFIG, HF_SPLIT, total
    )
    raw = load_dataset(HF_REPO, HF_CONFIG, split=f"{HF_SPLIT}[:{total}]", trust_remote_code=False)

    opt_examples = list(raw.select(range(n_opt)))
    test_examples = list(raw.select(range(n_opt, total)))
    dataset = _build_dataset(opt_examples + test_examples, corpus_id="hotpotqa")

    opt_split = f"opt_{n_opt}"
    test_split = f"test_{n_test}"
    splits = {
        opt_split: [f"hotpot_{ex['id']}" for ex in opt_examples],
        test_split: [f"hotpot_{ex['id']}" for ex in test_examples],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / DATASET_FILENAME).write_text(json.dumps(_dataset_to_json(dataset)))
    (output_dir / SPLITS_FILENAME).write_text(json.dumps(splits))

    _logger.info(
        "wrote %d documents, %d queries (%s=%d, %s=%d) to %s",
        len(dataset.documents),
        len(dataset.queries),
        opt_split,
        len(splits[opt_split]),
        test_split,
        len(splits[test_split]),
        output_dir,
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-opt", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--output-dir", default="data/hotpotqa")
    args = parser.parse_args()
    prepare(args.n_opt, args.n_test, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

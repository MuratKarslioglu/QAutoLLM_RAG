"""Loads a HotpotQA ``Dataset`` + opt/test split manifest prepared by
``scripts/prepare_hotpotqa.py`` (Brief §12 Q2).

Kept deliberately dumb (just JSON I/O) — all the actual HuggingFace
download/corpus-building logic lives in the prepare script, run once
offline, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

from qautollm.evaluation.evaluator import Dataset, Document, QueryExample

DATASET_FILENAME = "dataset.json"
SPLITS_FILENAME = "splits.json"


def load_hotpotqa_dataset(data_dir: str | Path) -> Dataset:
    data_dir = Path(data_dir)
    raw = json.loads((data_dir / DATASET_FILENAME).read_text())
    documents = tuple(Document(doc_id=d["doc_id"], text=d["text"]) for d in raw["documents"])
    queries = tuple(
        QueryExample(
            query_id=q["query_id"],
            question=q["question"],
            gold_answer=q["gold_answer"],
            gold_evidence_doc_ids=frozenset(q["gold_evidence_doc_ids"]),
        )
        for q in raw["queries"]
    )
    return Dataset(corpus_id=raw["corpus_id"], documents=documents, queries=queries)


def load_hotpotqa_splits(data_dir: str | Path) -> dict[str, list[str]]:
    """``{split_name: [query_id, ...]}``, e.g. ``{"opt_100": [...], "test_300": [...]}``."""
    data_dir = Path(data_dir)
    return json.loads((data_dir / SPLITS_FILENAME).read_text())

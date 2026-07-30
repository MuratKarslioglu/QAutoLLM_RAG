"""Real HotpotQA download + preparation (Brief §12 Q2, resolved). Marked
slow: downloads a small slice of the real dataset from HuggingFace.
"""

from pathlib import Path

import pytest

from qautollm.evaluation.hotpotqa_dataset import load_hotpotqa_dataset, load_hotpotqa_splits
from scripts.prepare_hotpotqa import prepare


@pytest.mark.slow
def test_prepare_and_load_hotpotqa_small_sample(tmp_path: Path):
    output_dir = prepare(n_opt=3, n_test=3, output_dir=tmp_path)

    dataset = load_hotpotqa_dataset(output_dir)
    splits = load_hotpotqa_splits(output_dir)

    assert len(dataset.queries) == 6
    assert len(dataset.documents) > 0
    assert set(splits) == {"opt_3", "test_3"}
    assert len(splits["opt_3"]) == 3
    assert len(splits["test_3"]) == 3
    # opt/test must be disjoint (Brief: test split "never touched during optimization")
    assert set(splits["opt_3"]).isdisjoint(splits["test_3"])

    all_query_ids = {q.query_id for q in dataset.queries}
    assert all_query_ids == set(splits["opt_3"]) | set(splits["test_3"])

    for q in dataset.queries:
        assert q.question
        assert q.gold_answer
        assert q.gold_evidence_doc_ids  # every HotpotQA example has >=1 supporting fact title

    doc_ids = {d.doc_id for d in dataset.documents}
    for q in dataset.queries:
        # every gold doc referenced by a query must actually exist in the shared corpus
        assert q.gold_evidence_doc_ids <= doc_ids


@pytest.mark.slow
def test_prepared_dataset_is_deterministic(tmp_path: Path):
    dir_a = prepare(n_opt=2, n_test=2, output_dir=tmp_path / "a")
    dir_b = prepare(n_opt=2, n_test=2, output_dir=tmp_path / "b")

    dataset_a = load_hotpotqa_dataset(dir_a)
    dataset_b = load_hotpotqa_dataset(dir_b)

    assert [q.query_id for q in dataset_a.queries] == [q.query_id for q in dataset_b.queries]
    assert {d.doc_id for d in dataset_a.documents} == {d.doc_id for d in dataset_b.documents}

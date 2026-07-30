import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qautollm.evaluation.test_split_eval import evaluate_pareto_front_on_test_split
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace
from qautollm.store.db import ArtifactDB

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"
_SPACE = SearchSpace.from_yaml(RAG_V2_PATH)

_RAW_A = {
    "chunking_strategy": "fixed",
    "chunk_size": 256,
    "chunk_overlap": 0,
    "embedding_model": "bge-small",
    "top_k": 3,
    "hybrid_on": 0,
    "reranker_on": 0,
    "context_budget": 1024,
}
_RAW_B = {**_RAW_A, "top_k": 5}  # a different, mutually-non-dominated config
_RAW_DOMINATED = {**_RAW_A, "chunk_size": 384}  # will be given strictly worse objectives


class _FakeEvaluator:
    """Records which configs it was asked to evaluate; returns a fixed
    result keyed by config_hash so the test can check exactly which
    (Pareto-surviving) configs got re-evaluated on the test split.
    """

    def __init__(self):
        self.calls: list[tuple[str, list[str] | None]] = []

    def evaluate(self, config, query_set_id, split_query_ids=None):
        self.calls.append((config.config_hash, split_query_ids))
        return SimpleNamespace(
            objectives=np.array([-0.9, -0.9, 0.1]),
            em=0.5,
            recall_at_k=0.8,
            ndcg=0.7,
        )


def _write_trial(db: ArtifactDB, experiment_id, trial_index, raw, objectives, status="ok"):
    cfg = encode(raw, _SPACE)
    db.write_trial(
        {
            "experiment_id": experiment_id,
            "trial_index": trial_index,
            "config_hash": cfg.config_hash,
            "config_json": json.dumps(raw, sort_keys=True),
            "objective_f1": objectives[0],
            "objective_faithfulness": objectives[1],
            "objective_latency_s": objectives[2],
            "status": status,
        }
    )
    return cfg


def test_only_pareto_optimal_configs_get_reevaluated_on_test_split(tmp_path):
    db = ArtifactDB(tmp_path / "db.sqlite")
    experiment_id = "exp1"

    cfg_a = _write_trial(db, experiment_id, 0, _RAW_A, [-0.9, -0.1, 0.2])
    cfg_b = _write_trial(db, experiment_id, 1, _RAW_B, [-0.1, -0.9, 0.05])
    # dominated by cfg_a in every objective:
    _write_trial(db, experiment_id, 2, _RAW_DOMINATED, [-0.5, -0.05, 0.3])
    # a failed trial must be ignored entirely:
    _write_trial(db, experiment_id, 3, _RAW_A, [0.0, 0.0, 0.0], status="failed")

    evaluator = _FakeEvaluator()
    report = evaluate_pareto_front_on_test_split(
        db, experiment_id, _SPACE, evaluator, "qs1", ["t1", "t2"]
    )

    reevaluated_hashes = {r["config_hash"] for r in report}
    assert reevaluated_hashes == {cfg_a.config_hash, cfg_b.config_hash}

    # the dominated + failed trials must never reach the evaluator:
    called_hashes = {h for h, _ in evaluator.calls}
    assert called_hashes == {cfg_a.config_hash, cfg_b.config_hash}

    for _, split_query_ids in evaluator.calls:
        assert split_query_ids == ["t1", "t2"]  # always the *test* split, never opt

    for r in report:
        assert r["test_objectives"] == [-0.9, -0.9, 0.1]
        assert r["test_em"] == 0.5
        assert r["test_recall_at_k"] == 0.8
        assert r["test_ndcg"] == 0.7
        assert r["raw"] in (_RAW_A, _RAW_B)


def test_empty_trials_gives_empty_report(tmp_path):
    db = ArtifactDB(tmp_path / "db.sqlite")
    evaluator = _FakeEvaluator()
    report = evaluate_pareto_front_on_test_split(
        db, "no_such_experiment", _SPACE, evaluator, "qs1", ["t1"]
    )
    assert report == []
    assert evaluator.calls == []

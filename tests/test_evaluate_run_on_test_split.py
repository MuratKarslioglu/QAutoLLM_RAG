import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import scripts.evaluate_run_on_test_split as evaluate_run_on_test_split_module
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace
from qautollm.store.db import ArtifactDB
from scripts.evaluate_run_on_test_split import evaluate_run_on_test_split

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"
_SPACE = SearchSpace.from_yaml(RAG_V2_PATH)

_RAW = {
    "chunking_strategy": "fixed",
    "chunk_size": 256,
    "chunk_overlap": 0,
    "embedding_model": "bge-small",
    "top_k": 3,
    "hybrid_on": 0,
    "reranker_on": 0,
    "context_budget": 1024,
}


def _write_config(tmp_path: Path, experiment_name: str, dataset_id: str = "hotpotqa") -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "name": experiment_name,
                    "seed": 42,
                    "budget": 10,
                    "checkpoints": [10],
                },
                "dataset": {
                    "id": dataset_id,
                    "optimization_split": "opt_100",
                    "test_split": "test_300",
                },
                "optimizer": {"name": "qkernel_bo", "init_points": 5},
                "generator": {"model": "unused", "max_new_tokens": 64},
                "search_space": str(RAG_V2_PATH),
            }
        )
    )
    return config_path


class _FakeEvaluator:
    def evaluate(self, config, query_set_id, split_query_ids=None):
        return SimpleNamespace(
            objectives=np.array([-0.9, -0.9, 0.1]), em=0.5, recall_at_k=0.8, ndcg=0.7
        )


def _fake_build_evaluator_factory(split_query_ids):
    def _fake_build_evaluator(config, run_dir):
        db = ArtifactDB(run_dir / "db.sqlite")
        return _FakeEvaluator(), db, "qs1", split_query_ids

    return _fake_build_evaluator


def test_no_test_split_raises(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, "exp_no_split", dataset_id="toy")
    monkeypatch.setattr(
        evaluate_run_on_test_split_module, "build_evaluator", _fake_build_evaluator_factory(None)
    )
    with pytest.raises(ValueError, match="no test split wired up"):
        evaluate_run_on_test_split(str(config_path), run_dir=tmp_path / "run")


def test_no_trials_raises(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, "exp_empty")
    monkeypatch.setattr(
        evaluate_run_on_test_split_module,
        "build_evaluator",
        _fake_build_evaluator_factory(["t1", "t2"]),
    )
    with pytest.raises(ValueError, match="no trials found"):
        evaluate_run_on_test_split(str(config_path), run_dir=tmp_path / "run")


def test_happy_path_writes_report(tmp_path, monkeypatch):
    experiment_name = "exp_ok"
    config_path = _write_config(tmp_path, experiment_name)
    run_dir = tmp_path / "run"

    monkeypatch.setattr(
        evaluate_run_on_test_split_module,
        "build_evaluator",
        _fake_build_evaluator_factory(["t1", "t2"]),
    )
    monkeypatch.setattr(
        evaluate_run_on_test_split_module,
        "load_hotpotqa_splits",
        lambda path: {"test_300": ["t1", "t2"]},
    )

    db = ArtifactDB(run_dir / "db.sqlite")
    cfg = encode(_RAW, _SPACE)
    db.write_trial(
        {
            "experiment_id": experiment_name,
            "trial_index": 0,
            "config_hash": cfg.config_hash,
            "config_json": json.dumps(_RAW, sort_keys=True),
            "objective_f1": -0.9,
            "objective_faithfulness": -0.9,
            "objective_latency_s": 0.1,
            "status": "ok",
        }
    )

    output_path = evaluate_run_on_test_split(str(config_path), run_dir=run_dir)

    report = json.loads(output_path.read_text())
    assert len(report) == 1
    assert report[0]["config_hash"] == cfg.config_hash
    assert report[0]["test_em"] == 0.5

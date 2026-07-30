import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qautollm.core.config import (
    DatasetConfig,
    ExperimentConfig,
    ExperimentMeta,
    GeneratorConfig,
    OptimizerConfig,
)
from qautollm.core.orchestrator import Orchestrator
from qautollm.optimizers.base import OptimizerInterface
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace
from qautollm.store.db import ArtifactDB
from qautollm.utils.seeds import make_rng

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"
_SPACE = SearchSpace.from_yaml(RAG_V2_PATH)


class _DeterministicEvaluator:
    """Pure function of config_hash -> objectives; no caching, no real models."""

    def evaluate(self, config, query_set_id, split_query_ids=None):
        seed = int(hashlib.sha256(config.config_hash.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        objectives = rng.uniform(0, 1, size=3)
        return SimpleNamespace(
            objectives=objectives,
            em=0.0,
            recall_at_k=0.0,
            ndcg=0.0,
            tokens_prompt=0,
            tokens_generated=0,
            peak_vram_mb=0.0,
            per_query=[],
            cache_report={},
        )

    def cache_report(self):
        return {}


class _FlakyEvaluator(_DeterministicEvaluator):
    """Fails deterministically on specific 1-based evaluate() call counts."""

    def __init__(self, fail_on: set[int]):
        self._fail_on = fail_on
        self._count = 0

    def evaluate(self, config, query_set_id, split_query_ids=None):
        self._count += 1
        if self._count in self._fail_on:
            raise RuntimeError("simulated evaluation failure")
        return super().evaluate(config, query_set_id, split_query_ids)


class _DuplicateOnceOptimizer(OptimizerInterface):
    """Suggests the shared init design, then deliberately re-suggests its
    last point exactly once (a guaranteed duplicate), then falls back to
    normal random sampling — for testing the orchestrator's duplicate
    handling without waiting on a natural collision.
    """

    def __init__(self, search_space, seed, init_design):
        super().__init__(search_space, seed, init_design)
        self._rng = make_rng(seed)
        self._last = None
        self._has_duplicated = False

    def suggest(self):
        init = self._next_init_point()
        if init is not None:
            self._last = init
            return init
        if not self._has_duplicated:
            self._has_duplicated = True
            return self._last
        raw = self.search_space.sample(self._rng)
        self._last = encode(raw, self.search_space)
        return self._last

    def update(self, config, objectives):
        pass

    def state_dict(self):
        return {
            "init_cursor": self._init_cursor,
            "has_duplicated": self._has_duplicated,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        self._init_cursor = state["init_cursor"]
        self._has_duplicated = state["has_duplicated"]
        self._rng.bit_generator.state = state["rng_state"]


class _DupOrchestrator(Orchestrator):
    def _build_optimizer(self, init_design):
        return _DuplicateOnceOptimizer(self.search_space, self.config.experiment.seed, init_design)


def _config(
    name: str, budget: int, seed: int = 42, optimizer: str = "random", init_points: int = 5
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentMeta(name=name, seed=seed, budget=budget, checkpoints=[budget]),
        dataset=DatasetConfig(id="toy", optimization_split="toy_10", test_split="toy_10"),
        optimizer=OptimizerConfig(name=optimizer, init_points=init_points),
        generator=GeneratorConfig(model="fake"),
        search_space=str(RAG_V2_PATH),
    )


def _rows(db: ArtifactDB, experiment_id: str):
    return db.read_trials(experiment_id)


def test_budget_accounting_stops_exactly_at_budget(tmp_path):
    config = _config("basic", budget=8)
    db = ArtifactDB(tmp_path / "db.sqlite")
    orch = Orchestrator(config, _SPACE, _DeterministicEvaluator(), db, tmp_path / "run")
    orch.run(resume=False)

    rows = _rows(db, "basic")
    assert len(rows) == 8
    assert all(r["duplicate_flag"] == 0 for r in rows)
    assert all(r["status"] == "ok" for r in rows)
    assert [r["trial_index"] for r in rows] == list(range(8))


def test_failed_trial_recorded_and_run_continues(tmp_path):
    config = _config("flaky", budget=6)
    db = ArtifactDB(tmp_path / "db.sqlite")
    orch = Orchestrator(config, _SPACE, _FlakyEvaluator(fail_on={3}), db, tmp_path / "run")
    orch.run(resume=False)

    rows = _rows(db, "flaky")
    assert len(rows) == 6  # a failed trial still consumed budget: it was genuinely attempted
    statuses = [r["status"] for r in rows]
    assert statuses.count("failed") == 1
    assert statuses[2] == "failed"
    assert rows[2]["objective_f1"] is None


def test_duplicate_suggestion_costs_zero_budget_but_is_logged(tmp_path):
    config = _config("dup", budget=8, init_points=5)
    db = ArtifactDB(tmp_path / "db.sqlite")
    orch = _DupOrchestrator(config, _SPACE, _DeterministicEvaluator(), db, tmp_path / "run")
    orch.run(resume=False)

    rows = _rows(db, "dup")
    assert len(rows) == 9  # 5 init + 1 forced duplicate + 3 more to reach budget=8
    duplicate_rows = [r for r in rows if r["duplicate_flag"] == 1]
    assert len(duplicate_rows) == 1
    assert duplicate_rows[0]["config_hash"] == rows[4]["config_hash"]
    assert sum(r["duplicate_flag"] == 0 for r in rows) == 8


class _FailOnceOptimizer(OptimizerInterface):
    """Drains the init design normally, then raises exactly once on a given
    (1-based, counting every ``suggest()`` call) call -- simulating a
    GP-fit/EHVI numerical failure -- before resuming normal random
    sampling. Used to test the orchestrator's suggest()-failure fallback
    without depending on a real BoTorch failure mode.
    """

    def __init__(self, search_space, seed, init_design, fail_on: set[int]):
        super().__init__(search_space, seed, init_design)
        self._rng = make_rng(seed)
        self._fail_on = fail_on
        self._suggest_calls = 0

    def suggest(self):
        self._suggest_calls += 1
        init = self._next_init_point()
        if init is not None:
            return init
        if self._suggest_calls in self._fail_on:
            raise RuntimeError("simulated GP-fit failure")
        raw = self.search_space.sample(self._rng)
        return encode(raw, self.search_space)

    def update(self, config, objectives):
        pass

    def state_dict(self):
        return {"init_cursor": self._init_cursor, "rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, state):
        self._init_cursor = state["init_cursor"]
        self._rng.bit_generator.state = state["rng_state"]


class _FailOnceOrchestrator(Orchestrator):
    def _build_optimizer(self, init_design):
        return _FailOnceOptimizer(
            self.search_space, self.config.experiment.seed, init_design, fail_on={6}
        )


def test_suggest_failure_falls_back_to_random_instead_of_crashing(tmp_path):
    """Regression test: optimizer.suggest() raising used to crash the whole
    run uncaught (unlike evaluate(), which already degrades gracefully) --
    the observed real-world symptom was a run silently stopping mid-budget
    with no error recorded anywhere.
    """
    config = _config("suggest_flaky", budget=8, init_points=5)
    db = ArtifactDB(tmp_path / "db.sqlite")
    orch = _FailOnceOrchestrator(config, _SPACE, _DeterministicEvaluator(), db, tmp_path / "run")
    orch.run(resume=False)  # must not raise despite the injected suggest() failure

    rows = _rows(db, "suggest_flaky")
    assert len(rows) == 8  # full budget still reached
    assert all(r["status"] == "ok" for r in rows)  # evaluate() itself never failed


def _extract(rows):
    return [
        (
            r["config_hash"],
            r["objective_f1"],
            r["objective_faithfulness"],
            r["objective_latency_s"],
            r["duplicate_flag"],
        )
        for r in rows
    ]


@pytest.mark.parametrize("optimizer_name", ["random", "tpe"])
def test_resume_reproduces_uninterrupted_run(tmp_path, optimizer_name):
    budget = 12

    config_golden = _config(f"golden_{optimizer_name}", budget=budget, optimizer=optimizer_name)
    db_golden = ArtifactDB(tmp_path / "golden.sqlite")
    orch_golden = Orchestrator(
        config_golden, _SPACE, _DeterministicEvaluator(), db_golden, tmp_path / "golden"
    )
    orch_golden.run(resume=False)
    golden = _extract(_rows(db_golden, config_golden.experiment.name))

    run_name = f"resumed_{optimizer_name}"
    run_dir = tmp_path / "resumed_run"
    db_path = tmp_path / "resumed.sqlite"

    config_partial = _config(run_name, budget=6, optimizer=optimizer_name)
    orch_partial = Orchestrator(
        config_partial, _SPACE, _DeterministicEvaluator(), ArtifactDB(db_path), run_dir
    )
    orch_partial.run(resume=False)  # simulates the process that got killed -9 after 6 trials

    # A brand-new Orchestrator object, reopening the same DB/run_dir from
    # scratch — nothing carried over in memory, everything comes from disk.
    config_full = _config(run_name, budget=budget, optimizer=optimizer_name)
    db_reopened = ArtifactDB(db_path)
    orch_resumed = Orchestrator(
        config_full, _SPACE, _DeterministicEvaluator(), db_reopened, run_dir
    )
    orch_resumed.run(resume=True)
    resumed = _extract(_rows(db_reopened, run_name))

    assert resumed == golden

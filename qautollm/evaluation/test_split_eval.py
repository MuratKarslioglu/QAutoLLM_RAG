"""Re-score a finished optimizer run's Pareto front on the held-out test
split (Brief §7: test_split is "never touched during optimization" — these
are the numbers that actually belong in the paper, not the
optimization-split numbers).

Shared by ``scripts/run_real_ablation.py`` (fresh multi-optimizer campaign)
and ``scripts/evaluate_run_on_test_split.py`` (re-score one already-finished
run without re-running optimization).
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import numpy as np

from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace
from qautollm.store.db import ArtifactDB
from qautollm.surrogate.pareto_archive import ParetoArchive


class _Evaluator(Protocol):
    def evaluate(self, config: Any, query_set_id: str, split_query_ids: list[str]) -> Any: ...


def evaluate_pareto_front_on_test_split(
    db: ArtifactDB,
    experiment_id: str,
    search_space: SearchSpace,
    evaluator: _Evaluator,
    query_set_id: str,
    test_query_ids: list[str],
) -> list[dict[str, Any]]:
    """Rebuild the Pareto front from this optimizer's completed trials, then
    re-evaluate each surviving config on the held-out test split.
    """
    trials = [t for t in db.read_trials(experiment_id) if t["status"] == "ok"]
    archive = ParetoArchive()
    raw_by_hash: dict[str, dict[str, Any]] = {}
    for t in trials:
        objectives = np.array(
            [t["objective_f1"], t["objective_faithfulness"], t["objective_latency_s"]]
        )
        archive.update(objectives, tag=t["config_hash"])
        raw_by_hash[t["config_hash"]] = json.loads(t["config_json"])

    results = []
    for tag in archive.tags:
        cfg = encode(raw_by_hash[tag], search_space)
        test_result = evaluator.evaluate(
            cfg, query_set_id=query_set_id, split_query_ids=test_query_ids
        )
        results.append(
            {
                "config_hash": tag,
                "raw": raw_by_hash[tag],
                "test_objectives": test_result.objectives.tolist(),
                "test_em": test_result.em,
                "test_recall_at_k": test_result.recall_at_k,
                "test_ndcg": test_result.ndcg,
            }
        )
    return results

#!/usr/bin/env python3
"""Re-score an already-finished single-optimizer run's Pareto front against
the held-out test split, without re-running optimization (Brief §7:
test_split is "never touched during optimization" — these are the numbers
that actually belong in the paper).

Reopens the run's existing ``db.sqlite`` (via ``build_evaluator``, same as
``run_experiment.py``) and reads its already-completed trials — no new
optimizer trials are run. For a *fresh* multi-optimizer campaign, use
``scripts/run_real_ablation.py`` instead, which already does this step for
every optimizer it runs. This script is for rescoring a run that was
launched standalone via ``run_experiment.py`` (e.g. a single-optimizer
run like ``main_hotpot_qkbo_v2``).

Usage:
    python -m scripts.evaluate_run_on_test_split --config configs/experiments/main_hotpot.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qautollm.core.config import ExperimentConfig
from qautollm.evaluation.hotpotqa_dataset import load_hotpotqa_splits
from qautollm.evaluation.test_split_eval import evaluate_pareto_front_on_test_split
from qautollm.space.search_space import SearchSpace
from qautollm.utils.logging import get_logger
from scripts.run_experiment import build_evaluator

_logger = get_logger(__name__)


def evaluate_run_on_test_split(
    config_path: str,
    run_dir: str | Path | None = None,
    output: str | Path | None = None,
) -> Path:
    config = ExperimentConfig.from_yaml(config_path)
    search_space = SearchSpace.from_yaml(config.search_space)
    run_dir = Path(run_dir) if run_dir else Path("results") / config.experiment.name

    evaluator, db, query_set_id, split_query_ids = build_evaluator(config, run_dir)
    if split_query_ids is None:
        raise ValueError(
            f"dataset {config.dataset.id!r} has no test split wired up "
            "(only real datasets like 'hotpotqa' support test-split evaluation)"
        )

    n_trials = len(db.read_trials(config.experiment.name))
    if n_trials == 0:
        raise ValueError(
            f"no trials found for experiment_id={config.experiment.name!r} in "
            f"{run_dir / 'db.sqlite'} -- did the run finish, and does --config "
            "match the one it was launched with?"
        )

    splits = load_hotpotqa_splits(Path("data/hotpotqa"))
    test_query_ids = splits[config.dataset.test_split]

    report = evaluate_pareto_front_on_test_split(
        db, config.experiment.name, search_space, evaluator, query_set_id, test_query_ids
    )

    output = Path(output) if output else run_dir / "test_split_report.json"
    output.write_text(json.dumps(report, indent=2))
    _logger.info(
        "re-scored %d Pareto-optimal config(s) out of %d completed trials; wrote %s",
        len(report),
        n_trials,
        output,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None, help="defaults to results/<experiment.name>")
    parser.add_argument(
        "--output", default=None, help="defaults to <run-dir>/test_split_report.json"
    )
    args = parser.parse_args()
    evaluate_run_on_test_split(args.config, args.run_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

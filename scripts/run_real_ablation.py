#!/usr/bin/env python3
"""Real-dataset ablation: runs every optimizer variant against the *same*
prepared real dataset (Brief §12, resolved), then evaluates each
optimizer's held-out Pareto front on the test split (Brief §7: test_split
is "never touched during optimization" — these are the numbers that
actually belong in the paper, not the optimization-split numbers).

Builds one (expensive: real embeddings + real generator) Evaluator and
reuses it across every optimizer sub-run — rebuilding the corpus index or
reloading the 3B generator once per optimizer would be wasteful and
contradicts Brief D1's "load generator once, keep it resident for the
whole run" (extended here to "whole ablation campaign"). Only the
Orchestrator (and its own run_dir for logs/checkpoint) differs per
optimizer; all of them write into the *same* db, distinguished by
experiment_id.

Usage:
    python -m scripts.run_real_ablation --config configs/experiments/main_hotpot.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import qautollm.optimizers  # noqa: F401 - populates OPTIMIZERS registry
from qautollm.core.config import ExperimentConfig
from qautollm.core.orchestrator import Orchestrator
from qautollm.evaluation.hotpotqa_dataset import load_hotpotqa_splits
from qautollm.evaluation.test_split_eval import evaluate_pareto_front_on_test_split
from qautollm.space.search_space import SearchSpace
from qautollm.utils.logging import add_file_logging, get_logger
from scripts.run_experiment import build_evaluator

_logger = get_logger(__name__)

OPTIMIZER_NAMES = ("random", "tpe", "structured_bo", "qkernel_bo", "nsga2")


def run_real_ablation(
    config_path: str,
    output_dir: str | Path = "results/real_ablation",
    optimizer_names: tuple[str, ...] = OPTIMIZER_NAMES,
    seeds: tuple[int, ...] | None = None,
    resume: bool = False,
) -> Path:
    """``seeds`` defaults to the config's own single seed (unchanged
    single-run behavior). Passing more than one seed repeats the whole
    optimizer x seed grid against the *same* shared evaluator/db (L1-L3
    caches are seed-independent, so only the optimizer/GP-fit work and any
    genuinely-new configs' L4 generation cost is repeated per seed) — this
    is what a paper-strength variance estimate across seeds needs.
    """
    base_config = ExperimentConfig.from_yaml(config_path)
    search_space = SearchSpace.from_yaml(base_config.search_space)
    output_dir = Path(output_dir)
    add_file_logging(output_dir / "run.log")
    seeds = seeds or (base_config.experiment.seed,)
    multi_seed = len(seeds) > 1

    shared_run_dir = output_dir / "_shared"
    evaluator, db, query_set_id, split_query_ids = build_evaluator(base_config, shared_run_dir)
    if split_query_ids is None:
        raise ValueError(
            f"dataset {base_config.dataset.id!r} has no test split wired up "
            "(only real datasets like 'hotpotqa' support test-split evaluation)"
        )

    splits = load_hotpotqa_splits(Path("data/hotpotqa"))
    test_query_ids = splits[base_config.dataset.test_split]

    test_report: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        seed_suffix = f"_seed{seed}" if multi_seed else ""
        seed_dir = output_dir / f"seed{seed}" if multi_seed else output_dir
        for optimizer_name in optimizer_names:
            report_key = f"{optimizer_name}{seed_suffix}"
            experiment_name = f"{base_config.experiment.name}_{report_key}"
            config = base_config.model_copy(
                update={
                    "experiment": base_config.experiment.model_copy(
                        update={"name": experiment_name, "seed": seed}
                    ),
                    "optimizer": base_config.optimizer.model_copy(update={"name": optimizer_name}),
                }
            )
            run_dir = seed_dir / optimizer_name
            _logger.info("running real ablation: %s", experiment_name)

            orchestrator = Orchestrator(
                config,
                search_space,
                evaluator,
                db,
                run_dir,
                query_set_id=query_set_id,
                split_query_ids=split_query_ids,
            )
            orchestrator.run(resume=resume)

            _logger.info(
                "evaluating %s's Pareto front on the held-out test split", experiment_name
            )
            test_report[report_key] = evaluate_pareto_front_on_test_split(
                db, experiment_name, search_space, evaluator, query_set_id, test_query_ids
            )

    report_path = output_dir / "test_split_report.json"
    report_path.write_text(json.dumps(test_report, indent=2))
    _logger.info("wrote test-split Pareto-front report to %s", report_path)
    _logger.info(
        "combine trials for a main table with: "
        "python -m scripts.export_results --db %s --output-dir %s%s",
        shared_run_dir / "db.sqlite",
        output_dir,
        " --seed-stats" if multi_seed else "",
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results/real_ablation")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="one or more seeds; defaults to the config's own single seed",
    )
    parser.add_argument(
        "--optimizers",
        choices=OPTIMIZER_NAMES,
        nargs="+",
        default=None,
        help="subset of optimizers to run; defaults to all 5",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume each optimizer sub-run from its own checkpoint.json instead of "
            "restarting at trial 0 (use after an interrupted campaign, e.g. a crash "
            "or reboot mid-run); a no-op for any optimizer that has no checkpoint yet"
        ),
    )
    args = parser.parse_args()
    seeds = tuple(args.seeds) if args.seeds else None
    optimizer_names = tuple(args.optimizers) if args.optimizers else OPTIMIZER_NAMES
    run_real_ablation(
        args.config,
        args.output_dir,
        optimizer_names=optimizer_names,
        seeds=seeds,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

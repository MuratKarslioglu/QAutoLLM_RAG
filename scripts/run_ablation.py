#!/usr/bin/env python3
"""Ablation entry point (Brief §9 P8): runs the toy dataset through every
optimizer variant (random, tpe, structured_bo, qkernel_bo, nsga2) and
produces the main comparison table + hypervolume convergence plot. This is
"the one command" for the P8 gate ("docker build + one command reproduces
the toy-set main table").

Usage:
    python -m scripts.run_ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import qautollm.optimizers  # noqa: F401 - populates OPTIMIZERS registry
from qautollm.core.config import (
    DatasetConfig,
    ExperimentConfig,
    ExperimentMeta,
    GeneratorConfig,
    OptimizerConfig,
)
from qautollm.core.orchestrator import Orchestrator
from qautollm.space.search_space import SearchSpace
from qautollm.utils.logging import get_logger
from scripts.export_results import build_main_table, export_table, make_convergence_plot
from scripts.run_experiment import build_evaluator

_logger = get_logger(__name__)

RAG_V2_PATH = "configs/search_spaces/rag_v2.yaml"
OPTIMIZER_NAMES = ("random", "tpe", "structured_bo", "qkernel_bo", "nsga2")


def run_ablation(
    budget: int = 25,
    init_points: int = 10,
    seed: int = 42,
    output_dir: str | Path = "results/ablation",
    optimizer_names: tuple[str, ...] = OPTIMIZER_NAMES,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    search_space = SearchSpace.from_yaml(RAG_V2_PATH)

    all_trials = []
    for optimizer_name in optimizer_names:
        experiment_name = f"ablation_{optimizer_name}"
        config = ExperimentConfig(
            experiment=ExperimentMeta(
                name=experiment_name, seed=seed, budget=budget, checkpoints=[budget]
            ),
            dataset=DatasetConfig(id="toy", optimization_split="toy_10", test_split="toy_10"),
            optimizer=OptimizerConfig(name=optimizer_name, init_points=init_points),
            generator=GeneratorConfig(model="fake-gen"),
            search_space=RAG_V2_PATH,
        )
        run_dir = output_dir / experiment_name
        _logger.info("running ablation: %s (budget=%d)", experiment_name, budget)

        evaluator, db, query_set_id, split_query_ids = build_evaluator(config, run_dir)
        orchestrator = Orchestrator(
            config,
            search_space,
            evaluator,
            db,
            run_dir,
            query_set_id=query_set_id,
            split_query_ids=split_query_ids,
        )
        orchestrator.run(resume=False)
        all_trials.extend(db.read_trials(experiment_name))

    trials = pd.DataFrame(all_trials)
    table = build_main_table(trials)

    export_table(table, output_dir / "main_table.md", fmt="markdown")
    export_table(table, output_dir / "main_table.csv", fmt="csv")
    make_convergence_plot(trials, output_dir / "convergence.png")
    _logger.info("wrote main table + convergence plot to %s", output_dir)
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--init-points", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/ablation")
    args = parser.parse_args()

    table = run_ablation(args.budget, args.init_points, args.seed, args.output_dir)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

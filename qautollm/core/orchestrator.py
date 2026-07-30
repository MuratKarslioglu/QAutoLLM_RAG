"""The experiment loop: budget accounting, duplicate detection, checkpoint/
resume (Brief §6, §9 P3).

Takes an already-built :class:`~qautollm.evaluation.evaluator.Evaluator`
(real models or fakes) — model construction / ModelManager lifecycle is the
caller's job (``scripts/run_experiment.py``), kept out of this module so it
stays testable without a GPU or model downloads.

Two counters, deliberately not the same thing:
  - ``_suggestion_count``: every ``optimizer.suggest()`` call, ever. Used as
    the DB's ``trial_index`` (must be unique per row, duplicates included).
  - ``_budget_used``: the actual budget counter driving the ``while`` loop
    and ``--resume``. A duplicate suggestion (same ``config_hash`` seen
    before) still gets evaluated (a 100% cache hit) and still updates the
    optimizer, but does *not* increment this (Brief §5.2: "zero budget cost").
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qautollm.core.config import ExperimentConfig
from qautollm.core.registry import OPTIMIZERS
from qautollm.evaluation.evaluator import Evaluator
from qautollm.optimizers.base import OptimizerInterface
from qautollm.space.encoding import CanonicalConfig, encode
from qautollm.space.search_space import SearchSpace
from qautollm.store.db import ArtifactDB
from qautollm.utils.logging import JsonlWriter, get_logger
from qautollm.utils.profiling import progress
from qautollm.utils.seeds import make_rng, seed_everything

_logger = get_logger(__name__)
_CHECKPOINT_EVERY = 5


class Orchestrator:
    def __init__(
        self,
        config: ExperimentConfig,
        search_space: SearchSpace,
        evaluator: Evaluator,
        db: ArtifactDB,
        run_dir: str | Path,
        query_set_id: str | None = None,
        split_query_ids: list[str] | None = None,
    ) -> None:
        self.config = config
        self.search_space = search_space
        self.evaluator = evaluator
        self.db = db
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_id = config.experiment.name
        self._log = JsonlWriter(self.run_dir / "trials.jsonl")
        self._checkpoint_path = self.run_dir / "checkpoint.json"

        # query_set_id is the *embedding cache* key (must match whatever
        # scripts/build_index.py used); defaults to optimization_split since
        # the toy dataset's one query set makes the two coincide.
        # split_query_ids optionally narrows evaluate() to a specific subset
        # of the dataset's queries (Brief: real datasets like HotpotQA embed
        # every query once under one query_set_id, then opt/test are
        # disjoint slices of the same embedded set — Brief §12 Q2).
        self._query_set_id = (
            query_set_id if query_set_id is not None else config.dataset.optimization_split
        )
        self._split_query_ids = split_query_ids

        self._budget_used = 0
        self._suggestion_count = 0
        self._seen_hashes: set[str] = set()
        self.optimizer: OptimizerInterface | None = None

    def _build_init_design(self) -> list[CanonicalConfig]:
        raws = self.search_space.sobol_design(
            self.config.optimizer.init_points, seed=self.config.experiment.seed
        )
        return [encode(raw, self.search_space) for raw in raws]

    def _build_optimizer(self, init_design: list[CanonicalConfig]) -> OptimizerInterface:
        cls = OPTIMIZERS.get(self.config.optimizer.name)
        return cls(
            self.search_space,
            self.config.experiment.seed,
            init_design,
            optimizer_config=self.config.optimizer,
        )

    def run(self, resume: bool = False) -> None:
        seed_everything(self.config.experiment.seed)
        # deterministic given the seed, so rebuilding it on resume is safe:
        init_design = self._build_init_design()
        self.optimizer = self._build_optimizer(init_design)

        if resume and self._checkpoint_path.exists():
            self._load_checkpoint()
            _logger.info(
                "resumed experiment=%s at budget_used=%d/%d",
                self.experiment_id,
                self._budget_used,
                self.config.experiment.budget,
            )
        else:
            self.db.write_experiment(
                {
                    "experiment_id": self.experiment_id,
                    "config_yaml": self.config.model_dump_json(),
                    "status": "running",
                    "started_at": datetime.now(UTC).isoformat(),
                    "finished_at": None,
                }
            )

        budget = self.config.experiment.budget
        bar = progress(range(budget), desc=self.experiment_id, total=budget)
        bar.n = self._budget_used
        bar.refresh()

        while self._budget_used < budget:
            self._run_one_trial()
            bar.n = self._budget_used
            bar.refresh()
            if self._budget_used % _CHECKPOINT_EVERY == 0 or self._budget_used == budget:
                self._save_checkpoint()
        bar.close()

        self._save_checkpoint()
        self.db.write_experiment(
            {
                "experiment_id": self.experiment_id,
                "status": "finished",
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        _logger.info("run complete: cache report=%s", self.evaluator.cache_report())

    def _run_one_trial(self) -> None:
        cfg = self._suggest_with_fallback()
        is_duplicate = cfg.config_hash in self._seen_hashes
        self._seen_hashes.add(cfg.config_hash)

        start = time.perf_counter()
        try:
            result = self.evaluator.evaluate(
                cfg, query_set_id=self._query_set_id, split_query_ids=self._split_query_ids
            )
            status = "ok"
        except Exception:
            # Brief §11: never swallow — log it, record status=failed, keep going.
            _logger.exception("trial failed: config_hash=%s", cfg.config_hash)
            result = None
            status = "failed"
        wall_clock_s = time.perf_counter() - start

        if result is not None:
            self.optimizer.update(cfg, result.objectives)
            self._maybe_write_kernel_log()

        self._write_trial_row(cfg, result, status, is_duplicate, wall_clock_s)

        self._suggestion_count += 1
        if not is_duplicate:
            self._budget_used += 1

    def _suggest_with_fallback(self) -> CanonicalConfig:
        """``evaluate()`` below already degrades to ``status="failed"`` and
        keeps going on any exception (Brief §11: "never swallow"). Until now
        ``optimizer.suggest()`` had no equivalent: a single GP-fit/EHVI
        failure inside ``structured_bo``/``qkernel_bo`` (BoTorch numerical
        errors get more likely as the training set grows) crashed the whole
        budget loop uncaught, silently ending a 100-trial run at, say, trial
        51 with no recorded row and no way to tell duplicate-detection or
        --resume "this trial never really happened." Fall back to a random
        valid sample instead so the loop always keeps moving; logged loudly
        (and, via ``add_file_logging``, persisted) so a *recurring* fallback
        is visible rather than silently degrading BO into random search.
        """
        try:
            return self.optimizer.suggest()
        except Exception:
            _logger.exception(
                "optimizer.suggest() failed for experiment=%s at suggestion_count=%d "
                "-- falling back to a random valid config instead of crashing the run",
                self.experiment_id,
                self._suggestion_count,
            )
            fallback_rng = make_rng(self.config.experiment.seed + self._suggestion_count * 7919)
            raw = self.search_space.sample(fallback_rng)
            return encode(raw, self.search_space)

    def _maybe_write_kernel_log(self) -> None:
        """Brief §8's ``kernel_log`` table: quantum runs only, and only
        once the optimizer has actually gone through an EHVI-guided (GP-fit)
        suggestion — duck-typed via ``last_kernel_time_s`` rather than an
        isinstance check, so this stays agnostic to which KernelBO subclass
        is in use.
        """
        kernel_time_s = getattr(self.optimizer, "last_kernel_time_s", None)
        if kernel_time_s is None or self.config.optimizer.name != "qkernel_bo":
            return
        self.db.write_kernel_log(
            {
                "experiment_id": self.experiment_id,
                "trial_index": self._suggestion_count,
                "shots": self.config.optimizer.quantum.shots,
                "psd_clipped_mass": self.optimizer.last_psd_clipped_mass,
                "kernel_time_s": kernel_time_s,
            }
        )

    def _write_trial_row(
        self,
        cfg: CanonicalConfig,
        result: Any,
        status: str,
        is_duplicate: bool,
        wall_clock_s: float,
    ) -> None:
        objectives = result.objectives if result is not None else None
        row = {
            "experiment_id": self.experiment_id,
            "optimizer_name": self.config.optimizer.name,
            "seed": self.config.experiment.seed,
            "dataset_id": self.config.dataset.id,
            "trial_index": self._suggestion_count,
            "config_hash": cfg.config_hash,
            "config_json": json.dumps(cfg.raw, sort_keys=True),
            "objective_f1": float(objectives[0]) if objectives is not None else None,
            "objective_faithfulness": float(objectives[1]) if objectives is not None else None,
            "objective_latency_s": float(objectives[2]) if objectives is not None else None,
            "em": result.em if result is not None else None,
            "recall_at_k": result.recall_at_k if result is not None else None,
            "ndcg": result.ndcg if result is not None else None,
            "tokens_prompt": result.tokens_prompt if result is not None else None,
            "tokens_generated": result.tokens_generated if result is not None else None,
            "peak_vram_mb": result.peak_vram_mb if result is not None else None,
            "cache_hits_json": json.dumps(result.cache_report) if result is not None else None,
            "duplicate_flag": int(is_duplicate),
            "wall_clock_s": wall_clock_s,
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.db.write_trial(row)
        self._log.write(row)

    def _save_checkpoint(self) -> None:
        payload = {
            "budget_used": self._budget_used,
            "suggestion_count": self._suggestion_count,
            "seen_hashes": sorted(self._seen_hashes),
            "optimizer_state": self.optimizer.state_dict(),
        }
        tmp_path = self._checkpoint_path.with_name(self._checkpoint_path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True))
        tmp_path.replace(self._checkpoint_path)  # atomic: survives kill -9 mid-write

    def _load_checkpoint(self) -> None:
        payload = json.loads(self._checkpoint_path.read_text())
        self._budget_used = payload["budget_used"]
        self._suggestion_count = payload["suggestion_count"]
        self._seen_hashes = set(payload["seen_hashes"])
        self.optimizer.load_state_dict(payload["optimizer_state"])

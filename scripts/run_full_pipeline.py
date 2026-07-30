#!/usr/bin/env python3
"""One-command runner for the full paper-data pipeline, chaining every
script needed for the current Q1-paper revision pass in the order they need
to run. Each stage is just ``python -m scripts.<name> ...`` under the hood
(via subprocess, so each stage's own logging/progress bars stream straight
to the terminal) — this script only adds stage sequencing, timing, and
fail-fast behavior on top.

This is the user's manual/GPU job, not something meant to run
unattended inside CI: the main HotpotQA campaign alone can take many
hours (multiple seeds x real 3B-model generation).

Defaults match the current headline comparison (2026-07-18): 3 seeds
(42/43/44) x (structured_bo, qkernel_bo) only -- the paired classical/
quantum comparison the paper actually needs a significance test for, not
all 5 optimizers at 1 seed. Override with ``--optimizers``/``--seeds`` for
a different scope.

Stages (each individually skippable):
    1. real HotpotQA campaign (N seeds x    (scripts.run_real_ablation)
       given optimizers) + main/seed-stats  (scripts.export_results)
       tables                               -> results/main_hotpot_v3
    2. real-scale synthetic Layer-A sweep  (scripts.run_synth)
       -> results/synth
    3. real-scale qubit-count sweep, 1-5   (scripts.run_qubit_sweep)
       (Brief D3 cap)                      -> results/qubit_sweep
    4. extended qubit-count sweep,         (scripts.run_qubit_sweep_extended)
       5/10/15/20 (beyond Brief D3)        -> results/qubit_sweep_extended
    5. paper tables/figures from 2+3+4     (scripts.make_paper_figures)
       -> results/paper

Usage:
    python -m scripts.run_full_pipeline
    python -m scripts.run_full_pipeline --seeds 42 43 44 45 46
    python -m scripts.run_full_pipeline --optimizers structured_bo qkernel_bo tpe
    python -m scripts.run_full_pipeline --skip-extended-qubit-sweep
    python -m scripts.run_full_pipeline --quick-synth-and-qubit   # smoke test only
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from qautollm.utils.logging import add_file_logging, get_logger

_logger = get_logger(__name__)

_RULE = "=" * 72

# Every stage below inherits this process's stdio and streams live to
# whatever terminal launched the pipeline -- fine while someone is watching,
# but if that terminal session ends (SSH drop, closed tab, laptop suspend)
# before the ~hours-long pipeline finishes, the only record of *why* it
# stopped goes with it. `_run` tees each stage's combined stdout/stderr into
# this file (via `tee`, not a Python-side pipe, so tqdm's own \r-redraw
# progress bars still reach the real terminal) so a crash is always
# diagnosable afterward, even from a completely fresh session.
_PIPELINE_LOG = Path("results/run_full_pipeline.log")


def _stage_start(name: str) -> float:
    _logger.info(_RULE)
    _logger.info("STAGE START: %s", name)
    _logger.info(_RULE)
    return time.monotonic()


def _stage_end(name: str, t0: float) -> None:
    elapsed = time.monotonic() - t0
    _logger.info("STAGE DONE: %s (%.1fs = %.2fh)", name, elapsed, elapsed / 3600)


def _run(cmd: list[str], stage_name: str) -> None:
    t0 = _stage_start(stage_name)
    _logger.info("$ %s", " ".join(cmd))
    # pipefail so a failing stage's exit code survives the `| tee`, not tee's own.
    log_arg = shlex.quote(str(_PIPELINE_LOG))
    shell_cmd = f"set -o pipefail; {shlex.join(cmd)} 2>&1 | tee -a {log_arg}"
    result = subprocess.run(["bash", "-c", shell_cmd])
    if result.returncode != 0:
        _logger.error(
            "STAGE FAILED: %s (exit code %d) -- stopping pipeline. Full output: %s",
            stage_name,
            result.returncode,
            _PIPELINE_LOG,
        )
        raise SystemExit(result.returncode)
    _stage_end(stage_name, t0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # v3 (2026-07-18): fixed faithfulness (NLI xsmall->base + ANSWER/EVIDENCE
    # generation format) -- see configs/experiments/main_hotpot_v3.yaml's own
    # header for why v2's results aren't reusable as-is. Fresh generator_id
    # means this cannot silently reuse v2's L4 cache.
    parser.add_argument("--config", default="configs/experiments/main_hotpot_v3.yaml")
    parser.add_argument("--output-root", default="results/main_hotpot_v3")
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["structured_bo", "qkernel_bo"],
        help="optimizers for the main HotpotQA campaign (default: just the paired "
        "classical/quantum comparison; pass e.g. --optimizers random tpe nsga2 "
        "structured_bo qkernel_bo for the full 5-way ablation)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="seeds for the main campaign (default: 3, the minimum for the paired "
        "Wilcoxon significance test in scripts.export_results). Pass a single seed "
        "for a quick single-seed run.",
    )
    parser.add_argument(
        "--quick-synth-and-qubit",
        action="store_true",
        help="use QUICK_SCALE for the synthetic sweeps instead of full scale "
        "(smoke test, minutes not hours -- not paper-strength data)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the main campaign's optimizer sub-runs from their own "
        "checkpoint.json instead of restarting each at trial 0 (use after an "
        "interrupted run, e.g. a crash or reboot mid-run)",
    )
    parser.add_argument("--skip-main-campaign", action="store_true")
    parser.add_argument("--skip-synth", action="store_true")
    parser.add_argument("--skip-qubit-sweep", action="store_true")
    parser.add_argument(
        "--skip-extended-qubit-sweep",
        action="store_true",
        help="skip scripts.run_qubit_sweep_extended (5/10/15/20 qubits, beyond Brief "
        "D3's 5-qubit cap -- exploratory, ~3h with lightning.qubit). Included by "
        "default since it's now part of the paper's qubit-count analysis.",
    )
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    _PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    add_file_logging(_PIPELINE_LOG)
    _logger.info("full pipeline output being tee'd to %s", _PIPELINE_LOG)

    py = sys.executable
    output_root = Path(args.output_root)
    synth_dir = Path("results/synth")
    qubit_dir = Path("results/qubit_sweep")
    extended_qubit_dir = Path("results/qubit_sweep_extended")

    pipeline_t0 = time.monotonic()

    # -- 1: main real-dataset campaign (N seeds) + main/seed-stats tables ---
    if not args.skip_main_campaign:
        cmd = [
            py,
            "-m",
            "scripts.run_real_ablation",
            "--config",
            args.config,
            "--output-dir",
            str(output_root),
            "--optimizers",
            *args.optimizers,
            "--seeds",
            *[str(s) for s in args.seeds],
        ]
        if args.resume:
            cmd.append("--resume")
        _run(cmd, f"real ablation campaign (seeds={args.seeds}) -> {output_root}")

        export_cmd = [
            py,
            "-m",
            "scripts.export_results",
            "--db",
            str(output_root / "_shared" / "db.sqlite"),
            "--output-dir",
            str(output_root),
        ]
        if len(args.seeds) > 1:
            export_cmd.append("--seed-stats")
        _run(export_cmd, "export main table (+ seed-stats/significance if >1 seed)")
    else:
        _logger.info("skipping main campaign (--skip-main-campaign)")

    # -- 2: real-scale synthetic Layer-A sweep -------------------------------
    if not args.skip_synth:
        cmd = [py, "-m", "scripts.run_synth", "--output-dir", str(synth_dir)]
        if args.quick_synth_and_qubit:
            cmd.append("--quick")
        _run(cmd, f"synthetic Layer-A sweep -> {synth_dir}")
    else:
        _logger.info("skipping synthetic Layer-A sweep (--skip-synth)")

    # -- 3: real-scale qubit-count sweep, 1-5 (Brief D3 cap) -----------------
    if not args.skip_qubit_sweep:
        cmd = [py, "-m", "scripts.run_qubit_sweep", "--output-dir", str(qubit_dir)]
        if args.quick_synth_and_qubit:
            cmd.append("--quick")
        _run(cmd, f"qubit-count sweep -> {qubit_dir}")
    else:
        _logger.info("skipping qubit-count sweep (--skip-qubit-sweep)")

    # -- 4: beyond-Brief-D3 qubit range (5/10/15/20) -------------------------
    if not args.skip_extended_qubit_sweep:
        cmd = [
            py,
            "-m",
            "scripts.run_qubit_sweep_extended",
            "--output-dir",
            str(extended_qubit_dir),
        ]
        if args.quick_synth_and_qubit:
            cmd.append("--quick")
        _run(cmd, f"extended qubit-count sweep (5/10/15/20) -> {extended_qubit_dir}")
    else:
        _logger.info("skipping extended qubit sweep (--skip-extended-qubit-sweep)")

    # -- 5: paper tables/figures from stages 2+3+4 ---------------------------
    if not args.skip_figures:
        _run(
            [
                py,
                "-m",
                "scripts.make_paper_figures",
                "--qubit-sweep-csv",
                str(qubit_dir / "qubit_sweep_results.csv"),
                "--qubit-sweep-extended-csv",
                str(extended_qubit_dir / "qubit_sweep_extended_results.csv"),
                "--synth-csv",
                str(synth_dir / "sweep_results.csv"),
                "--output-dir",
                "results/paper",
            ],
            "paper tables + figures",
        )
    else:
        _logger.info("skipping paper figures (--skip-figures)")

    total = time.monotonic() - pipeline_t0
    _logger.info(_RULE)
    _logger.info("FULL PIPELINE DONE in %.1fs (%.2fh)", total, total / 3600)
    _logger.info(_RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

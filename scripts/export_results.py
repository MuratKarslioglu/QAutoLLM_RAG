#!/usr/bin/env python3
"""Tables/plots for the paper (Brief §4, §9 P8): reads the ``trials`` table
from one or more experiment DBs and produces a per-optimizer main
comparison table (hypervolume + mean metrics) plus a hypervolume
convergence plot.

Usage:
    python -m scripts.export_results --db results/ablation/ablation_random/db.sqlite [--db ...] \\
        --output-dir results/ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from qautollm.store.db import ArtifactDB
from qautollm.surrogate.pareto_archive import ParetoArchive
from qautollm.utils.logging import get_logger

_logger = get_logger(__name__)

_OBJECTIVE_COLUMNS = ["objective_f1", "objective_faithfulness", "objective_latency_s"]


def load_trials(db_paths: list[str | Path]) -> pd.DataFrame:
    """Read and concatenate the ``trials`` table from one or more DBs
    (Brief §9 P8's ablation runs each write their own per-experiment DB,
    Brief §7's per-experiment convention from run_experiment.py).
    """
    frames = []
    for path in db_paths:
        db = ArtifactDB(path)
        rows = db.read_trials()
        if rows:
            frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _shared_reference_point(trials: pd.DataFrame, epsilon: float = 1e-3) -> np.ndarray:
    ok = trials[trials["status"] == "ok"]
    if ok.empty:
        return np.array([1.0, 1.0, 1.0])
    return ok[_OBJECTIVE_COLUMNS].to_numpy().max(axis=0) + epsilon


def build_main_table(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per ``optimizer_name``: hypervolume (against one reference
    point shared across every optimizer, for a fair comparison) plus mean
    quality/system metrics over its own successful trials.
    """
    if trials.empty:
        return pd.DataFrame(
            columns=[
                "optimizer",
                "n_trials",
                "n_ok",
                "mean_f1",
                "mean_faithfulness",
                "mean_latency_s",
                "mean_em",
                "mean_recall_at_k",
                "mean_ndcg",
                "hypervolume",
            ]
        )

    ref_point = _shared_reference_point(trials)
    rows = []
    for optimizer_name, group in trials.groupby("optimizer_name"):
        ok = group[group["status"] == "ok"]
        archive = ParetoArchive()
        for obj in ok[_OBJECTIVE_COLUMNS].to_numpy():
            archive.update(obj)
        rows.append(
            {
                "optimizer": optimizer_name,
                "n_trials": len(group),
                "n_ok": len(ok),
                "mean_f1": -ok["objective_f1"].mean() if len(ok) else float("nan"),
                "mean_faithfulness": (
                    -ok["objective_faithfulness"].mean() if len(ok) else float("nan")
                ),
                "mean_latency_s": ok["objective_latency_s"].mean() if len(ok) else float("nan"),
                "mean_em": ok["em"].mean() if len(ok) else float("nan"),
                "mean_recall_at_k": ok["recall_at_k"].mean() if len(ok) else float("nan"),
                "mean_ndcg": ok["ndcg"].mean() if len(ok) else float("nan"),
                "hypervolume": archive.hypervolume(ref_point) if len(ok) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("optimizer").reset_index(drop=True)


def build_seed_aggregated_table(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per optimizer, aggregated *across seeds*: each seed's own
    hypervolume/mean metrics are computed first (all seeds share the same
    reference point, so they stay comparable), then mean+std is taken
    across those per-seed values. A single seed's numbers can't tell
    quantum and classical kernels apart from noise — this is what a
    paper-strength variance estimate needs instead.
    """
    columns = [
        "optimizer",
        "n_seeds",
        "hypervolume_mean",
        "hypervolume_std",
        "mean_f1_mean",
        "mean_f1_std",
        "mean_faithfulness_mean",
        "mean_faithfulness_std",
        "mean_latency_s_mean",
        "mean_latency_s_std",
    ]
    if trials.empty:
        return pd.DataFrame(columns=columns)

    ref_point = _shared_reference_point(trials)
    per_seed_rows = []
    for (optimizer_name, seed), group in trials.groupby(["optimizer_name", "seed"]):
        ok = group[group["status"] == "ok"]
        archive = ParetoArchive()
        for obj in ok[_OBJECTIVE_COLUMNS].to_numpy():
            archive.update(obj)
        per_seed_rows.append(
            {
                "optimizer": optimizer_name,
                "seed": seed,
                "hypervolume": archive.hypervolume(ref_point) if len(ok) else 0.0,
                "mean_f1": -ok["objective_f1"].mean() if len(ok) else float("nan"),
                "mean_faithfulness": (
                    -ok["objective_faithfulness"].mean() if len(ok) else float("nan")
                ),
                "mean_latency_s": ok["objective_latency_s"].mean() if len(ok) else float("nan"),
            }
        )

    agg = (
        pd.DataFrame(per_seed_rows)
        .groupby("optimizer")
        .agg(
            n_seeds=("seed", "nunique"),
            hypervolume_mean=("hypervolume", "mean"),
            hypervolume_std=("hypervolume", "std"),
            mean_f1_mean=("mean_f1", "mean"),
            mean_f1_std=("mean_f1", "std"),
            mean_faithfulness_mean=("mean_faithfulness", "mean"),
            mean_faithfulness_std=("mean_faithfulness", "std"),
            mean_latency_s_mean=("mean_latency_s", "mean"),
            mean_latency_s_std=("mean_latency_s", "std"),
        )
        .reset_index()
        .sort_values("optimizer")
        .reset_index(drop=True)
    )
    std_cols = [c for c in agg.columns if c.endswith("_std")]
    agg[std_cols] = agg[std_cols].fillna(0.0)  # a single seed has no std, not NaN
    return agg


_SUPPORTED_METRICS = ("hypervolume", "mean_f1", "mean_faithfulness", "mean_latency_s")


def _per_seed_metric(
    trials: pd.DataFrame, optimizer_name: str, metric: str, ref_point: np.ndarray
) -> dict[int, float]:
    """Same per-seed aggregation :func:`build_seed_aggregated_table` uses,
    for one optimizer/metric -- shared by :func:`compare_optimizers_paired`
    so the significance test is computed on *exactly* the numbers the
    seed-stats table reports, not a re-derivation that could silently
    drift from it.
    """
    sub = trials[trials["optimizer_name"] == optimizer_name]
    out: dict[int, float] = {}
    for seed, group in sub.groupby("seed"):
        ok = group[group["status"] == "ok"]
        if not len(ok):
            out[seed] = 0.0 if metric == "hypervolume" else float("nan")
            continue
        if metric == "hypervolume":
            archive = ParetoArchive()
            for obj in ok[_OBJECTIVE_COLUMNS].to_numpy():
                archive.update(obj)
            out[seed] = archive.hypervolume(ref_point)
        elif metric == "mean_f1":
            out[seed] = -ok["objective_f1"].mean()
        elif metric == "mean_faithfulness":
            out[seed] = -ok["objective_faithfulness"].mean()
        elif metric == "mean_latency_s":
            out[seed] = ok["objective_latency_s"].mean()
        else:
            raise ValueError(f"unsupported metric {metric!r}; expected one of {_SUPPORTED_METRICS}")
    return out


def compare_optimizers_paired(
    trials: pd.DataFrame, optimizer_a: str, optimizer_b: str, metric: str = "hypervolume"
) -> dict[str, float | int | str]:
    """Paired Wilcoxon signed-rank test on a per-seed metric between two
    optimizers (same shared reference point, same seeds -- Brief §5.2's
    paired-comparison design already guarantees seeds line up 1:1). This is
    the statistic the paper's headline classical-vs-quantum claim needs: a
    single seed's point estimate (or even a raw mean+std across a handful
    of seeds) can't say whether an observed gap is signal or noise, and eye-
    balling two close numbers is not a substitute for a significance test.

    ``metric`` defaults to ``"hypervolume"`` (the paper's primary claim);
    ``"mean_f1"``/``"mean_faithfulness"``/``"mean_latency_s"`` let the same
    test be run per-metric -- needed once hypervolume and a component
    metric disagree in direction (2026-08-09 real campaign: qkernel_bo
    ahead on mean_f1, behind on both mean_faithfulness and hypervolume) and
    that disagreement itself needs its own significance number rather than
    just eyeballing the main table.

    Returns ``n_seeds`` plus (once ``n_seeds >= 3``, scipy's own floor for a
    non-degenerate two-sided Wilcoxon test) ``statistic``/``p_value`` and
    the mean paired difference (``a`` minus ``b``); with fewer seeds the
    test is left out (``statistic``/``p_value`` are ``None``) rather than
    reporting a meaningless number.
    """
    if trials.empty:
        return {
            "optimizer_a": optimizer_a,
            "optimizer_b": optimizer_b,
            "metric": metric,
            "n_seeds": 0,
        }

    ref_point = _shared_reference_point(trials)
    vals_a = _per_seed_metric(trials, optimizer_a, metric, ref_point)
    vals_b = _per_seed_metric(trials, optimizer_b, metric, ref_point)
    shared_seeds = sorted(set(vals_a) & set(vals_b))

    result: dict[str, float | int | str] = {
        "optimizer_a": optimizer_a,
        "optimizer_b": optimizer_b,
        "metric": metric,
        "n_seeds": len(shared_seeds),
    }
    if len(shared_seeds) < 3:
        result["statistic"] = None
        result["p_value"] = None
        result["mean_diff_a_minus_b"] = float("nan")
        return result

    a_vals = np.array([vals_a[s] for s in shared_seeds])
    b_vals = np.array([vals_b[s] for s in shared_seeds])
    diffs = a_vals - b_vals
    if np.all(diffs == 0):
        # scipy.stats.wilcoxon raises on an all-zero-difference sample
        # rather than returning a (statistic, p_value) pair.
        result["statistic"] = 0.0
        result["p_value"] = 1.0
    else:
        stat, p_value = wilcoxon(a_vals, b_vals)
        result["statistic"] = float(stat)
        result["p_value"] = float(p_value)
    result["mean_diff_a_minus_b"] = float(diffs.mean())
    return result


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavored markdown table — avoids pulling in the
    `tabulate` package just for `DataFrame.to_markdown`.
    """
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def export_table(df: pd.DataFrame, output_path: str | Path, fmt: str = "markdown") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "markdown":
        output_path.write_text(_to_markdown_table(df))
    elif fmt == "csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"unknown format {fmt!r}; expected 'markdown' or 'csv'")


def make_convergence_plot(trials: pd.DataFrame, output_path: str | Path) -> None:
    """Hypervolume-so-far vs. trial index, one line per optimizer, all
    against the same shared reference point as :func:`build_main_table`.
    """
    output_path = Path(output_path)
    if trials.empty:
        return
    ref_point = _shared_reference_point(trials)

    fig, ax = plt.subplots(figsize=(6, 4))
    for optimizer_name, group in trials.groupby("optimizer_name"):
        ok = group[group["status"] == "ok"].sort_values("trial_index")
        archive = ParetoArchive()
        hv_curve = []
        for obj in ok[_OBJECTIVE_COLUMNS].to_numpy():
            archive.update(obj)
            hv_curve.append(archive.hypervolume(ref_point))
        ax.plot(ok["trial_index"], hv_curve, marker=".", label=optimizer_name)

    ax.set_xlabel("trial index")
    ax.set_ylabel("hypervolume so far")
    ax.set_title("Hypervolume convergence")
    ax.legend(fontsize="small")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_seed_variance_plot(trials: pd.DataFrame, output_path: str | Path) -> None:
    """Per-seed hypervolume, one marker per seed per optimizer plus a
    mean +/- std bar -- the seed-to-seed spread a bare mean+std table can't
    show at a glance, and what a reader needs to sanity-check a paired
    significance claim (:func:`compare_optimizers_paired`) visually rather
    than just trusting a p-value.
    """
    output_path = Path(output_path)
    if trials.empty:
        return
    ref_point = _shared_reference_point(trials)

    per_seed_hv: dict[str, dict[int, float]] = {}
    for optimizer_name, group in trials.groupby("optimizer_name"):
        per_seed_hv[optimizer_name] = {}
        for seed, seed_group in group.groupby("seed"):
            ok = seed_group[seed_group["status"] == "ok"]
            archive = ParetoArchive()
            for obj in ok[_OBJECTIVE_COLUMNS].to_numpy():
                archive.update(obj)
            per_seed_hv[optimizer_name][seed] = archive.hypervolume(ref_point) if len(ok) else 0.0

    optimizers = sorted(per_seed_hv)
    if not optimizers:
        return

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, optimizer_name in enumerate(optimizers):
        values = np.array(list(per_seed_hv[optimizer_name].values()))
        jitter = (rng.random(len(values)) - 0.5) * 0.15
        color = f"C{i}"
        ax.scatter(np.full(len(values), i) + jitter, values, alpha=0.6, color=color, zorder=2)
        ax.errorbar(
            [i],
            [values.mean()],
            yerr=[values.std()],
            fmt="_",
            color=color,
            markersize=20,
            capsize=6,
            elinewidth=2,
            zorder=3,
        )

    n_seeds = len(next(iter(per_seed_hv.values())))
    ax.set_xticks(range(len(optimizers)))
    ax.set_xticklabels(optimizers)
    ax.set_ylabel("hypervolume (per seed)")
    ax.set_title(f"Per-seed hypervolume (n={n_seeds} seeds)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", action="append", required=True, help="repeatable")
    parser.add_argument("--output-dir", default="results/export")
    parser.add_argument(
        "--seed-stats",
        action="store_true",
        help="also write a per-optimizer mean+std-across-seeds table (needs a 'seed' "
        "column with more than one distinct value per optimizer to be meaningful)",
    )
    args = parser.parse_args()

    trials = load_trials(args.db)
    output_dir = Path(args.output_dir)

    table = build_main_table(trials)
    export_table(table, output_dir / "main_table.md", fmt="markdown")
    export_table(table, output_dir / "main_table.csv", fmt="csv")
    make_convergence_plot(trials, output_dir / "convergence.png")
    _logger.info("wrote main table + convergence plot to %s", output_dir)
    print(table.to_string(index=False))

    if args.seed_stats:
        seed_table = build_seed_aggregated_table(trials)
        export_table(seed_table, output_dir / "main_table_seed_stats.md", fmt="markdown")
        export_table(seed_table, output_dir / "main_table_seed_stats.csv", fmt="csv")
        make_seed_variance_plot(trials, output_dir / "hypervolume_by_seed.png")
        _logger.info("wrote seed-aggregated table + per-seed variance plot to %s", output_dir)
        print(seed_table.to_string(index=False))

        optimizers_present = set(trials["optimizer_name"].unique())
        if {"structured_bo", "qkernel_bo"} <= optimizers_present:
            comparisons = {
                metric: compare_optimizers_paired(
                    trials, "qkernel_bo", "structured_bo", metric=metric
                )
                for metric in _SUPPORTED_METRICS
            }
            comparison_path = output_dir / "qkernel_vs_structured_significance.json"
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_path.write_text(json.dumps(comparisons, indent=2))
            for metric, comparison in comparisons.items():
                _logger.info(
                    "qkernel_bo vs structured_bo on %s (paired Wilcoxon, n_seeds=%d): "
                    "p_value=%s, mean_diff=%s",
                    metric,
                    comparison["n_seeds"],
                    comparison.get("p_value"),
                    comparison.get("mean_diff_a_minus_b"),
                )
            _logger.info("wrote %s", comparison_path)
            print(json.dumps(comparisons, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

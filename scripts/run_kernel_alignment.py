#!/usr/bin/env python3
"""Kernel geometric-difference diagnostic (2026-08-11).

After the real HotpotQA campaign (n=10 seeds) came back with the quantum
kernel *significantly behind* the classical one on hypervolume (paired
Wilcoxon p=0.0195), the natural next question is "why", and whether any
still-unexplored quantum kernel design could plausibly close that gap on
*this* data -- as opposed to guessing at more sweeps.

Huang et al. 2021 ("Power of data in quantum machine learning",
arXiv:2011.01938) give a way to answer this without training anything new:
the geometric difference

    g(K1 || K2) = sqrt(|| sqrt(K2) (K1 + reg*N*I)^-1 sqrt(K2) ||_inf)

bounds how much a K2-based model could *ever* outperform a K1-based one on
this fixed, unlabeled set of points, for *any* target function. g in
[0, sqrt(N)]. This is a necessary-but-not-sufficient condition: if
g(K_classical || K_quantum) is small (O(1), not growing with N), that is a
proof that no relabeling of this exact data could make the quantum kernel
win -- not just "it didn't win this time", but "it structurally cannot,
regardless of which objective is being optimized". If g is instead close
to its sqrt(N) ceiling, an advantage isn't ruled out (still says nothing
about whether the *actual* HotpotQA objectives are the kind of hard-to-
learn-classically function that would realize it).

Uses real trial configs already sitting in the completed main_hotpot_v3
campaign's database -- no new LLM inference, no re-running the real task.
Both kernels are fit via the exact same qautollm.surrogate.gp.fit_independent_gps
production code path used during BO (Brief's "differ only in which
GroupKernel factory" fairness rule applies here too), so the Gram matrices
compared are the *actual* fitted kernels, not untrained defaults.

Usage:
    python -m scripts.run_kernel_alignment
    python -m scripts.run_kernel_alignment --db results/main_hotpot_v3/_shared/db.sqlite \\
        --config configs/experiments/main_hotpot_v3.yaml --n-points 100
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import scipy.linalg
import torch

from qautollm.core.config import ExperimentConfig
from qautollm.kernels.classical import make_classical_group_kernel
from qautollm.kernels.quantum import quantum_group_kernel_factory
from qautollm.kernels.structural import StructuralKernel
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace
from qautollm.surrogate.gp import augmented_features, fit_independent_gps
from qautollm.utils.logging import get_logger

_logger = get_logger(__name__)

OBJECTIVE_NAMES = ("f1", "faithfulness", "latency_s")


def load_real_trials(db_path: Path, experiment_prefix: str, n_points: int, seed: int) -> tuple:
    """Distinct-by-config_hash trial configs + objectives from the real
    campaign's own database, subsampled to ``n_points`` for a tractable
    (O(N^3) eigendecomposition/sqrtm) geometric-difference computation.
    Pools both optimizers' trials -- both explored the same search space,
    and more distinct points make for a more representative diagnostic.
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        "select config_hash, config_json, objective_f1, objective_faithfulness, "
        "objective_latency_s from trials where experiment_id like ? and status = 'ok'",
        (f"{experiment_prefix}%",),
    )
    rows = cur.fetchall()
    con.close()
    if not rows:
        raise ValueError(
            f"no trials found for experiment prefix {experiment_prefix!r} in {db_path}"
        )

    seen: dict[str, tuple] = {}
    for config_hash, config_json, f1, faithfulness, latency in rows:
        seen.setdefault(config_hash, (json.loads(config_json), (f1, faithfulness, latency)))

    distinct = list(seen.values())
    rng = np.random.default_rng(seed)
    if len(distinct) > n_points:
        idx = rng.choice(len(distinct), size=n_points, replace=False)
        distinct = [distinct[i] for i in idx]

    raw_configs = [d[0] for d in distinct]
    objectives = np.array([d[1] for d in distinct], dtype=np.float64)
    return raw_configs, objectives


def _to_correlation_matrix(k: np.ndarray) -> np.ndarray:
    """Rescale a PSD Gram matrix to unit diagonal (K_ii = 1).

    ``ScaleKernel``'s fitted ``outputscale`` is an arbitrary positive
    multiplier the L-BFGS-B fit lands on for reasons that have nothing to
    do with kernel *shape* (e.g. we already saw a near-degenerate quantum
    kernel fit to outputscale~1e-4 purely because the raw Gram matrix was
    close to constant). Feeding that raw scale into g() makes the result
    mostly measure which kernel's optimizer run landed on a bigger
    outputscale, not which kernel geometry differs. Normalizing to unit
    diagonal (the correlation matrix) removes that nuisance factor -- and
    is the natural normalization for the quantum kernel specifically, whose
    un-scaled fidelity Gram matrix already has K_ii = 1 by construction.
    """
    d = np.sqrt(np.clip(np.diag(k), 1e-12, None))
    return k / np.outer(d, d)


def effective_rank(k: np.ndarray, tol: float = 1e-8) -> int:
    """Count eigenvalues above ``tol`` relative to the largest one -- a
    plain, easy-to-report companion to g(): Brief D3 caps every group at
    <=5 qubits, so the quantum feature map's Hilbert space (<=2**5 = 32 per
    group) bounds how many independent directions the quantum Gram matrix
    can have, structurally, regardless of N. The classical kernel (Matern
    x categorical-overlap) carries no such cap.
    """
    eigvals = np.linalg.eigvalsh(k)
    return int(np.sum(eigvals > tol * eigvals.max()))


def geometric_difference(k_base: np.ndarray, k_other: np.ndarray, reg: float = 1e-6) -> float:
    """g(K_base || K_other) -- see module docstring. Both inputs are
    correlation-normalized (unit diagonal) before the eigendecomposition so
    the result reflects kernel *shape*, not fitted outputscale.
    """
    k_base = _to_correlation_matrix(k_base)
    k_other = _to_correlation_matrix(k_other)
    n = k_base.shape[0]
    base_reg_inv = np.linalg.inv(k_base + reg * n * np.eye(n))
    sqrt_other = scipy.linalg.sqrtm(k_other).real
    m = sqrt_other @ base_reg_inv @ sqrt_other
    m = (m + m.T) / 2
    eigmax = float(np.linalg.eigvalsh(m).max())
    return float(np.sqrt(max(eigmax, 0.0)))


def run_diagnostic(
    config_path: Path, db_path: Path, experiment_prefix: str, n_points: int, seed: int
) -> dict:
    base_config = ExperimentConfig.from_yaml(config_path)
    space = SearchSpace.from_yaml(base_config.search_space)
    depth = base_config.optimizer.quantum.feature_map_depth  # 1 in production -- match it exactly

    raw_configs, objectives = load_real_trials(db_path, experiment_prefix, n_points, seed)
    configs = [encode(raw, space) for raw in raw_configs]
    train_x = augmented_features(configs)
    # sign convention matches KernelBO._suggest_via_ehvi's train_y_max
    train_y_max = -torch.tensor(objectives, dtype=torch.float64)
    n = train_x.shape[0]

    _logger.info("fitting classical and quantum GPs on %d real trial configs", n)

    def _classical_kernel() -> StructuralKernel:
        return StructuralKernel.from_search_space(space, make_classical_group_kernel)

    classical_model = fit_independent_gps(train_x, train_y_max, _classical_kernel)
    quantum_factory = quantum_group_kernel_factory(depth=depth, shots=None, seed=seed)

    def _quantum_kernel() -> StructuralKernel:
        return StructuralKernel.from_search_space(space, quantum_factory)

    quantum_model = fit_independent_gps(train_x, train_y_max, _quantum_kernel)

    sqrt_n = float(np.sqrt(n))
    per_objective = []
    for i, name in enumerate(OBJECTIVE_NAMES):
        with torch.no_grad():
            k_classical = (
                classical_model.models[i].covar_module(train_x, train_x).evaluate().numpy()
            )
            k_quantum = quantum_model.models[i].covar_module(train_x, train_x).evaluate().numpy()
        g_c_to_q = geometric_difference(k_classical, k_quantum)
        g_q_to_c = geometric_difference(k_quantum, k_classical)
        per_objective.append(
            {
                "objective": name,
                "g_classical_to_quantum": g_c_to_q,
                "g_classical_to_quantum_over_sqrt_n": g_c_to_q / sqrt_n,
                "g_quantum_to_classical": g_q_to_c,
                "g_quantum_to_classical_over_sqrt_n": g_q_to_c / sqrt_n,
                "classical_effective_rank": effective_rank(k_classical),
                "quantum_effective_rank": effective_rank(k_quantum),
                "n_points": n,
            }
        )

    return {
        "n_points": n,
        "sqrt_n": sqrt_n,
        "feature_map_depth": depth,
        "per_objective": per_objective,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/main_hotpot_v3.yaml")
    parser.add_argument("--db", default="results/main_hotpot_v3/_shared/db.sqlite")
    parser.add_argument("--experiment-prefix", default="main_hotpot_v3_")
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/kernel_alignment")
    args = parser.parse_args()

    result = run_diagnostic(
        Path(args.config), Path(args.db), args.experiment_prefix, args.n_points, args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "geometric_difference.json"
    json_path.write_text(json.dumps(result, indent=2))

    lines = [
        "| objective | g(classical→quantum) | / sqrt(N) | g(quantum→classical) | / sqrt(N) | "
        "classical rank | quantum rank |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["per_objective"]:
        n_points = row["n_points"]
        lines.append(
            f"| {row['objective']} | {row['g_classical_to_quantum']:.3f} | "
            f"{row['g_classical_to_quantum_over_sqrt_n']:.3f} | "
            f"{row['g_quantum_to_classical']:.3f} | "
            f"{row['g_quantum_to_classical_over_sqrt_n']:.3f} | "
            f"{row['classical_effective_rank']}/{n_points} | "
            f"{row['quantum_effective_rank']}/{n_points} |"
        )
    table = "\n".join(lines)
    (output_dir / "geometric_difference_table.md").write_text(
        f"N={result['n_points']} points, sqrt(N)={result['sqrt_n']:.2f}, "
        f"feature_map_depth={result['feature_map_depth']}\n\n{table}\n"
    )
    _logger.info("wrote %s and geometric_difference_table.md", json_path)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

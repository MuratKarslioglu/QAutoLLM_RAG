#!/usr/bin/env python3
"""Two statistics the manuscript's §5.2 relies on but `export_results.py`
does not compute on its own:

1. Holm-Bonferroni correction across the three secondary-metric paired
   Wilcoxon tests (F1, faithfulness, latency) in
   `results/main_hotpot_v3/qkernel_vs_structured_significance.json` —
   correcting for testing three metrics against the same two optimizers
   instead of treating each as an independent standalone finding.
2. Paired significance on the held-out `test_300` split
   (`results/main_hotpot_v3/test_split_report.json`), i.e. whether any
   secondary-metric difference observed on the `opt_100` optimization
   split replicates on data neither optimizer ever selected configs
   against.

Usage:
    python -m scripts.compute_publication_stats
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = Path("results/main_hotpot_v3")


def holm_correction() -> pd.DataFrame:
    sig = json.loads((RESULTS_DIR / "qkernel_vs_structured_significance.json").read_text())
    secondary = ["mean_f1", "mean_faithfulness", "mean_latency_s"]
    raw_p = pd.DataFrame(
        {"metric": secondary, "raw_p": [sig[m]["p_value"] for m in secondary]}
    ).sort_values("raw_p").reset_index(drop=True)

    m = len(raw_p)
    alpha = 0.05
    holm_p, running_max = [], 0.0
    for i, row in raw_p.iterrows():
        adj = (m - i) * row["raw_p"]
        running_max = max(running_max, adj)
        holm_p.append(min(running_max, 1.0))
    raw_p["holm_adjusted_p"] = holm_p
    raw_p["reject_at_0_05"] = raw_p["holm_adjusted_p"] < alpha
    return raw_p


def heldout_significance() -> pd.DataFrame:
    report = json.loads((RESULTS_DIR / "test_split_report.json").read_text())

    # test_objectives = [-f1, -faithfulness, latency] (negated metrics are
    # minimized during optimization; flip sign back for reporting).
    per_seed_optimizer: dict[tuple[str, int], list[list[float]]] = defaultdict(list)
    for key, configs in report.items():
        optimizer, seed_str = key.rsplit("_seed", 1)
        for cfg in configs:
            per_seed_optimizer[(optimizer, int(seed_str))].append(cfg["test_objectives"])

    rows = []
    for (optimizer, seed), obj_lists in per_seed_optimizer.items():
        arr = np.array(obj_lists)
        rows.append(
            {
                "optimizer": optimizer,
                "seed": seed,
                "heldout_f1_mean": -arr[:, 0].mean(),
                "heldout_faithfulness_mean": -arr[:, 1].mean(),
                "heldout_latency_mean": arr[:, 2].mean(),
            }
        )
    per_seed = pd.DataFrame(rows).sort_values(["optimizer", "seed"])

    results = []
    for metric in ("heldout_f1_mean", "heldout_faithfulness_mean", "heldout_latency_mean"):
        wide = per_seed.pivot(index="seed", columns="optimizer", values=metric).sort_index()
        q, c = wide["qkernel_bo"].to_numpy(), wide["structured_bo"].to_numpy()
        stat, p = wilcoxon(q, c)
        results.append(
            {
                "metric": metric,
                "quantum_mean": float(q.mean()),
                "classical_mean": float(c.mean()),
                "mean_diff_q_minus_c": float((q - c).mean()),
                "wilcoxon_p_value": float(p),
                "wins_quantum": int((q > c).sum()),
                "wins_classical": int((q < c).sum()),
            }
        )
    return pd.DataFrame(results)


if __name__ == "__main__":
    holm = holm_correction()
    holm.to_csv(RESULTS_DIR / "secondary_metrics_holm.csv", index=False)
    print("Holm-corrected secondary metrics:")
    print(holm.to_string(index=False))

    heldout = heldout_significance()
    heldout.to_csv(RESULTS_DIR / "heldout_significance.csv", index=False)
    print("\nHeld-out (test_300) paired significance:")
    print(heldout.to_string(index=False))

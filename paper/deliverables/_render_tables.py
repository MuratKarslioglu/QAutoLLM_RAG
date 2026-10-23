#!/usr/bin/env python3
"""Renders every table this paper cites as a clean, standalone PNG image
(paper-style: white background, no color-coded cells, just a bordered
grid) -- kept alongside the source .md/.csv/.json so the numbers survive
even if someone only has the images. Re-run after any source table
changes: python3 paper/deliverables/_render_tables.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(parents=True, exist_ok=True)

FONT = "DejaVu Sans"


def render_table(df: pd.DataFrame, title: str, out_path: Path, col_widths=None, fontsize=10) -> None:
    n_rows, n_cols = df.shape
    # width per column estimated from the longest string it holds (header or
    # cell), in character-widths -- matplotlib's auto_set_column_width alone
    # still clips long headers like "mean_recall_at_k", so size the figure
    # to the content instead of the content to a fixed figure.
    col_chars = [
        max(len(str(df.columns[j])), df.iloc[:, j].astype(str).str.len().max() if n_rows else 0)
        for j in range(n_cols)
    ]
    fig_h = 0.55 + 0.36 * (n_rows + 1)
    fig_w = max(6.0, 0.11 * sum(c + 2 for c in col_chars))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize + 2, fontweight="bold", pad=14, loc="left")

    table = ax.table(
        cellText=df.to_numpy(),
        colLabels=df.columns,
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.5)
    table.auto_set_column_width(col=list(range(n_cols)))

    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2b3a55")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f6fa")
        else:
            cell.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fmt(df: pd.DataFrame, decimals: dict) -> pd.DataFrame:
    df = df.copy()
    for col, nd in decimals.items():
        if col in df.columns:
            df[col] = df[col].map(lambda v: f"{v:.{nd}f}" if pd.notna(v) else "")
    return df


def main() -> None:
    # 1. Real campaign main table (trial-level, n=1089/1051)
    df = pd.read_csv(ROOT / "results/main_hotpot_v3/main_table.csv")
    df = fmt(
        df,
        {
            "mean_f1": 4,
            "mean_faithfulness": 4,
            "mean_latency_s": 4,
            "mean_em": 4,
            "mean_recall_at_k": 4,
            "mean_ndcg": 4,
            "hypervolume": 4,
        },
    )
    render_table(
        df,
        "Table 1 -- Real HotpotQA campaign, trial-level (n_trials=1089/1051)",
        OUT / "01_main_table.png",
    )

    # 2. Seed-aggregated table (n=10 seeds, mean +/- std)
    raw = pd.read_csv(ROOT / "results/main_hotpot_v3/main_table_seed_stats.csv")
    rows = []
    for _, r in raw.iterrows():
        rows.append(
            {
                "optimizer": r["optimizer"],
                "n_seeds": int(r["n_seeds"]),
                "hypervolume": f"{r['hypervolume_mean']:.4f} +/- {r['hypervolume_std']:.4f}",
                "mean_f1": f"{r['mean_f1_mean']:.4f} +/- {r['mean_f1_std']:.4f}",
                "mean_faithfulness": f"{r['mean_faithfulness_mean']:.4f} +/- {r['mean_faithfulness_std']:.4f}",
                "mean_latency_s": f"{r['mean_latency_s_mean']:.4f} +/- {r['mean_latency_s_std']:.4f}",
            }
        )
    render_table(
        pd.DataFrame(rows),
        "Table 2 -- Real HotpotQA campaign, seed-aggregated (mean +/- std, n=10 seeds)",
        OUT / "02_main_table_seed_stats.png",
        fontsize=9,
    )

    # 3. Per-metric significance (paired Wilcoxon, n=10 seeds)
    sig = json.loads((ROOT / "results/main_hotpot_v3/qkernel_vs_structured_significance.json").read_text())
    rows = []
    for metric, r in sig.items():
        rows.append(
            {
                "metric": metric,
                "mean diff (qkernel - structured)": f"{r['mean_diff_a_minus_b']:.5f}",
                "p-value": f"{r['p_value']:.4f}" if r["p_value"] is not None else "n/a",
                "n_seeds": r["n_seeds"],
                "significant (p<0.05)": "yes" if (r["p_value"] or 1) < 0.05 else "no",
            }
        )
    render_table(
        pd.DataFrame(rows),
        "Table 3 -- Per-metric significance, qkernel_bo vs structured_bo (paired Wilcoxon, n=10 seeds)",
        OUT / "03_per_metric_significance.png",
        fontsize=9,
    )

    # 3b. Secondary-metric Holm correction + held-out (test_300) replication
    # (regenerate via: python -m scripts.compute_publication_stats)
    holm = pd.read_csv(ROOT / "results/main_hotpot_v3/secondary_metrics_holm.csv")
    holm = fmt(holm, {"raw_p": 4, "holm_adjusted_p": 4})
    render_table(
        holm,
        "Table 3b -- Secondary metrics, Holm-corrected across the 3-metric family (n=10 seeds)",
        OUT / "03b_secondary_metrics_holm.png",
        fontsize=9,
    )

    heldout = pd.read_csv(ROOT / "results/main_hotpot_v3/heldout_significance.csv")
    heldout = fmt(
        heldout,
        {"quantum_mean": 4, "classical_mean": 4, "mean_diff_q_minus_c": 4, "wilcoxon_p_value": 4},
    )
    render_table(
        heldout,
        "Table 3c -- Held-out test_300 paired significance (never optimized against)",
        OUT / "03c_heldout_significance.png",
        fontsize=9,
    )

    # 4. Qubit-count advantage table (synthetic, 1-10 qubits/group)
    df = pd.read_csv(ROOT / "results/paper/qubit_advantage_table.csv")
    df = fmt(df, {"interaction": 1, "structured_bo_hv": 4, "qkernel_bo_hv": 4, "quantum_advantage": 4})
    df["best_qubit_count"] = df["best_qubit_count"].astype(int)
    render_table(
        df,
        "Table 4 -- Qubit-count sweep: best-qubit-count classical vs quantum hypervolume",
        OUT / "04_qubit_advantage_table.png",
    )

    # 5. Layer-A synthetic sweep summary (18 conditions x 5 optimizers = 90 rows)
    df = pd.read_csv(ROOT / "results/paper/layer_a_summary_table.csv")
    df = fmt(df, {"mean": 4, "std": 4})
    render_table(
        df,
        "Table 5 -- Synthetic Layer-A sweep: hypervolume by n_groups / conditional_fraction / "
        "interaction / optimizer",
        OUT / "05_layer_a_summary_table.png",
        fontsize=8,
    )

    # 6. Kernel geometric-difference diagnostic
    gd = json.loads((ROOT / "results/kernel_alignment/geometric_difference.json").read_text())
    rows = []
    for r in gd["per_objective"]:
        rows.append(
            {
                "objective": r["objective"],
                "g(classical->quantum)": f"{r['g_classical_to_quantum']:.3f}",
                "g(C->Q) / sqrt(N)": f"{r['g_classical_to_quantum_over_sqrt_n']:.3f}",
                "g(quantum->classical)": f"{r['g_quantum_to_classical']:.3f}",
                "g(Q->C) / sqrt(N)": f"{r['g_quantum_to_classical_over_sqrt_n']:.3f}",
                "classical rank": f"{r['classical_effective_rank']}/{r['n_points']}",
                "quantum rank": f"{r['quantum_effective_rank']}/{r['n_points']}",
            }
        )
    render_table(
        pd.DataFrame(rows),
        f"Table 6 -- Kernel geometric-difference diagnostic (N={gd['n_points']} real trial configs, "
        f"depth={gd['feature_map_depth']})",
        OUT / "06_geometric_difference_table.png",
        fontsize=9,
    )

    # 7. Depth x qubit grid summary (post-fix, mean hypervolume by qubits/depth)
    df = pd.read_csv(ROOT / "results/depth_qubit_grid/depth_qubit_grid_results.csv")
    df = df[~df["failed"]]
    piv = (
        df.pivot_table(
            index=["qubits_per_group", "depth"], columns="optimizer", values="hypervolume", aggfunc="mean"
        )
        .reset_index()
    )
    piv["diff (qkernel - structured)"] = piv["qkernel_bo"] - piv["structured_bo"]
    piv = fmt(piv, {"qkernel_bo": 4, "structured_bo": 4, "diff (qkernel - structured)": 4})
    piv = piv.rename(columns={"qubits_per_group": "qubits/group", "qkernel_bo": "qkernel_bo HV", "structured_bo": "structured_bo HV"})
    render_table(
        piv,
        "Table 7 -- Depth x qubit grid (post-fix, mean over n=3 seeds x 2 interaction levels)",
        OUT / "07_depth_qubit_grid_table.png",
        fontsize=9,
    )


if __name__ == "__main__":
    main()

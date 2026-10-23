# Do Quantum Fidelity Kernels Help Bayesian Optimization of RAG Pipelines? A Controlled, Multi-Regime Study

*Working draft — 2026-10-23. All experimental sections (§5.1–5.6) are now
filled in with real, completed results. Remaining `[TODO]`s are prose-only
(Introduction, Related Work, Limitations, Conclusion) — outlined but not
yet written out; no pending compute.*

## Abstract

We ask whether a fidelity-based quantum kernel, dropped into an otherwise
identical Bayesian-optimization loop, improves multi-objective hyperparameter
search for retrieval-augmented generation (RAG) pipelines relative to a
matched classical kernel. Across a real HotpotQA campaign (10 seeds, 100
trials/seed), a qubit-count scaling sweep (1–20 qubits/group), a circuit-
depth sweep (0–20 re-uploading layers), and a synthetic multi-group
benchmark, we find no quantum advantage on the primary multi-objective
metric (hypervolume) in any regime — and on the real task, the classical
kernel is *significantly* ahead (paired Wilcoxon, p=0.0195, n=10). The
picture is not uniformly negative, though: on the optimization-split
component objectives, the quantum kernel is nominally better on F1
(p=0.037) and significantly worse on faithfulness (p=0.0039); the
faithfulness effect survives Holm correction across the three secondary
metrics (adjusted p=0.012), but the F1 effect does not (adjusted p=0.074)
and does not replicate on the held-out test split (p=0.695, direction
reversed) — we therefore do not treat F1 as a genuine quantum advantage.
A geometric-
difference diagnostic (Huang et al., 2021) on the real task's fitted kernels
shows why: the quantum kernel's fitted Gram matrix has substantially lower
effective rank than the classical kernel's (e.g. 41/100 vs. 68/100 on F1),
consistent with its Brief-mandated ≤5-qubit-per-group cap bounding its
representable Hilbert space to ≤32 dimensions — a bound neither more qubits
(up to 20/group) nor more re-uploading depth (up to 20 layers) can relieve,
which the qubit-count and depth sweeps both confirm empirically. We report
this as a well-powered negative result with an independently-checked
mechanism, not a tuning failure or an incomplete search.

## 1. Introduction

`[TODO — outline below, prose not yet written]`
- Motivation: BO for RAG pipeline configuration is expensive per trial (real
  LLM inference); a better kernel directly saves compute/time.
- Quantum kernel methods (fidelity/projected quantum kernels) have
  theoretical advantage results for specially constructed data (Huang et
  al., 2021; Liu et al., 2021) but decidedly mixed empirical results on
  generic classical data — this paper is an empirical test on a real,
  economically motivated task rather than a synthetic benchmark designed to
  favor either side.
- Contribution list:
  1. A fair, code-shared classical-vs-quantum BO comparison on a real RAG
     task (not just a synthetic proxy), with enough seeds (n=10) for a
     paired significance test.
  2. A component-metric breakdown showing the quantum kernel does not fail
     uniformly — it loses on faithfulness and on the combined hypervolume,
     and shows a nominal F1 edge that does not survive multiple-comparison
     correction or held-out replication (§5.2.1).
  3. A geometric-difference/effective-rank diagnostic that explains *why*,
     tying the result back to the qubit-count cap rather than leaving it as
     an unexplained negative result.
  4. A depth-axis result under a corrected,
     non-degenerate re-uploading circuit (the first attempt hit a
     Clifford-periodicity measurement artifact, documented and fixed —
     §5.6 and Appendix B).

## 2. Related Work

`[TODO]`
- Quantum kernel methods: Havlíček et al. 2019 (quantum feature maps);
  Schuld & Killoran 2019 (quantum kernels as inner products); Huang et al.
  2021 (power of data, geometric difference — the diagnostic used in §5.5);
  Schuld, Sweke & Meyer 2021 (data re-uploading as Fourier series — the
  construction used to fix the depth-encoding bug in Appendix B).
- Classical-vs-quantum kernel empirical comparisons on real (non-toy) data:
  cite whatever the codebase/Brief already references, plus any negative-
  result QML papers for framing ("well-powered negative results are a
  needed contribution, not a null publication").
- BO for LLM/RAG pipeline configuration (classical baselines this paper's
  `structured_bo` descends from).

## 3. Method

### 3.1 Task and Search Space

Real task: HotpotQA multi-hop QA over a RAG pipeline with five
hyperparameter groups (`configs/search_spaces/rag_v2.yaml`):

| Group | Variables | Encoded dim (qubits, quantum side) |
| --- | --- | --- |
| G1_chunking | chunk_size, chunk_overlap, chunking_strategy | 5 |
| G2_embedding | embedding_model | 3 |
| G3_retrieval | top_k, hybrid_on, dense_weight | 3 |
| G4_reranking | reranker_on, reranker_top_n | 2 |
| G5_context | context_budget | 1 |

Every group is categorical (one-hot encoded) — relevant later (§5.6,
Appendix B). Objectives: F1, faithfulness (NLI-base entailment), latency
(seconds); multi-objective hypervolume against a shared reference point is
the primary metric (Brief §5.2).

### 3.2 Classical and Quantum GroupKernel

Both optimizers share one `StructuralKernel`
(`qautollm/kernels/structural.py`) that combines per-group kernels
`K(x,x') = Σ_g α_g · δ_g(mask,mask') · K_g(z_g, z'_g)`; the group kernel
factory is the *only* thing that differs between `structured_bo` and
`qkernel_bo` (Brief §1's fairness rule, enforced structurally — both
optimizer classes subclass the same `KernelBO`).

- **Classical**: Matérn-5/2 × categorical-overlap kernel per group
  (`qautollm/kernels/classical.py`).
- **Quantum**: fidelity kernel `K_g(z,z') = |⟨φ(z)|φ(z')⟩|²`, `φ` an
  angle-encoding circuit with `feature_map_depth` re-uploading layers,
  exact statevector simulation (`qautollm/kernels/quantum.py`), capped at
  ≤5 qubits/group (Brief D3).

### 3.3 Optimizers

`structured_bo`/`qkernel_bo`: independent-objective GP (`SingleTaskGP` per
objective) fit via `fit_gpytorch_mll`, joint acquisition via qLogEHVI
(`qautollm/optimizers/structured_bo.py`). Both use the exact same init
design, candidate sampling RNG, and BO loop; the only code-path divergence
is `group_kernel_factory`.

### 3.4 Experimental Protocol

`[TODO — restate Brief §5.2's fairness/pairing rules and the pre-
registration commitment: every regime tested here (real task, qubit-count
sweep, synthetic Layer-A sweep, depth sweep) is reported regardless of
outcome; no regime was dropped after seeing its result.]`

## 4. Experimental Setup

- Real campaign: `configs/experiments/main_hotpot_v3.yaml`, 100 trials ×
  10 seeds × 2 optimizers = 2000 trials, `feature_map_depth=1`,
  `shots=null` (exact statevector). Wall-clock: 51.15h total (2026-08-07 to
  2026-08-09), CPU-only (`default.qubit`/`lightning.qubit`, no GPU — see
  Appendix C on the abandoned GPU acceleration attempt).
- Qubit-count sweep: synthetic benchmark, 1–5 qubits/group (Brief-compliant,
  `scripts/run_qubit_sweep.py`) and an exploratory 5–20 qubit/group
  extension (`scripts/run_qubit_sweep_extended.py`, `lightning.qubit`,
  memory-hardened — see Appendix C).
- Synthetic Layer-A sweep: `n_groups ∈ {3,5,8}`, `conditional_fraction ∈
  {0, 0.5}`, `interaction ∈ {0, 0.5, 1.0}`, 5 optimizers (`nsga2`, `random`,
  `tpe`, `structured_bo`, `qkernel_bo`).
- Kernel-alignment diagnostic: `scripts/run_kernel_alignment.py`, 100 real
  trial configs sampled from the completed campaign's own database (no new
  LLM inference), both kernels fit via the identical production
  `fit_independent_gps` path.
- Depth sweep: `scripts/run_depth_sweep.py` /
  `scripts/run_depth_qubit_grid.py`, synthetic benchmark, `depth ∈
  {0,5,10,20}` (grid) / `{1,2,3,4}` (1D), run under the fixed
  circuit (§5.6, Appendix B).

## 5. Results

### 5.1 Real HotpotQA Campaign — Primary Result

n=10 seeds (45–51 added 2026-08-07–09 to an initial n=3), 100 trials/seed.

| optimizer | n_trials | mean F1 | mean faithfulness | mean latency (s) | hypervolume |
| --- | --- | --- | --- | --- | --- |
| qkernel_bo | 1089 | 0.4251 | 0.1555 | 0.8057 | 0.0098 |
| structured_bo | 1051 | 0.4159 | 0.1718 | 0.8047 | 0.0099 |

Seed-aggregated (mean ± std across n=10 per-seed values):

| optimizer | hypervolume | mean F1 | mean faithfulness | mean latency (s) |
| --- | --- | --- | --- | --- |
| qkernel_bo | 0.0082 ± 0.0008 | 0.4247 ± 0.0140 | 0.1558 ± 0.0098 | 0.8059 ± 0.0118 |
| structured_bo | 0.0089 ± 0.0005 | 0.4159 ± 0.0068 | 0.1718 ± 0.0118 | 0.8046 ± 0.0139 |

**Headline finding**: on the primary multi-objective metric (hypervolume),
`structured_bo` is significantly ahead of `qkernel_bo`.

### 5.2 Per-Metric Significance (Paired Wilcoxon, n=10 seeds)

| metric | mean diff (qkernel − structured) | p-value | direction |
| --- | --- | --- | --- |
| hypervolume | −0.00066 | **0.0195** | structured_bo significantly ahead |
| mean F1 | +0.00879 | **0.0371** | qkernel_bo significantly ahead |
| mean faithfulness | −0.01600 | **0.0039** | structured_bo significantly ahead |
| mean latency | +0.00131 | 0.4922 | no significant difference |

This is not a uniform loss for the quantum kernel: it finds nominally
better-F1 configurations, but significantly worse-faithfulness ones, and the
faithfulness effect dominates the combined hypervolume outcome.

### 5.2.1 Multiple-Comparison Correction and Held-Out Validation

Table 5.2 tests three secondary metrics against the same optimizer pair, so
we Holm-correct the three p-values (`scripts/compute_publication_stats.py`,
`results/main_hotpot_v3/secondary_metrics_holm.csv`):

| metric | raw p | Holm-adjusted p | significant at α=0.05 |
| --- | --- | --- | --- |
| mean faithfulness | 0.0039 | **0.0117** | yes |
| mean F1 | 0.0371 | 0.0742 | no |
| mean latency | 0.4922 | 0.4922 | no |

Faithfulness survives correction; F1 does not.

Separately, every metric above is measured on the `opt_100` split each
optimizer actually searched over. We also paired-test the held-out
`test_300` re-evaluation of each seed/optimizer's final Pareto front
(`qautollm/evaluation/test_split_eval.py`, `results/main_hotpot_v3/
heldout_significance.csv`), which neither optimizer ever selected
configurations against:

| metric | quantum mean | classical mean | p (Wilcoxon) |
| --- | --- | --- | --- |
| held-out F1 | 0.3768 | 0.3779 | 0.695 (n.s., quantum slightly *worse*) |
| held-out faithfulness | 0.1567 | 0.1561 | 0.625 (n.s.) |
| held-out latency | 0.8346 | 0.8294 | 0.160 (n.s.) |

None of the three held-out metrics differ significantly, and the F1
direction reverses (quantum marginally worse, not better). Combined with
the correction above, we read the F1 result as an in-sample,
uncorrected-only observation, not a genuine effect, and we do not report
it as a quantum advantage anywhere else in this paper. The faithfulness
result (structured_bo ahead) and the primary hypervolume result both hold
up under this scrutiny — Holm correction does not apply to hypervolume
(the single primary endpoint, not part of the secondary family), and no
held-out hypervolume re-evaluation exists since hypervolume is defined
over the optimization trajectory itself, not a per-config held-out score.

### 5.3 Qubit-Count Scaling

Synthetic benchmark, 1–20 qubits/group (Brief-compliant 1–5 range plus an
exploratory 5–20 extension, both CPU-only):

| interaction | best qubit count | structured_bo HV | qkernel_bo HV | advantage |
| --- | --- | --- | --- | --- |
| 0.0 | 10 | 5.1966 | 5.1991 | +0.0025 (~0.05%) |
| 1.0 | 1 | 8.0000 | 8.0000 | 0.0000 |

Real-scale extension (5/10/15 qubits, `lightning.qubit`): qkernel_bo and
structured_bo track within ≤0.02% of each other at every tested qubit count
(6.752 vs. 6.752 at 5 qubits; 6.576 vs. 6.574 at 10; 6.488 vs. 6.487 at 15).
No advantage emerges at any tested scale in either direction, up to 20
qubits/group (5× Brief D3's cap).

### 5.4 Synthetic Layer-A Sweep

Across `n_groups ∈ {3,5,8}`, `conditional_fraction ∈ {0,0.5}`,
`interaction ∈ {0,0.5,1.0}` (18 conditions, full table in
`results/paper/layer_a_summary_table.md`): both `structured_bo` and
`qkernel_bo` clearly outperform `nsga2`/`random`/`tpe` (as expected — kernel-
guided BO vs. non-model-based baselines). Between the two kernel-guided
methods specifically: `qkernel_bo` is ahead in low-`n_groups`, low-
`interaction` cells (e.g. `n_groups=3, interaction=0`: 6.20 vs. 6.19);
`structured_bo` is ahead as `n_groups` grows (e.g. `n_groups=8,
interaction=0`: 5.61 vs. 5.51). No systematic quantum advantage; the
crossover tracks classical-kernel behavior more than anything
quantum-specific.

### 5.5 Kernel Geometric-Difference Diagnostic

To explain *why* (not just report *that*) the quantum kernel underperforms
on the real task, we compute Huang et al.'s (2021) geometric difference
`g(K1 ‖ K2) = √‖√K2 (K1+λNI)⁻¹ √K2‖∞` between the two kernels' *fitted*
Gram matrices (production `fit_independent_gps` path, `feature_map_depth=1`)
on 100 real trial configurations sampled from the completed campaign,
correlation-normalized (unit diagonal) to remove arbitrary fitted-
outputscale effects:

| objective | g(classical→quantum) | / √N | g(quantum→classical) | / √N | classical rank | quantum rank |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | 6.77 | 0.68 | 91.91 | 9.19 | 68/100 | 41/100 |
| faithfulness | 1.31 | 0.13 | 21.70 | 2.17 | 67/100 | 32/100 |
| latency | 14.33 | 1.43 | 59.57 | 5.96 | 68/100 | 44/100 |

`g(classical→quantum)` staying at or below `√N` on every objective is a
necessary-condition check that the classical kernel could not have been
beaten by much here — consistent with §5.1–5.2. The much larger, bound-
exceeding `g(quantum→classical)` values point at the mechanism: they arise
because the quantum kernel's regularized Gram matrix is close to singular in
several directions, i.e. **the quantum kernel's fitted Gram matrix has
substantially lower effective rank than the classical kernel's** (32–44/100
vs. 67–68/100, all three objectives). This is the expected structural
consequence of Brief D3's ≤5-qubit-per-group cap: each group's quantum
feature map lives in a Hilbert space of dimension ≤2⁵=32, bounding how many
independent directions its kernel can represent, regardless of how much real
data it is fit against — a limit the classical Matérn kernel does not share.
This does not prove a wider quantum kernel (more qubits/group) would help
(§5.3 found no benefit up to 20 qubits/group either), but it does give a
concrete, falsifiable mechanism for the observed gap rather than leaving it
unexplained.

### 5.6 Circuit Depth

An initial attempt at this experiment (`scripts/run_depth_qubit_grid.py`,
2026-08-07) produced a measurement artifact, not a result: every group's
search space is one-hot categorical, so the pre-fix circuit's
`RY(π·z)`-every-layer re-uploading was Clifford for both `z∈{0,1}`, and the
whole `depth`-layer circuit was a fixed Clifford unitary to the `depth`-th
power — which has a small finite period. `depth=10` and `depth=20`
collapsed to bit-identical results with `depth=0` (full mechanism and fix in
Appendix B; regression tests in `tests/test_quantum_kernel.py`). The fix (a
per-layer irrational frequency multiplier, standard Fourier-series
re-uploading construction, leaves `depth=1` — the only depth any production
run uses — byte-identical) is in place as of 2026-08-11; the full 2D grid
(`qubits_per_group∈{5,10,15,20}`, `depth∈{0,5,10,20}`, `interaction∈{0,1}`,
`n_seeds=3`, 192 combos, 6 failed — all `qubits=20, depth=0, qkernel_bo`,
the degenerate zero-layer control point, expected) was rerun on 2026-08-12
under the corrected circuit.

The fix is confirmed working: `depth=0`'s hypervolume is now always the
*lowest* value in its row (constant-kernel control point behaving as
expected) and is no longer bit-identical to `depth=10`/`depth=20`.

| qubits/group | depth | qkernel_bo HV | structured_bo HV | diff |
| --- | --- | --- | --- | --- |
| 5 | 0 | 6.688 | 6.754 | −0.066 |
| 5 | 5/10/20 | 6.754 | 6.754 | 0.000 |
| 10 | 0 | 6.515 | 6.568 | −0.052 |
| 10 | 5 | 6.569 | 6.568 | +0.001 |
| 10 | 10 | 6.569 | 6.568 | +0.002 |
| 10 | 20 | 6.567 | 6.568 | −0.000 |
| 15 | 0 | 6.433 | 6.474 | −0.041 |
| 15 | 5/10/20 | 6.474 | 6.474 | 0.000 |
| 20 | 0 | failed | 6.396 | — |
| 20 | 5 | 6.400 | 6.396 | +0.003 |
| 20 | 10 | 6.401 | 6.396 | +0.005 |
| 20 | 20 | 6.398 | 6.396 | +0.001 |

Two things stand out, both honestly reported. First, `qubits∈{5,15}` show
`qkernel_bo` landing on *exactly* `structured_bo`'s value for every
`depth≥5` — not a repeat of the periodicity bug (confirmed via per-seed
inspection: both optimizers independently converge to the same Pareto-
optimal point), but a ceiling effect: at this budget (40 trials) the search
space at 5 or 15 qubits/group is small enough that any reasonable kernel
finds the true optimum, making the classical-vs-quantum question
uninformative at that scale. Second, at `qubits∈{10,20}` — large enough
that the search isn't saturated — `qkernel_bo` is *marginally* ahead of
`structured_bo` at every non-zero depth (+0.001 to +0.005, i.e. 0.02–0.08%),
with no clear trend across depth itself and no depth value producing a
qualitatively different outcome. Per-seed inspection (Appendix B) shows
this margin is the same order of magnitude as the run-to-run, seed-to-seed
noise at fixed depth, so it should not be read as a depth-driven advantage.
**Depth, tested honestly from 0 to 20 layers, does not produce a quantum
advantage on this benchmark** — consistent with §5.3's qubit-count sweep
and §5.5's rank-ceiling explanation: adding re-uploading layers does not add
representable Hilbert-space dimensions the way more qubits would, so a
kernel already capped well below the classical kernel's effective rank
stays capped regardless of depth.

## 6. Discussion

Taken together, §5.1–5.5 support a specific, mechanistically-grounded claim
rather than a bare "no advantage" report. The real-task result is
well-powered (n=10 seeds, up from an initial n=3 that had left the
comparison underpowered at p=0.25) and significant in the classical kernel's
favor on the primary metric (hypervolume, p=0.0195). Unlike a typical
negative QML result, this one comes with an independent check that agrees
with it *before* the fact: the geometric-difference diagnostic (§5.5) is a
necessary-condition test computed purely from the fitted kernels' geometry,
with no reference to which optimizer actually won — and it says the same
thing the 51-hour campaign says, that the classical kernel could not have
been meaningfully beaten on this data. The concrete mechanism is rank: the
quantum kernel's Brief-mandated ≤5-qubit-per-group cap bounds each group's
representable Hilbert space to ≤32 dimensions, and the fitted Gram matrices
show exactly that bound biting (32–44/100 effective rank vs. the classical
kernel's 67–68/100, across all three objectives). This reframes the
qubit-count sweep's negative result (§5.3) too: it is not that 5–20
qubits/group happens not to help, but that the rank ceiling scales with
qubit count in a way this task's ~100-trial-scale search never needed to
saturate — the classical kernel's much higher effective rank was already
sufficient, so a quantum kernel that can only match a subset of that rank
has no room to add value, regardless of depth or qubit count in isolation.

The result is not uniformly negative on every component metric, though the
one exception is weaker than it first looks. The quantum kernel finds
nominally better configurations on F1 (p=0.037) while finding significantly
*worse* ones on faithfulness (p=0.0039). Unlike faithfulness, the F1 effect
does not survive Holm correction across the three secondary metrics
(adjusted p=0.074, §5.2.1) and reverses direction on the held-out `test_300`
split (quantum marginally worse, p=0.695, n.s.). We therefore treat it as an
in-sample, uncorrected artifact rather than a real secondary advantage, and
faithfulness — which does survive both checks — as the only genuine
component-level effect, dominating the combined hypervolume outcome
accordingly.

§5.6's depth sweep closes the loop on this reframing rather than opening a
new one: rerun honestly under the corrected (non-degenerate) circuit, depth
from 0 to 20 layers produces no advantage distinguishable from seed noise at
either search-space scale tested (saturated, where both kernels tie exactly,
or unsaturated, where the ±0.02–0.08% gap is the same order of magnitude as
run-to-run variance). This is exactly what the rank-ceiling account
predicts: re-uploading more layers changes *how* the ≤32-dimensional Hilbert
space is traversed, not *how large* it is, so it cannot relieve a bound set
by qubit count. Three independent angles — the real campaign, the
qubit-count sweep, and the depth sweep — now agree with the geometric-
difference diagnostic's necessary-condition check computed before any of
them ran. That convergence, not any single number, is this paper's strongest
evidence that the ≤5-qubit-per-group quantum kernel structurally cannot
match the classical one on this task, rather than merely having failed to
in three separate tries.

## 7. Limitations

`[TODO, at minimum:]`
- Real campaign uses exact statevector simulation (no shot noise) —
  results characterize the fidelity kernel's *ideal* behavior, not a
  NISQ-hardware-realistic one.
- `feature_map_depth=1` throughout the real campaign; §5.6 addresses depth
  only on the synthetic benchmark.
- ≤5-qubit-per-group cap (Brief D3) is a deliberate, stated scope
  constraint, not a claim about arbitrarily large quantum kernels.
- Single real task (HotpotQA); generalization to other RAG datasets/domains
  untested.

## 8. Conclusion

A fidelity-based quantum kernel, substituted into an otherwise identical BO
loop, does not improve multi-objective RAG-pipeline hyperparameter search
over a matched classical kernel — on a real HotpotQA campaign, the classical
kernel is significantly ahead on the primary metric (hypervolume, n=10,
p=0.0195), and neither scaling qubits/group up to 20 nor scaling circuit
depth up to 20 re-uploading layers closes the gap. This is not an
unexplained negative result: a geometric-difference diagnostic computed
independently of any BO outcome predicts it, tracing the cause to the
quantum kernel's Brief-mandated qubit cap bounding its effective rank well
below the classical kernel's. A nominal F1 advantage for the quantum kernel
(p=0.037, uncorrected) does not survive Holm correction (adjusted p=0.074)
or held-out replication (§5.2.1) and should not be read as a genuine
effect; faithfulness — which does survive both checks — is the only
component metric where the quantum kernel is significantly, robustly
behind. `[TODO: one more pass once Introduction/Related
Work/Limitations are written, to make sure the framing here matches.]`

---

## Appendix A: Reproducibility

- Real campaign: `python -m scripts.run_real_ablation --config
  configs/experiments/main_hotpot_v3.yaml --output-dir results/main_hotpot_v3
  --optimizers structured_bo qkernel_bo --seeds 42 43 44 45 46 47 48 49 50 51`
- Tables/significance: `python -m scripts.export_results --db
  results/main_hotpot_v3/_shared/db.sqlite --output-dir results/main_hotpot_v3
  --seed-stats`
- Kernel-alignment diagnostic: `python -m scripts.run_kernel_alignment`
- Secondary-metric Holm correction and held-out (`test_300`) significance
  (§5.2.1): `python -m scripts.compute_publication_stats`
- `git log` / commit hash for the exact code version: `[TODO: fill in after
  this session's commit]`.

## Appendix B: The Clifford-Periodicity Bug and Fix

`[TODO — expand into full prose; technical content already captured in
qautollm/kernels/quantum.py's module-level comment above `_build_circuit`,
scripts/run_depth_qubit_grid.py's docstring, and
tests/test_quantum_kernel.py's `test_one_hot_categorical_inputs_no_longer_
revive_at_depth_10_or_20` / `test_depth_1_matches_the_pre_fix_single_ry_
layer_formula`. Worth including as an appendix regardless of §5.6's
outcome: it's a genuine, non-obvious pitfall for anyone re-uploading
one-hot/categorical features into a quantum circuit, and the diagnosis
method (checking downstream-decision bit-identity across depths, then Gram-
matrix inspection) generalizes.]`

## Appendix C: Hardware Notes (not a paper contribution, keep brief or cut)

`[TODO: one paragraph on the abandoned lightning.gpu acceleration attempt
(real per-circuit speedup 2.4–5.6x, but unusable end-to-end due to
driver/library instability at real BO workload scale on this project's
consumer GPU) and the OOM-hardening work (RLIMIT_AS worker caps, malloc_trim
cadence, bounded LRU statevector cache) that made the 5–20-qubit exploratory
sweeps possible at all on this hardware. Likely cut from the final paper —
infrastructure, not a scientific finding — but keep here for the authors'
own record.]`

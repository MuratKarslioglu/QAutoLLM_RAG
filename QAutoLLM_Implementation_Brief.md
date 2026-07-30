# QAutoLLM-RAG — Implementation Brief
## Instructions for the coding agent building this research framework

> **Read this entire document before writing any code.** You are implementing the software for an academic paper. Correctness, determinism, and reproducibility take priority over speed of delivery and over feature count. When this document and your own preference conflict, this document wins. When something is genuinely ambiguous, ask the user (Murat) before deciding — do not silently invent behavior.

---

## 1. What this project is

**QAutoLLM-RAG** is a research framework for *budget-aware, multi-objective optimization of Retrieval-Augmented Generation (RAG) pipeline configurations* using both classical optimizers and a quantum-kernel-based Bayesian Optimization method. It is the software artifact of an academic paper targeting a Q1/Q2 journal.

The scientific question: given a mixed, conditional configuration space (chunking, embedding model, retrieval, reranking, context budget) and a small evaluation budget (25–100 evaluations), under what conditions does a **group-decomposed quantum fidelity kernel** surrogate outperform an **identically structured classical Matérn kernel** surrogate?

Consequences for you as the implementer:

- **Fair comparison is the product.** The classical BO and quantum BO must share every line of code except the inner kernel function. Same encoding, same GP, same acquisition, same initialization, same budget accounting. Any asymmetry invalidates the paper.
- **Determinism is non-negotiable.** Same config + same seed ⇒ bit-identical results. Greedy decoding everywhere, fixed prompts, seeded everything. This also makes caching exact.
- **Every expensive computation is cached.** The experiment plan is ~6,000 pipeline evaluations on a single laptop. Without the Artifact Store working correctly, the experiments physically do not fit in the schedule.
- **Negative results are fine; broken measurements are not.** Never "fix" a result. Log everything.

## 2. Hard constraints

**Hardware target:** single laptop — NVIDIA RTX 4070 Laptop (8 GB VRAM), 48 GB RAM, Ubuntu, CUDA available.

- **D1 — One large model resident in VRAM at a time.** Pipeline stages run sequentially: embed the corpus offline → unload embedder → load generator once and keep it resident for the whole run. Reranker and NLI model are small (~0.5 GB each) and may co-reside with the generator. Enforce with an explicit `ModelManager` that owns load/unload; never let two modules independently call `.to("cuda")`.
- **D2 — Vector indexes live on disk / RAM, never VRAM.** Use `faiss-cpu`.
- **D3 — Quantum simulation runs on CPU.** Group circuits are ≤ 5 qubits; statevector simulation via PennyLane. Kernel computation must stay < 5% of evaluation wall-clock.
- **D4 — Determinism.** `temperature=0`/greedy decoding, `torch.manual_seed`, `numpy` seeding, sorted iteration over dicts/sets where order matters, no wall-clock-dependent logic in results. Provide a `utils/seeds.py::seed_everything(seed)` used at every entry point.
- **Python ≥ 3.11.** No LangChain / LlamaIndex — all RAG stages are implemented directly against `transformers`, `sentence-transformers`, `faiss`, `rank-bm25`. (Reason: full control over determinism, caching, and VRAM.)

## 3. Environment setup (create exactly this)

The repository must be installable with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Create `requirements.txt` with the following content (floor pins; after your first successful end-to-end test, freeze the actually-resolved versions into `requirements.lock` and commit both):

```text
# --- core scientific stack ---
numpy>=1.26
scipy>=1.11
pandas>=2.1
matplotlib>=3.8
tqdm>=4.66
pyyaml>=6.0
pydantic>=2.5

# --- LLM / RAG stack ---
torch>=2.2            # install CUDA build; see note below
transformers>=4.44
accelerate>=0.33
bitsandbytes>=0.43    # 4-bit quantized generator inference
sentence-transformers>=3.0
faiss-cpu>=1.8
rank-bm25>=0.2.2
datasets>=2.20

# --- optimization stack ---
optuna>=3.6           # TPE baseline
pymoo>=0.6.1          # NSGA-II baseline
gpytorch>=1.11        # GP surrogate
botorch>=0.11         # EHVI acquisition

# --- quantum stack ---
pennylane>=0.37
pennylane-lightning>=0.37

# --- dev / quality ---
pytest>=8.0
pytest-cov>=5.0
ruff>=0.5
```

**Torch note:** default PyPI torch may be CPU-only on some setups. Add a `README` line: `pip install torch --index-url https://download.pytorch.org/whl/cu121` (or the CUDA version detected on the machine) *before* `pip install -r requirements.txt`. Verify with a `scripts/check_env.py` that prints CUDA availability, VRAM, and all key package versions — write this script first.

Also create `pyproject.toml` (package name `qautollm`, `ruff` config, `pytest` config) and a `.gitignore` (`.venv/`, `results/`, `artifacts/`, `*.lock` excluded from ignore).

## 4. Repository layout (create exactly this)

```text
qautollm/
├── configs/
│   ├── experiments/            # experiment profiles (YAML) — see §7
│   ├── search_spaces/          # space + group + conditional definitions
│   └── models/                 # model name/quantization/prompt-template configs
├── qautollm/
│   ├── core/
│   │   ├── config.py           # pydantic models for all YAML schemas
│   │   ├── registry.py         # string -> class registries (optimizers, kernels, metrics)
│   │   ├── orchestrator.py     # the experiment loop; budget accounting; resume
│   │   └── model_manager.py    # D1 rule enforcement; load/unload; VRAM logging
│   ├── space/
│   │   ├── search_space.py     # variable defs, sampling, validity checking
│   │   ├── encoding.py         # CanonicalConfig: z-vector + activity mask + groups
│   │   ├── groups.py           # group definitions G1..G5
│   │   └── constraints.py      # conditional rules (reranker_on -> reranker_top_n etc.)
│   ├── rag/
│   │   ├── chunking.py         # fixed / sentence / semantic strategies
│   │   ├── embedding.py        # embedding model wrapper (offline batch indexing)
│   │   ├── indexing.py         # FAISS dense + BM25 sparse index build/load
│   │   ├── retrieval.py        # dense / hybrid retrieval with dense_weight
│   │   ├── reranking.py        # cross-encoder rerank of top-k
│   │   ├── prompting.py        # fixed prompt templates; context budget packing
│   │   └── generation.py       # 4-bit generator, greedy decoding, token accounting
│   ├── surrogate/
│   │   ├── gp.py               # multi-output = independent GPs per objective
│   │   ├── acquisition.py      # EHVI (q=1); shared by BOTH BO variants
│   │   └── pareto_archive.py   # non-dominated archive + hypervolume
│   ├── kernels/
│   │   ├── structural.py       # group-decomposed skeleton: K = Σ α_g δ_g K_g  (SHARED)
│   │   ├── classical.py        # Matérn-5/2 + categorical overlap group kernels
│   │   └── quantum.py          # angle-encoding feature maps, fidelity kernel,
│   │                           #   shot control, PSD projection (eigenvalue clipping)
│   ├── optimizers/
│   │   ├── base.py             # OptimizerInterface (see §5)
│   │   ├── random_search.py
│   │   ├── tpe.py              # Optuna wrapper
│   │   ├── nsga2.py            # pymoo wrapper
│   │   ├── structured_bo.py    # classical kernel BO
│   │   └── qkernel_bo.py       # quantum kernel BO (differs ONLY in kernel choice)
│   ├── evaluation/
│   │   ├── answer_metrics.py   # EM, token-F1 vs gold
│   │   ├── faithfulness.py     # NLI entailment rate (answer sentences vs context)
│   │   ├── retrieval_metrics.py# Recall@k, nDCG, MRR (needs gold evidence)
│   │   ├── system_metrics.py   # latency breakdown, tokens, peak VRAM
│   │   └── evaluator.py        # runs a config over a query set -> ObjectiveVector
│   ├── store/
│   │   ├── hashing.py          # canonical content hashing of param dicts
│   │   ├── artifact_store.py   # L1–L5 cache (see §5.3)
│   │   └── db.py               # SQLite schema + writers/readers (see §8)
│   ├── synth/
│   │   └── benchmark.py        # synthetic mixed-space test functions (Layer A)
│   └── utils/
│       ├── seeds.py
│       ├── logging.py          # structured JSONL run logs
│       └── profiling.py        # timing context managers, VRAM probes
├── scripts/
│   ├── check_env.py
│   ├── build_index.py          # pre-build all (chunking × embedding) indexes
│   ├── run_experiment.py       # main entry: --config path.yaml [--resume]
│   ├── run_synth.py            # Layer A synthetic benchmark
│   ├── run_ablation.py
│   └── export_results.py       # tables/plots for the paper
├── tests/                      # see §10 — encoding & kernel tests are MANDATORY
├── docker/Dockerfile
├── docs/
├── results/                    # gitignored
├── requirements.txt
├── requirements.lock
├── pyproject.toml
└── README.md
```

## 5. Core contracts (implement these interfaces exactly)

### 5.1 Canonical encoding

```python
@dataclass(frozen=True)
class CanonicalConfig:
    raw: dict[str, Any]           # original parameter dict (only ACTIVE params present)
    z: np.ndarray                 # [0,1]-normalized vector; one-hot for categoricals;
                                  #   inactive conditional dims filled with 0.5 sentinel
    mask: np.ndarray              # activity mask over groups, shape (n_groups,)
    group_slices: dict[str, slice]  # group name -> slice into z
    config_hash: str              # canonical hash of `raw` (sorted keys)
```

Rules: encoding is a pure function of `raw`; `decode(encode(x)) == x` for all valid configs (property-test this); invalid combinations (e.g. `reranker_top_n` present while `reranker_on=False`) raise at construction.

### 5.2 Optimizer interface

```python
class OptimizerInterface(ABC):
    def __init__(self, search_space: SearchSpace, seed: int, init_design: list[CanonicalConfig]): ...
    def suggest(self) -> CanonicalConfig: ...
    def update(self, config: CanonicalConfig, objectives: np.ndarray) -> None: ...
    def state_dict(self) -> dict: ...        # for checkpoint/resume
    def load_state_dict(self, state: dict) -> None: ...
```

The orchestrator generates **one shared Sobol init design of 10 points per seed** and passes the *same* design to every optimizer at that seed (paired comparison). Duplicate suggestions are detected by `config_hash`; an optimizer suggesting a duplicate gets the cached result at zero budget cost, but the orchestrator logs the duplicate.

### 5.3 Artifact Store (the load-bearing component)

Content-addressed cache with five levels; key = canonical hash of exactly the parameters that determine the output:

| Level | Output | Key fields |
|---|---|---|
| L1 | chunk list | corpus_id, chunking_strategy, chunk_size, chunk_overlap |
| L2 | dense index + BM25 index | L1 key + embedding_model |
| L3 | retrieved doc ids/scores per query | L2 key + query_set_id + top_k + hybrid params + rerank params |
| L4 | generated answer + token counts per query | L3 key + context_budget + prompt_template_id + generator_id |
| L5 | kernel entries K_g(z_g, z'_g) | kernel_id + group + (hash(z_g), hash(z'_g)) sorted pair + shots |

API: `store.get_or_compute(level, key_fields: dict, compute_fn) -> Artifact`. Storage: files under `artifacts/` (L1/L2 as files, L3–L5 in SQLite tables). Every hit/miss is counted; the run summary reports hit rates per level. **Latency accounting rule:** system-metric latency is always measured on a cold, real execution and stored with the L4 artifact; cache hits reuse the stored latency — never measure the cache lookup time as pipeline latency.

### 5.4 Structural kernel skeleton (shared)

`structural.py` implements `K(x,x') = Σ_g α_g · δ_g(mask, mask') · K_g(z_g, z'_g)` with learnable `α_g` and mismatch factor `κ ∈ [0,1)`, taking a pluggable `GroupKernel` per group. `classical.py` and `quantum.py` only provide `GroupKernel` implementations. `structured_bo.py` and `qkernel_bo.py` must differ **only** in which `GroupKernel` factory they pass — enforce this by making both a thin subclass of one `KernelBO` class parametrized by kernel family. Quantum specifics: PennyLane fidelity kernel per group, `shots ∈ {None (statevector), 1024, 256}`; finite-shot Gram matrices go through eigenvalue-clipping PSD projection and the clipped mass is logged.

### 5.5 Evaluator

`evaluator.evaluate(config, dataset, split) -> EvalResult` runs the full pipeline through the Artifact Store over the query set and returns: `objectives = np.array([−F1, −faithfulness, latency])` (minimization convention documented in one place only), plus analysis metrics (EM, Recall@k, nDCG, tokens, peak VRAM) and per-query records for later statistical analysis.

## 6. Data flow of one experiment run

```text
run_experiment.py --config configs/experiments/main_hotpot.yaml
  1. load + validate config (pydantic)           4. loop until budget exhausted:
  2. seed_everything(seed)                            a. cfg = optimizer.suggest()
  3. build shared Sobol init; evaluate it             b. result = evaluator.evaluate(cfg)   # cache-first
     (through cache) for this optimizer               c. optimizer.update(cfg, result.objectives)
                                                      d. db.write(trial row); archive.update()
                                                      e. checkpoint every N trials (resume-safe)
  5. finalize: hypervolume@{25,50,100}, Pareto front dump, cache-hit report, JSONL log close
```

`--resume` must restore: optimizer state, budget counter, RNG states, Pareto archive — and produce identical continuation to an uninterrupted run (test this).

## 7. Experiment config schema (YAML)

```yaml
experiment:
  name: main_hotpot_qkbo
  seed: 42
  budget: 100
  checkpoints: [25, 50, 100]
dataset:
  id: hotpotqa_sub
  optimization_split: opt_100      # ~100 queries
  test_split: test_300             # never touched during optimization
optimizer:
  name: qkernel_bo                 # random | tpe | nsga2 | structured_bo | qkernel_bo
  init_points: 10
  quantum:                         # only read by qkernel_bo
    shots: null                    # null = statevector
    feature_map_depth: 1
generator:
  model: qwen2.5-3b-instruct-4bit
  max_new_tokens: 64
search_space: configs/search_spaces/rag_v2.yaml
hardware:
  max_vram_gb: 7.0
```

`rag_v2.yaml` defines the 9 decision variables with domains, groups G1–G5, and the two conditional rules (`dense_weight` requires `hybrid_on`; `reranker_top_n` requires `reranker_on`). Domains: chunking_strategy {fixed, sentence, semantic}; chunk_size {256,384,512,768}; chunk_overlap {0,32,64,96}; embedding_model {bge-small, e5-base, gte-small}; top_k {3,5,8,10}; hybrid_on {0,1}; dense_weight {0.3,0.5,0.7,0.9}; reranker_on {0,1}; reranker_top_n {3,5,8}; context_budget {1024,1536,2048}.

## 8. SQLite schema (minimum)

`trials`: experiment_id, optimizer_name, seed, dataset_id, trial_index, config_hash, config_json, objective_f1, objective_faithfulness, objective_latency_s, em, recall_at_k, ndcg, tokens_prompt, tokens_generated, peak_vram_mb, cache_hits_json, duplicate_flag, wall_clock_s, timestamp.
`experiments`: experiment_id, config_yaml, git_commit, package_versions_json, hardware_json, status, started_at, finished_at.
`kernel_log` (quantum runs): experiment_id, trial_index, shots, psd_clipped_mass, kernel_time_s.
Plus artifact tables for L3–L5. Every result the paper will report must be reconstructable from the DB alone.

## 9. Implementation phases — build in this order, stop at each gate

| Phase | Deliverable | Acceptance gate (verify before moving on) |
|---|---|---|
| P1 | repo skeleton, env, `check_env.py`, `space/` (search space, encoding, constraints) + tests | encode/decode round-trip property tests pass; invalid combos rejected; `ruff` clean |
| P2 | `store/` (hashing, artifact store, db) + `rag/` full pipeline + `evaluation/` | one config evaluates end-to-end on a 10-query toy set; re-running is a 100% cache hit and bit-identical; VRAM stays < 7 GB |
| P3 | orchestrator + random_search + tpe + resume + JSONL logging | full budget-25 run on toy set completes; kill −9 mid-run then `--resume` reproduces the uninterrupted result exactly |
| P4 | `surrogate/` + `kernels/structural.py` + `kernels/classical.py` + structured_bo | structured_bo beats random search on a cheap synthetic function; GP fit time < 2 s at n=100 |
| P5 | `kernels/quantum.py` + qkernel_bo | quantum and classical BO diff only in kernel factory (assert in test); PSD projection tested; kernel time < 5% of a real evaluation |
| P6 | multi-objective: EHVI, Pareto archive, hypervolume, nsga2 | hypervolume validated against pymoo reference values; 3-objective toy run end-to-end |
| P7 | `synth/` Layer-A benchmark + `run_synth.py` | reproduces a full synthetic sweep (interaction × conditionality × dim) with plots |
| P8 | `run_ablation.py`, `export_results.py`, Dockerfile, README, docs | `docker build` + one command reproduces the toy-set main table |

Do not start P5 before P4's gate passes, etc. At each gate, print a short report and wait for user confirmation.

## 10. Quality standards

- Type hints everywhere; `ruff` with no ignores beyond line length; Google-style docstrings on public APIs.
- **Mandatory unit tests:** encoding round-trip (property-based, hypothesis-style loops are fine with plain pytest), constraint enforcement, hash stability across process restarts, kernel PSD after clipping, structural-kernel symmetry K(x,x')=K(x',x), classical-vs-quantum code-path symmetry, cache determinism, resume equivalence, hypervolume correctness.
- No global mutable state except the registries. No `print` in library code — use `utils/logging`.
- Commit style: small commits per module, message prefix `[P<phase>]`.
- Never download models implicitly inside library code; model fetch happens in `scripts/` with explicit user-visible progress.

## 11. What NOT to do

- Do not add LangChain, LlamaIndex, Hydra, MLflow, or a web dashboard. SQLite + JSONL + matplotlib is the stack.
- Do not implement extra optimizers, extra metrics, extra quantum ansätze, or "nice to have" abstractions beyond this brief.
- Do not use sampling-based decoding anywhere, including "just for testing".
- Do not let the quantum path acquire any capability (feature scaling, extra hyperparameters, warm starts) the classical path lacks, or vice versa.
- Do not swallow exceptions in the orchestrator; a failed trial is recorded with `status=failed` and the run continues.

## 12. Questions you should ask the user before/while building

1. Which exact HuggingFace model IDs to pin for generator, embeddings, reranker, and NLI model (licenses + availability on his machine).
2. Dataset preparation: does he already have HotpotQA/SciFact subsets locally, or should `scripts/` include dataset download/subsetting code?
3. Confirm Python version available on the target Ubuntu machine.

Everything else in this brief is decided. Begin with Phase P1.

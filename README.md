# QAutoLLM-RAG

Budget-aware, multi-objective optimization of RAG pipeline configurations
using classical and quantum-kernel Bayesian Optimization. Software artifact
for the paper's core question: under what conditions does a
group-decomposed quantum fidelity kernel BO surrogate outperform an
identically-structured classical Matérn kernel BO surrogate, on a mixed,
conditional configuration space and a small evaluation budget?

Full design spec: `QAutoLLM_Implementation_Brief.md`.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# install the CUDA build of torch BEFORE requirements.txt (adjust cuXXX to your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install -e .
```

Verify the environment (CUDA availability, VRAM, package versions):

```bash
python scripts/check_env.py
```

After the first successful end-to-end run, freeze resolved versions:

```bash
pip freeze > requirements.lock
```

Run the test suite (`-m "not slow"` skips the tests that download small
real HF models for a one-time integration check):

```bash
ruff check .
pytest -q -m "not slow"
```

## Quickstart: toy-set experiments (no GPU, no model downloads)

Every command below runs against a fully synthetic toy dataset with a fake
generator/reranker/faithfulness scorer.

```bash
# one experiment, one optimizer
python -m scripts.run_experiment --config configs/experiments/toy_smoke.yaml
python -m scripts.run_experiment --config configs/experiments/toy_smoke.yaml --resume

# every optimizer (random, tpe, structured_bo, qkernel_bo, nsga2) against the
# toy set in one command -> results/ablation/{main_table.md,main_table.csv,convergence.png}
python -m scripts.run_ablation

# tables/plots from any set of experiment DBs
python -m scripts.export_results --db results/ablation/ablation_random/db.sqlite \
    --db results/ablation/ablation_structured_bo/db.sqlite --output-dir results/export

# Layer-A synthetic benchmark sweep (interaction x conditionality x dim)
python -m scripts.run_synth --quick   # seconds, smoke-scale
python -m scripts.run_synth           # full scale — can take a couple of hours
                                       # (dominated by structured_bo/qkernel_bo GP fits)
```

## Real experiments (HotpotQA + real models, needs a GPU)

Brief §12 is resolved: generator is Qwen2.5-3B-Instruct (4-bit), other
model defaults are in `configs/models/*.yaml`, and HotpotQA is downloaded
+ subset by `scripts/prepare_hotpotqa.py` (one shared corpus — the union of
every paragraph across both splits — rather than HotpotQA's own
per-question 10-paragraph distractor set, so retrieval is a real search
over many candidates, not a pick-2-of-10 task).

```bash
# one-time: download + build opt_100/test_300 (fast, no GPU, network only)
python -m scripts.prepare_hotpotqa --n-opt 100 --n-test 300

# real run: downloads Qwen2.5-3B-Instruct (~6GB) + real embedding/reranker/
# NLI models on first use, needs a CUDA GPU (Brief hardware target: 8GB VRAM)
python -m scripts.run_experiment --config configs/experiments/main_hotpot.yaml
```

**Performance note**: `scripts/build_index.py` pre-builds all
chunking x embedding combinations (Brief D1) — with the toy dataset's fake
embedder this is instant, but with *real* embedding models on a real
corpus, encoding every chunking variant for all 3 embedding models is
genuinely slow (CPU-bound sentence-transformers encoding, no way around
it) and can take a long time depending on corpus size. It only needs to
run once per corpus though — the Artifact Store caches every
(chunking, embedding) combination, so subsequent runs against the same
`data/hotpotqa/` are fast.

## Docker

```bash
docker build -f docker/Dockerfile -t qautollm .
docker run --rm qautollm        # runs scripts.run_ablation, prints the main table
```

The image only covers the CPU-only toy-set path (see `docker/Dockerfile`'s
header comment) — it does not install a CUDA torch build or request GPU
passthrough.

## Project status

All 8 build phases (Brief §9) are implemented and gated:

| Phase | What | Gate |
|---|---|---|
| P1 | search space, encoding, config, registry | encode/decode round-trip; invalid combos rejected |
| P2 | Artifact Store (L1-L5), RAG pipeline, evaluation | toy-set eval end-to-end; 100% cache hit + bit-identical on rerun |
| P3 | orchestrator, random/tpe optimizers, resume | budget-25 run + real `kill -9` + `--resume` reproduces it exactly |
| P4 | surrogate GP, structural/classical kernel, structured_bo | structured_bo beats random search; GP fit < 2s @ n=100 |
| P5 | quantum fidelity kernel, qkernel_bo | classical/quantum diff only in kernel factory; PSD projection tested |
| P6 | EHVI, Pareto archive, hypervolume, nsga2 | hypervolume validated against pymoo; 3-objective toy run |
| P7 | Layer-A synthetic benchmark, run_synth.py | full sweep (interaction x conditionality x dim) with plots |
| P8 | ablation, export_results, Dockerfile | docker build + one command reproduces the toy-set main table |

Brief §12 is resolved: model IDs confirmed (`configs/models/*.yaml`),
HotpotQA is downloaded/prepared by `scripts/prepare_hotpotqa.py`. What's
still outstanding is *running* the real (GPU-backed, multi-GB download)
experiment campaign — that's the user's manual step, same pattern used
throughout this project for anything heavy/environment-dependent.

# Architecture notes

Orientation for reading the codebase; the authoritative spec is
`QAutoLLM_Implementation_Brief.md` at the repo root.

## Data flow, one trial

```
SearchSpace.sample()/sobol_design() -> raw dict
    -> encode(raw, space) -> CanonicalConfig (z-vector + group mask + hash)
    -> Evaluator.evaluate(config) -> EvalResult (objectives = [-F1, -faithfulness, latency])
    -> optimizer.update(config, objectives)
    -> Orchestrator writes the trial row (db.sqlite + trials.jsonl)
```

Every optimizer (`qautollm/optimizers/`) implements `OptimizerInterface`
(`suggest`/`update`/`state_dict`/`load_state_dict`) and drains the same
shared Sobol init design before doing anything strategy-specific — that's
what makes the classical/quantum/baseline comparison paired rather than
independently-seeded noise.

## The classical/quantum symmetry

`structured_bo.py` and `qkernel_bo.py` both subclass the same `KernelBO`
(in `structured_bo.py`) and share every method — GP fitting, EHVI
acquisition, Pareto archive, resume — verbatim. They differ in exactly one
thing: which `GroupKernel` factory they pass into
`StructuralKernel.from_search_space()` (`kernels/classical.py`'s Matérn +
categorical-overlap vs. `kernels/quantum.py`'s PennyLane fidelity kernel).
`tests/test_kernel_bo_parity.py` asserts this at the function-object level,
not just behaviorally.

## The Artifact Store (why caching is load-bearing, not an optimization)

`qautollm/store/artifact_store.py` implements the L1-L5 cache from Brief
§5.3. L1/L2 (chunking, indexes) are files on disk — FAISS indexes aren't
reliably picklable across versions, so `compute_fn` for those levels
receives a destination directory to write into rather than returning a
value. L3/L4 (retrieval, generation) are rows in `db.sqlite`'s `artifacts`
table. Every `evaluate()` call is cache-first: with ~6000 planned pipeline
evaluations on one laptop, this cache is what makes the schedule fit, not
a nice-to-have.

## Determinism

`qautollm/utils/seeds.py::seed_everything(seed)` is the one place all RNGs
get seeded. Two non-obvious sources of hidden non-determinism found and
fixed during development, worth knowing if you touch the optimizer code:

- **TPE resume**: Optuna's `TPESampler` consumes its own RNG on every live
  `ask()` call, independent of what the trial is later told — rebuilding a
  study by replaying (params, objective) history is *not* equivalent to
  continuing the original study. Fixed by pickling the live `optuna.Study`
  into the checkpoint instead (`optimizers/tpe.py`).
- **KernelBO resume**: BoTorch's `fit_gpytorch_mll` can reach for torch's
  *global* RNG on some fallback path, not the optimizer's own seeded RNG.
  Fixed with `torch.manual_seed(self.seed)` at the top of
  `KernelBO._suggest_via_ehvi` (`optimizers/structured_bo.py`).

## What's still open

Brief §12: exact HuggingFace model IDs (defaults live in
`configs/models/*.yaml`, override there) and real dataset (HotpotQA/SciFact)
loading — `scripts/run_experiment.py`/`run_ablation.py` only wire up the
fully-synthetic toy dataset for now.

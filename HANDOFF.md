# Session handoff (temporary — delete in the next commit)

This file exists only so a fresh session can pick up the work. Remove it
with `git rm HANDOFF.md` in the first commit of the next session.

## State as of 2026-10-23
- Code: 8 build phases done, tests green (`pytest -q -m "not slow"`; re-run to confirm).
- Experiments: all finished, no pending compute. Raw per-seed DBs live in
  `results/main_hotpot_v3/{seed42..51,_shared}` (git-ignored, ~1.9 GB, local only);
  `data/` is also git-ignored and local only.
- Headline result: negative result — no quantum advantage on hypervolume in any
  regime; classical significantly ahead on the real task (Wilcoxon p=0.0195, n=10).
  F1 edge for quantum does not survive Holm (adj. p=0.074) or held-out replication
  (p=0.695). Faithfulness: quantum significantly worse (Holm p=0.012).
  Mechanism: <=5 qubits/group caps the quantum Gram matrix's effective rank.

## What is left (all in `paper/manuscript.md`, `grep -n TODO`)
1. §1 Introduction — outline present, prose not written.
2. §2 Related Work — outline/citations list present, prose not written.
3. §3.4 — restate Brief §5.2 fairness/pairing rules + "all regimes reported
   regardless of outcome" commitment.
4. §7 Limitations — four bullets drafted (exact statevector, depth=1 in real
   campaign, <=5 qubit cap, single task); expand to prose.
5. §8 Conclusion — text exists; one consistency pass after 1/2/4 are written.
6. Appendix A — fill in the final code commit hash.
7. Appendix B — expand the Clifford-periodicity bug write-up (sources named in the TODO).
8. Appendix C — one paragraph on abandoned lightning.gpu attempt / OOM hardening (likely cut).
9. After the manuscript: pick a venue, then fill `journal`/`doi` in the README citation.

## Conventions
- Commits: no Co-Authored-By / attribution trailers; author is the repo owner only.
- Nothing in the manuscript is hand-entered; tables/figures regenerate via the
  commands in README "Results".
- Caches (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `*.egg-info`) are
  disposable and git-ignored; re-run `pip install -e .` if the package is not importable.

## Moving to a new machine
Source machine used Python 3.12.3 (`pyproject.toml` requires >=3.11). Copy the
whole folder except `.venv/` (keep `.git/`, `data/`, `results/` incl. the
git-ignored `results/main_hotpot_v3/seed*/` and `_shared/` raw DBs).

```bash
cd QAutoLLM_RAG
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # only if a GPU is present; match driver
pip install -r requirements.txt     # requirements.lock is the exact freeze from the old machine
pip install -e .
python scripts/check_env.py
pytest -q -m "not slow"             # expect 271 passed (~4 min)
git status                          # should be clean; no hard-coded absolute paths exist in the repo
```
- Hugging Face models/datasets are NOT in the folder (`~/.cache/huggingface`,
  37 GB on the old machine). They re-download on first use, and are only
  needed to run new real experiments, not to write the paper or regenerate tables.
- Regenerating tables needs the raw DBs: `python -m scripts.export_results ...`
  (see README "Results"). Without them, the tracked CSV/JSON in `results/` suffice.
- Remote `origin` = github.com/MuratKarslioglu/QAutoLLM_RAG; local `main` was
  not pushed as of this handoff — check `git status -sb`.

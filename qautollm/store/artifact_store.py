"""Content-addressed Artifact Store: the L1-L5 cache (Brief §5.3).

Two storage backends, chosen per level:
  - **File levels** (L1 chunk lists, L2 dense+BM25 indexes): each entry is a
    directory ``artifacts/<level>/<key>/``. FAISS indexes aren't reliably
    picklable across faiss-cpu versions, so file levels always get a real
    directory to write into rather than a single returned value — pass
    ``compute_fn(dest: Path) -> None`` that writes whatever files it needs.
  - **DB levels** (L3 retrieval, L4 generation, L5 kernel entries): each
    entry is one row in :class:`qautollm.store.db.ArtifactDB`'s shared
    ``artifacts`` table — pass ``compute_fn() -> JSON-serializable value``.

**Latency accounting rule** (Brief §5.3): ``latency_s`` on the returned
:class:`Artifact` is always the wall-clock time of the *cold* compute that
first produced the entry. A cache hit reuses that stored latency rather than
measuring the (near-zero) cache lookup time — never conflate the two.
"""

from __future__ import annotations

import shutil
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qautollm.store.db import ArtifactDB
from qautollm.store.hashing import canonical_hash

_FILE_LEVELS = ("L1", "L2")
_DB_LEVELS = ("L3", "L4", "L5")
_ALL_LEVELS = _FILE_LEVELS + _DB_LEVELS

_LATENCY_SIDECAR = "_latency_s.txt"


@dataclass(frozen=True)
class Artifact:
    key: str
    level: str
    value: Any  # Path (file levels) or JSON-able value (db levels)
    hit: bool
    latency_s: float


class ArtifactStore:
    def __init__(self, root: str | Path, db: ArtifactDB) -> None:
        self._root = Path(root)
        self._db = db
        self._hits: dict[str, int] = defaultdict(int)
        self._misses: dict[str, int] = defaultdict(int)

    def get_or_compute(
        self,
        level: str,
        key_fields: dict[str, Any],
        compute_fn: Callable[..., Any],
    ) -> Artifact:
        if level not in _ALL_LEVELS:
            raise ValueError(f"unknown artifact level {level!r}; expected one of {_ALL_LEVELS}")
        key = canonical_hash(key_fields)
        if level in _FILE_LEVELS:
            return self._get_or_compute_file(level, key, compute_fn)
        return self._get_or_compute_db(level, key, compute_fn)

    def _get_or_compute_file(
        self, level: str, key: str, compute_fn: Callable[[Path], None]
    ) -> Artifact:
        dest = self._root / level / key
        latency_path = dest / _LATENCY_SIDECAR

        if dest.exists() and latency_path.exists():
            self._hits[level] += 1
            latency_s = float(latency_path.read_text())
            return Artifact(key=key, level=level, value=dest, hit=True, latency_s=latency_s)

        self._misses[level] += 1
        if dest.exists():
            shutil.rmtree(dest)  # leftover from a previously interrupted compute
        dest.mkdir(parents=True)

        start = time.perf_counter()
        compute_fn(dest)
        latency_s = time.perf_counter() - start
        latency_path.write_text(repr(latency_s))
        return Artifact(key=key, level=level, value=dest, hit=False, latency_s=latency_s)

    def _get_or_compute_db(self, level: str, key: str, compute_fn: Callable[[], Any]) -> Artifact:
        cached = self._db.get_artifact(level, key)
        if cached is not None:
            self._hits[level] += 1
            value, latency_s = cached
            return Artifact(key=key, level=level, value=value, hit=True, latency_s=latency_s)

        self._misses[level] += 1
        start = time.perf_counter()
        value = compute_fn()
        latency_s = time.perf_counter() - start
        self._db.put_artifact(level, key, value, latency_s)
        return Artifact(key=key, level=level, value=value, hit=False, latency_s=latency_s)

    def report(self) -> dict[str, dict[str, float | int]]:
        """Per-level hit/miss counts and hit rate, for the run summary."""
        out: dict[str, dict[str, float | int]] = {}
        for level in sorted(set(self._hits) | set(self._misses)):
            hits, misses = self._hits[level], self._misses[level]
            total = hits + misses
            hit_rate = hits / total if total else 0.0
            out[level] = {"hits": hits, "misses": misses, "hit_rate": hit_rate}
        return out

"""SQLite schema + writers/readers (Brief §8).

Every result the paper reports must be reconstructable from this DB alone.
Holds: the L3-L5 artifact cache (as one generic ``artifacts`` table, since
all three levels share the same shape — key/value/latency), plus
``trials``/``experiments``/``kernel_log`` written by the orchestrator (P3+).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    level TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    latency_s REAL NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (level, key)
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    config_yaml TEXT,
    git_commit TEXT,
    package_versions_json TEXT,
    hardware_json TEXT,
    status TEXT,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS trials (
    experiment_id TEXT NOT NULL,
    optimizer_name TEXT,
    seed INTEGER,
    dataset_id TEXT,
    trial_index INTEGER NOT NULL,
    config_hash TEXT,
    config_json TEXT,
    objective_f1 REAL,
    objective_faithfulness REAL,
    objective_latency_s REAL,
    em REAL,
    recall_at_k REAL,
    ndcg REAL,
    tokens_prompt INTEGER,
    tokens_generated INTEGER,
    peak_vram_mb REAL,
    cache_hits_json TEXT,
    duplicate_flag INTEGER,
    wall_clock_s REAL,
    status TEXT,
    timestamp TEXT,
    PRIMARY KEY (experiment_id, trial_index)
);

CREATE TABLE IF NOT EXISTS kernel_log (
    experiment_id TEXT NOT NULL,
    trial_index INTEGER NOT NULL,
    shots INTEGER,
    psd_clipped_mass REAL,
    kernel_time_s REAL
);
"""


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class ArtifactDB:
    """SQLite-backed store for L3-L5 artifacts plus the experiment/trial/
    kernel-log tables. One connection per process; safe for the single-
    process orchestrator this codebase targets (Brief §11: no distributed
    infra).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- L3-L5 artifact cache -------------------------------------------------

    def get_artifact(self, level: str, key: str) -> tuple[Any, float] | None:
        row = self._conn.execute(
            "SELECT value_json, latency_s FROM artifacts WHERE level = ? AND key = ?",
            (level, key),
        ).fetchone()
        if row is None:
            return None
        value_json, latency_s = row
        return json.loads(value_json), latency_s

    def put_artifact(self, level: str, key: str, value: Any, latency_s: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO artifacts (level, key, value_json, latency_s, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                level,
                key,
                json.dumps(_to_jsonable(value), sort_keys=True),
                latency_s,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    # -- experiments / trials / kernel_log ------------------------------------

    def write_experiment(self, row: dict[str, Any]) -> None:
        self._upsert("experiments", row)

    def write_trial(self, row: dict[str, Any]) -> None:
        self._upsert("trials", row)

    def write_kernel_log(self, row: dict[str, Any]) -> None:
        columns = sorted(row)
        placeholders = ", ".join("?" for _ in columns)
        self._conn.execute(
            f"INSERT INTO kernel_log ({', '.join(columns)}) VALUES ({placeholders})",
            [row[c] for c in columns],
        )
        self._conn.commit()

    def read_trials(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        if experiment_id is None:
            cur = self._conn.execute("SELECT * FROM trials ORDER BY experiment_id, trial_index")
        else:
            cur = self._conn.execute(
                "SELECT * FROM trials WHERE experiment_id = ? ORDER BY trial_index",
                (experiment_id,),
            )
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]

    def _upsert(self, table: str, row: dict[str, Any]) -> None:
        columns = sorted(row)
        placeholders = ", ".join("?" for _ in columns)
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            [row[c] for c in columns],
        )
        self._conn.commit()

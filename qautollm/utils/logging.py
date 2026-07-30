"""Structured JSONL run logging (Brief §4, §10).

Library code must never call ``print`` (Brief §10). Use
:func:`get_logger` for human-readable progress messages (stdlib logging,
one line per event) and :class:`JsonlWriter` for the machine-readable event
stream that gets replayed for resume/analysis.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib logger writing single-line, leveled messages to stderr.

    Configuring `logging` (not `print`) keeps progress output separable from
    piped/redirected stdout, and gives every caller consistent timestamps.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            stream=sys.stderr,
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def add_file_logging(path: str | Path) -> None:
    """Mirror every subsequent log line (and any uncaught exception) to
    ``path``, in addition to stderr.

    Long real-dataset runs live entirely in one terminal session; if that
    session closes (SSH drop, terminal killed, laptop suspend) before the
    process exits, an unhandled exception's traceback is gone forever and a
    run that died at trial 50/100 leaves no trace anywhere of why. Call once
    near the top of a script's ``main()``, as soon as the run/output
    directory is known.
    """
    get_logger(__name__)  # ensure the stderr handler is configured first
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)

    root_logger = logging.getLogger(__name__)

    def _log_uncaught(exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        root_logger.critical(
            "unhandled exception -- process exiting", exc_info=(exc_type, exc_value, exc_tb)
        )
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _log_uncaught


class JsonlWriter:
    """Append-only structured event log: one JSON object per line.

    Used for the per-trial event stream that ``--resume`` replays and that
    ``export_results.py`` reads back for plots/tables — never parse stdout
    for this, always read the JSONL file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str))
            f.write("\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        with open(self._path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

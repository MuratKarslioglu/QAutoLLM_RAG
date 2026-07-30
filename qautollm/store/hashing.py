"""Canonical content hashing of parameter dicts (Brief §5.3, §5.1).

Used both for `CanonicalConfig.config_hash` and for every Artifact Store
cache key (L1-L5). One function, one place: every hash in the codebase for
a given kind of key must go through here so cache keys stay stable across
process restarts and across the two BO variants.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(fields: dict[str, Any]) -> str:
    """Deterministic JSON serialization: sorted keys, no whitespace ambiguity."""
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)


def canonical_hash(fields: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical JSON serialization of ``fields``."""
    payload = canonical_json(fields).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pair_hash(hash_a: str, hash_b: str) -> str:
    """Order-independent hash of two hashes (used for L5 kernel-entry keys)."""
    lo, hi = sorted((hash_a, hash_b))
    return canonical_hash({"a": lo, "b": hi})

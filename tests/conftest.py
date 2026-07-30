import sys
from pathlib import Path

# pytest's default "prepend" import mode adds tests/ (no __init__.py above it)
# to sys.path, not the repo root — add the root too so `scripts.*` (a plain
# namespace package, meant to be run as CLI entry points, not pip-installed)
# is importable from test modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#!/usr/bin/env python3
"""
Interactive / query CLI using **ECS active corpus + VectorStore** only.

Same as ``python scripts/run_with_docs.py --ecs ...`` — no ``docs/`` path and no CLI re-ingest
from disk. Ingest via ``scripts/start_corpus_ingest.py`` (or equivalent) first.

Examples:
  python scripts/run_with_ecs.py --interactive
  python scripts/run_with_ecs.py --query "What is …?"
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

_rest = sys.argv[1:]
if "--ecs" not in _rest:
    sys.argv = [sys.argv[0], "--ecs", *_rest]

_spec = importlib.util.spec_from_file_location("_run_with_docs_cli", _root / "scripts" / "run_with_docs.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
_mod.main()

#!/usr/bin/env python3
"""
Lightweight environment check for ReChunk (humans + CI).

Run from repo root:  python scripts/rechunk_doctor.py

Exits 0 always unless --strict is set (then exits 1 if any check fails).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))


def _ok(msg: str) -> None:
    print(f"  OK   {msg}")


def _warn(msg: str) -> None:
    print(f"  WARN {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL {msg}")


def check_python() -> bool:
    ok = sys.version_info >= (3, 10)
    if ok:
        _ok(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    else:
        _fail("Python 3.10+ required")
    return ok


def check_openai_key() -> bool:
    if os.environ.get("OPENAI_API_KEY", "").strip():
        _ok("OPENAI_API_KEY is set (needed for embeddings + LLM chunking + CLI synthesis)")
        return True
    _warn("OPENAI_API_KEY unset — vectorization worker and query synthesis will fail until set")
    return False


def check_rechunk_import() -> bool:
    try:
        import rechunk  # noqa: F401

        _ok("Package `rechunk` imports (pip install -e . from repo root)")
        return True
    except Exception as e:
        _fail(f"Cannot import rechunk: {e}")
        return False


def check_strategies_file() -> bool:
    p = _project_root / "rechunk_strategies.json"
    if p.is_file():
        _ok(f"rechunk_strategies.json present ({p.name})")
        return True
    _warn("No rechunk_strategies.json — run e.g. run_interactive / run_with_docs to create defaults")
    return False


def check_storage_dirs() -> bool:
    ecs = _project_root / "storage" / "ecs"
    vs = _project_root / "storage" / "vector_store_dev"
    _ok(f"storage/ecs exists={ecs.is_dir()}  storage/vector_store_dev exists={vs.is_dir()}")
    return True


async def check_temporal_connect(addr: str, timeout: float) -> bool:
    try:
        from temporalio.client import Client

        await asyncio.wait_for(Client.connect(addr), timeout=timeout)
        _ok(f"Temporal server reachable at {addr!r}")
        return True
    except Exception as e:
        _warn(f"Temporal not reachable at {addr!r} ({e}) — start e.g. `temporal server start-dev`")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="ReChunk environment doctor")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if OPENAI_API_KEY missing or Temporal unreachable",
    )
    parser.add_argument(
        "--temporal-address",
        default=None,
        help="Override TEMPORAL_ADDRESS (default from env or localhost:7233)",
    )
    parser.add_argument(
        "--temporal-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for Temporal connect (default 3)",
    )
    args = parser.parse_args()

    print("ReChunk doctor\n")
    ok_python = check_python()
    ok_import = check_rechunk_import()
    check_storage_dirs()
    check_strategies_file()
    key_ok = check_openai_key()

    addr = (args.temporal_address or os.environ.get("TEMPORAL_ADDRESS") or "localhost:7233").strip()
    temporal_ok = asyncio.run(check_temporal_connect(addr, args.temporal_timeout))

    emb = os.environ.get("RECHUNK_OPENAI_EMBEDDING_MODEL", "").strip()
    if emb:
        _ok(f"RECHUNK_OPENAI_EMBEDDING_MODEL={emb!r}")
    else:
        _ok("RECHUNK_OPENAI_EMBEDDING_MODEL unset (default text-embedding-3-small in code)")

    print("\nNext steps (full ECS + VectorStore path):")
    print("  1. Terminal A:  python temporal_workers.py")
    print("  2. Ingest:     python scripts/start_corpus_ingest.py path/to/docs --wait")
    print("  3. Vectorize:  python scripts/start_strategy_chunking.py s_default")
    print("  Or use:        python scripts/run_interactive.py path/to/docs")
    print("\nSee README.md (Quick start) and AGENTS.md for detail.")

    if not ok_python or not ok_import:
        return 1
    if args.strict and (not key_ok or not temporal_ok):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

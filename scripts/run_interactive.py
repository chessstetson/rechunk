#!/usr/bin/env python3
"""
Interactive entrypoint: **corpus path → local ECS ingest → queue initial embeddings** (Temporal), then Q&A.

Delegates to ``run_with_docs.py --interactive``. It **does not** prompt for your first question until
the vector index has at least one chunk (embeddings from the worker / cache).

Requires ``OPENAI_API_KEY`` and **``python temporal_workers.py``** so vectorization completes.

Examples::

    python scripts/run_interactive.py
    python scripts/run_interactive.py path/to/docs

Environment: ``TEMPORAL_ADDRESS`` (default ``localhost:7233``) for workflow submission.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_RUN_WITH_DOCS = _SCRIPT_DIR / "run_with_docs.py"


def _prompt_corpus_path() -> Path:
    try:
        raw = input("Corpus path (directory or document file): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.", file=sys.stderr)
        sys.exit(130)
    if not raw:
        print("No path given.", file=sys.stderr)
        sys.exit(1)
    return Path(raw).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest a document tree into ECS, queue strategy embeddings (Temporal), "
            "wait until chunks exist, then prompt for questions interactively."
        )
    )
    parser.add_argument(
        "corpus",
        nargs="?",
        type=Path,
        default=None,
        help="Root directory or single file (.txt / .md / .pdf / .docx)",
    )
    parser.add_argument("--model", default=None, help="OpenAI chat model (forwarded to run_with_docs)")
    parser.add_argument("--top-k", type=int, default=None, metavar="K", help="Retrieval top-k")
    parser.add_argument(
        "--no-vector-index-cache",
        action="store_true",
        help="Forward to run_with_docs (rebuild disk vector cache)",
    )
    args = parser.parse_args()

    path = args.corpus
    if path is None:
        path = _prompt_corpus_path()
    path = path.expanduser().resolve()
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    cmd: list[str] = [sys.executable, str(_RUN_WITH_DOCS), str(path), "--interactive"]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.no_vector_index_cache:
        cmd.append("--no-vector-index-cache")

    return int(subprocess.call(cmd, cwd=_SCRIPT_DIR.parent))


if __name__ == "__main__":
    raise SystemExit(main())

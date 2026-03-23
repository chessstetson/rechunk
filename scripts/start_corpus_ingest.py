#!/usr/bin/env python3
"""
Start :class:`FilesystemCorpusIngestWorkflow` — filesystem → ECS + active hash manifest only.

Does **not** chunk or embed. After this completes (and the ingest worker has processed the task),
run ``scripts/start_strategy_chunking.py <strategy_id>`` to enqueue vectorization workflows (reads ECS only).

Requires:
  * Temporal server
  * Ingest worker polling ``rechunk-ingest`` (``python temporal_workers.py ingest`` or ``both``)

Usage:
  python scripts/start_corpus_ingest.py <docs_root> [--wait] [--address HOST:PORT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from rechunk.temporal_client import trigger_filesystem_ingest_sync


def _doc_ids(docs_root: Path) -> list[str]:
    exts = {".txt", ".md", ".pdf", ".docx"}
    ids: list[str] = []
    for f in sorted(docs_root.rglob("*")):
        if f.is_file() and f.suffix.lower() in exts:
            ids.append(str(f.relative_to(docs_root)))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal ingest: corpus tree into ECS (no chunking)")
    parser.add_argument("docs_root", type=Path, help="Root directory of documents")
    parser.add_argument("--wait", action="store_true", help="Block until the ingest workflow completes")
    parser.add_argument("--address", default="localhost:7233", help="Temporal server address")
    args = parser.parse_args()

    root = args.docs_root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    doc_ids = _doc_ids(root)
    if not doc_ids:
        print(f"No .txt/.md/.pdf/.docx files under {root}", file=sys.stderr)
        sys.exit(1)

    wid = trigger_filesystem_ingest_sync(
        root,
        doc_ids,
        temporal_address=args.address,
        wait_for_result=args.wait,
    )
    if wid is None:
        sys.exit(1)
    if not args.wait:
        print(f"Workflow id: {wid}")


if __name__ == "__main__":
    main()

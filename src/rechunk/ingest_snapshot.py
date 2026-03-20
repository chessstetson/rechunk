"""
Temporal ingest snapshot: workflow carries a path to a JSON file, not doc lists in history.

The snapshot is written on the same filesystem the worker reads (e.g. under ``storage/ingest_snapshots/``)
before ``StrategyChunkingWorkflow`` starts. Format is versioned so a DB or remote store can replace
the writer later while keeping the same workflow input shape (``ingest_snapshot_path`` only).

Snapshot JSON (version 1)::

    {
      "version": 1,
      "docs_root": "/absolute/path/to/corpus",
      "documents": [
        {"doc_id": "relative/path.txt", "content_hash": "<sha256 hex>"}
      ]
    }
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from rechunk.cache import compute_content_hash
from rechunk.doc_loader import extract_file_content

INGEST_SNAPSHOT_VERSION = 1


def ingest_snapshot_dir() -> Path:
    env = os.environ.get("RECHUNK_INGEST_SNAPSHOT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "storage" / "ingest_snapshots"


def build_and_write_ingest_snapshot(
    docs_root: Path,
    doc_ids: list[str],
    *,
    strategy_id: str = "run",
) -> Path:
    """
    Hash each document under ``docs_root`` (same rules as ``load_doc_manifest``), write snapshot file.

    Returns path to the written JSON (absolute). Skips unreadable / missing files (not listed).
    """
    root = docs_root.resolve()
    documents: list[dict[str, str]] = []
    for doc_id in doc_ids:
        path = root / doc_id
        if not path.exists():
            continue
        text = extract_file_content(path)
        if not text or not text.strip():
            continue
        h = compute_content_hash(text)
        documents.append({"doc_id": doc_id, "content_hash": h})

    payload = {
        "version": INGEST_SNAPSHOT_VERSION,
        "docs_root": str(root),
        "documents": documents,
    }
    out_dir = ingest_snapshot_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{strategy_id}_{uuid.uuid4().hex[:12]}.json"
    out = out_dir / name
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out.resolve()


def read_ingest_snapshot(snapshot_path: Path) -> tuple[Path, list[dict[str, str]]]:
    """
    Read and validate snapshot. Re-hashes each file and requires match with stored ``content_hash``.

    Returns ``(docs_root, manifest)`` where ``manifest`` items are ``{"doc_id", "content_hash"}``
    compatible with the rest of the chunking pipeline.
    """
    path = snapshot_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Ingest snapshot not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Ingest snapshot must be a JSON object")
    if raw.get("version") != INGEST_SNAPSHOT_VERSION:
        raise ValueError(f"Unsupported ingest snapshot version: {raw.get('version')}")
    docs_root = Path(raw["docs_root"])
    docs = raw.get("documents") or []
    if not isinstance(docs, list):
        raise ValueError("documents must be a list")

    manifest: list[dict[str, str]] = []
    for i, row in enumerate(docs):
        if not isinstance(row, dict):
            raise ValueError(f"documents[{i}] must be an object")
        doc_id = row.get("doc_id")
        expected_hash = row.get("content_hash")
        if not doc_id or not expected_hash:
            raise ValueError(f"documents[{i}] needs doc_id and content_hash")
        fp = docs_root / doc_id
        if not fp.exists():
            continue
        text = extract_file_content(fp)
        if not text or not text.strip():
            continue
        actual = compute_content_hash(text)
        if actual != expected_hash:
            raise ValueError(
                f"Content hash mismatch for {doc_id!r}: snapshot {expected_hash[:12]}... "
                f"vs disk {actual[:12]}..."
            )
        manifest.append({"doc_id": doc_id, "content_hash": expected_hash})

    return docs_root, manifest

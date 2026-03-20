"""
Active corpus manifest: hash-only JSON maintained by ingest (Temporal worker).

After a successful :class:`StrategyChunkingWorkflow` run, the worker merges this
workflow's manifest ``content_hash`` values into a single sorted deduped list so
the Q&A CLI can use ``--manifest`` without a manual export step.

Path: ``storage/corpus_content_hashes.json`` under project root by default, or
``RECHUNK_ACTIVE_CORPUS_MANIFEST`` (absolute or relative path).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from rechunk.hash_manifest import normalize_content_hash


def active_corpus_manifest_path() -> Path:
    env = os.environ.get("RECHUNK_ACTIVE_CORPUS_MANIFEST")
    if env:
        return Path(env).expanduser().resolve()
    # rechunk/active_corpus_manifest.py -> parents[2] = project root
    return Path(__file__).resolve().parents[2] / "storage" / "corpus_content_hashes.json"


def merge_content_hashes_into_active_manifest(new_hashes: Iterable[str]) -> list[str]:
    """
    Union ``new_hashes`` with any existing hash-only JSON array at
    :func:`active_corpus_manifest_path`, then write sorted unique lowercase hex strings.

    Invalid entries in an existing file are skipped. Creates parent dirs as needed.
    Returns the merged sorted list written to disk.
    """
    path = active_corpus_manifest_path()
    existing: set[str] = set()
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            raw = []
        if isinstance(raw, list):
            for x in raw:
                if isinstance(x, str):
                    try:
                        existing.add(normalize_content_hash(x))
                    except ValueError:
                        continue

    for h in new_hashes:
        existing.add(normalize_content_hash(h))

    merged = sorted(existing)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return merged

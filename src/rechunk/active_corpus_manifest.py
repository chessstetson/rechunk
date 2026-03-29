"""
Active corpus manifest: hash-only JSON maintained by ingest (Temporal worker).

Ingest updates this hash-only artifact from ECS (see ``IndexService.sync_active_manifest_file``).
The Q&A CLI can use ``--manifest`` to point at the same path without a manual export step.

Path: ``storage/corpus_content_hashes.json`` under project root by default, or
``RECHUNK_ACTIVE_CORPUS_MANIFEST`` (absolute or relative path).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from rechunk.hash_manifest import normalize_content_hash
from rechunk.repo_paths import project_root


def active_corpus_manifest_path() -> Path:
    env = os.environ.get("RECHUNK_ACTIVE_CORPUS_MANIFEST")
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / "storage" / "corpus_content_hashes.json"


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


def write_active_manifest_exact(hashes: Iterable[str]) -> list[str]:
    """
    Replace the active corpus manifest with **exactly** these hashes (sorted, unique).

    Use when the source of truth is :class:`ExtractedContentService` active membership,
    not a union of workflow outputs.
    """
    path = active_corpus_manifest_path()
    normalized: set[str] = set()
    for h in hashes:
        try:
            normalized.add(normalize_content_hash(h))
        except ValueError:
            continue
    merged = sorted(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return merged

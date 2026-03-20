"""
Hash-only corpus manifest: JSON wire format for the retrieval boundary.

Format: a UTF-8 JSON **array** of lowercase SHA-256 hex strings (64 chars each), e.g.::

    [ "abc9...64chars", "def0...64chars" ]

Alternatively an object for forward compatibility::

    { "content_hashes": [ "...", "..." ] }

No paths, titles, or storage IDs — hashes only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rechunk.corpus import ContentRef

_SHA256_HEX_LEN = 64
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _normalize_hash(h: str) -> str:
    s = h.strip().lower()
    if not _HEX_RE.match(s):
        raise ValueError(
            f"Invalid content_hash (expected {_SHA256_HEX_LEN} hex chars): {h!r}"
        )
    return s


def normalize_content_hash(h: str) -> str:
    """Validate and return lowercase SHA-256 hex (64 chars)."""
    return _normalize_hash(h)


def load_content_refs_from_manifest(manifest_path: Path) -> list[ContentRef]:
    """
    Load :class:`ContentRef` instances (hash only, no ``source_hint``) from JSON.

    Deduplicates repeated hashes while preserving first-seen order.
    """
    path = manifest_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "content_hashes" in raw:
        hashes = raw["content_hashes"]
    elif isinstance(raw, list):
        hashes = raw
    else:
        raise ValueError(
            "Manifest must be a JSON array of hex strings or "
            '{"content_hashes": ["...", ...]}'
        )
    if not isinstance(hashes, list):
        raise ValueError("content_hashes must be a JSON array")
    out: list[ContentRef] = []
    seen: set[str] = set()
    for i, item in enumerate(hashes):
        if not isinstance(item, str):
            raise ValueError(f"Manifest entry {i} must be a string (hex hash), got {type(item)}")
        hl = _normalize_hash(item)
        if hl in seen:
            continue
        seen.add(hl)
        out.append(ContentRef(content_hash=hl, source_hint=None))
    if not out:
        raise ValueError("Manifest contains no valid content hashes")
    return out


def write_hash_manifest(manifest_path: Path, content_hashes: list[str]) -> None:
    """
    Write a JSON array of normalized lowercase SHA-256 hashes (deduped, order preserved).
    """
    seen: set[str] = set()
    normalized: list[str] = []
    for h in content_hashes:
        hl = _normalize_hash(h)
        if hl not in seen:
            seen.add(hl)
            normalized.append(hl)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")


def write_manifest_from_filesystem_scan(corpus_path: Path, manifest_path: Path) -> None:
    """
    Scan ``corpus_path`` (file or tree), hash each document, write hash-only manifest.

    Uses :func:`rechunk.corpus.scan_filesystem_corpus` (same rules as the CLI filesystem path).
    """
    from rechunk.corpus import scan_filesystem_corpus

    refs, _ = scan_filesystem_corpus(corpus_path)
    write_hash_manifest(manifest_path, [r.content_hash for r in refs])

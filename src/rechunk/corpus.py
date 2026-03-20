"""
Corpus layer: content identity and filesystem enumeration without holding full text.

Retrieval uses :class:`ContentRef` only. Enumeration walks the tree like
:func:`rechunk.documents.load_documents` but discards text after hashing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from rechunk.cache import compute_content_hash
from rechunk.doc_loader import extract_file_content


@dataclass(frozen=True, slots=True)
class ContentRef:
    """
    Identity for retrieval / cache pooling: ``content_hash`` only.

    ``source_hint`` is optional (e.g. relative path) for human-readable CLI
    messages only — not part of indexing identity.
    """

    content_hash: str
    source_hint: str | None = None


def scan_filesystem_corpus(path: Path) -> tuple[list[ContentRef], list[str]]:
    """
    Scan a file or directory tree for supported types; hash content and discard text.

    Returns ``(content_refs, doc_ids)`` with the **same length** and **aligned order**:
    each pair corresponds to one unique content hash kept in the corpus (duplicates
    by hash are skipped, matching :func:`rechunk.documents.load_documents`).

    ``doc_ids`` are relative paths under ``path`` (or the file basename for a single
    file) for **legacy** Temporal / worker wiring only.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    refs: list[ContentRef] = []
    doc_ids: list[str] = []
    seen_hashes: set[str] = set()

    if path.is_file():
        content = extract_file_content(path)
        if content:
            h = compute_content_hash(content)
            del content
            refs.append(ContentRef(content_hash=h, source_hint=path.name))
            doc_ids.append(path.name)
        else:
            raise FileNotFoundError(f"Unable to read or empty file: {path}")
    else:
        candidates = sorted(path.rglob("*"))
        for f in (p for p in candidates if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}):
            content = extract_file_content(f)
            if content:
                h = compute_content_hash(content)
                del content
                if h in seen_hashes:
                    print(f"[SKIP] Duplicate content (same hash): {f.relative_to(path)}", file=sys.stderr)
                    continue
                seen_hashes.add(h)
                rel = str(f.relative_to(path))
                refs.append(ContentRef(content_hash=h, source_hint=rel))
                doc_ids.append(rel)
            else:
                print(f"[WARN] Skipping {f}: could not extract text", file=sys.stderr)
        if not refs:
            raise FileNotFoundError(f"No supported files under {path}")

    return refs, doc_ids

"""
Data models for :mod:`rechunk.extracted_content` (v12 architecture).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExtractedContent:
    """
    Canonical extracted document record keyed by ``content_hash`` (hash of
    normalized extracted text).
    """

    content_hash: str
    logical_doc_id: str
    mime_type: str
    extracted_text: str | None = None
    structured_content: dict[str, Any] | None = None
    source_hint: str | None = None
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class SourceDocumentRef:
    """
    Reference to a source document for :meth:`ExtractedContentService.ensure_content`.

    Adapters populate a subset of fields (e.g. filesystem ingest uses ``path``).
    """

    logical_doc_id: str
    source_kind: str
    path: Path | None = None
    raw_bytes: bytes | None = None
    mime_type: str | None = None
    source_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

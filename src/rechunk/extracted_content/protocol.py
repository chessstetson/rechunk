"""
:class:`ExtractedContentService` protocol — single source of truth for document content (v12).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rechunk.extracted_content.models import ExtractedContent, SourceDocumentRef


@runtime_checkable
class ExtractedContentService(Protocol):
    """
    Accepts source documents, extracts/normalizes, deduplicates by ``content_hash``,
    stores content, and tracks which hashes are **active** (queryable).

    Downstream components pull :meth:`list_active_hashes` and diff against their own state.
    """

    def ensure_content(self, source_doc_ref: SourceDocumentRef) -> ExtractedContent:
        """
        Idempotent ingest: extract, normalize, store, update active membership per
        ``logical_doc_id`` rules (new hash active; prior hash for same logical id deactivated).
        """
        ...

    def deactivate_logical_doc(self, logical_doc_id: str, *, reason: str | None = None) -> None:
        """Remove logical document's content from the active/queryable set."""
        ...

    def apply_source_inventory(
        self,
        source_kind: str,
        observed_logical_doc_ids: list[str],
    ) -> None:
        """
        Optional reconciliation: deactivate active content for logical ids of this
        ``source_kind`` that are not in ``observed_logical_doc_ids``.
        """
        ...

    def get_content(self, content_hash: str) -> ExtractedContent:
        """Load extracted content by hash (used by vectorization activities)."""
        ...

    def has_content(self, content_hash: str) -> bool:
        """Whether this hash is known in storage (active or inactive)."""
        ...

    def list_active_hashes(self) -> list[str]:
        """
        Sorted list of **active** ``content_hash`` values (the current queryable corpus).

        Not the set of all hashes ever ingested.
        """
        ...

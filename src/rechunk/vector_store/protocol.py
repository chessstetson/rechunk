"""
:class:`VectorStore` protocol — spans + vectors in one store; no separate chunk cache (v12).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """
    **Rows**: per-document vectorized units (chunk spans + embedding + metadata), reusable
    when corpus membership changes as long as ``content_hash`` and strategy/embed config match.

    **Collections**: assembled retrieval views for a specific ``corpus_snapshot_id`` and
    strategy/embedding/schema triple. A miss triggers assembly from rows for active hashes.

    Row dicts are implementation-defined but should include at least ``content_hash``,
    ``span_start``, ``span_end``, and embedding payload per v12 plan.
    """

    def get_collection(
        self,
        *,
        corpus_snapshot_id: str,
        strategy_fingerprint: str,
        embedding_fingerprint: str,
        vector_schema_version: str,
    ) -> Any | None:
        """
        Return a cached collection handle (e.g. :class:`VectorStoreIndex`) or ``None``.
        """
        ...

    def put_collection(
        self,
        *,
        corpus_snapshot_id: str,
        strategy_fingerprint: str,
        embedding_fingerprint: str,
        vector_schema_version: str,
        index_obj: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a materialized collection for this corpus + retrieval configuration."""
        ...

    def list_vectorized_hashes(
        self,
        *,
        strategy_fingerprint: str,
        embedding_fingerprint: str,
        vector_schema_version: str,
    ) -> list[str]:
        """
        Content hashes that already have at least one vector row for this strategy/embed/schema.

        Used to diff against :meth:`ExtractedContentService.list_active_hashes` (global per
        strategy config, not scoped by ``corpus_snapshot_id``).
        """
        ...

    def upsert_rows(
        self,
        *,
        strategy_fingerprint: str,
        embedding_fingerprint: str,
        vector_schema_version: str,
        rows: list[dict[str, Any]],
    ) -> None:
        """Insert or update vector rows (idempotent per row identity)."""
        ...

    def read_rows_for_hash(
        self,
        *,
        content_hash: str,
        strategy_fingerprint: str,
        embedding_fingerprint: str,
        vector_schema_version: str,
    ) -> list[dict[str, Any]]:
        """Load stored rows for one content hash (retrieval assembly)."""
        ...

    def row_bundle_stat(
        self,
        *,
        content_hash: str,
        strategy_fingerprint: str,
        embedding_fingerprint: str,
        vector_schema_version: str,
    ) -> tuple[float, int] | None:
        """``(mtime, size)`` of the on-disk row bundle for this hash, or ``None`` if missing."""
        ...

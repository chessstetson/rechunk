"""
Corpus manager: pluggable enumeration of the active corpus for retrieval.

Retrieval and ``rag_index`` depend only on :meth:`CorpusManager.list_active_content_refs`.
Path/manifest specifics stay inside concrete implementations.

Legacy Temporal chunking may use :meth:`CorpusManager.temporal_ingest_hints` when the
backend can provide ``docs_root`` + ``doc_ids``; hash-only manifest backends return ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from rechunk.corpus import ContentRef, scan_filesystem_corpus
from rechunk.extracted_content import FilesystemExtractedContentService
from rechunk.hash_manifest import load_content_refs_from_manifest


@dataclass(frozen=True, slots=True)
class TemporalIngestHints:
    """
    Legacy workflow inputs: filesystem root and relative document ids.

    Not part of the content-addressed retrieval boundary; used only to start
    ``StrategyChunkingWorkflow`` until ingest moves to abstract handles.
    """

    docs_root: Path
    doc_ids: list[str]


@runtime_checkable
class CorpusManager(Protocol):
    """Active corpus as :class:`ContentRef` list; optional Temporal path hints."""

    def list_active_content_refs(self) -> list[ContentRef]:
        """Return deduplicated active corpus members (order stable for logging)."""
        ...

    def temporal_ingest_hints(self) -> TemporalIngestHints | None:
        """Return filesystem hints for legacy chunking workflows, or ``None`` if unavailable."""
        ...

    def summary_message(self, content_count: int) -> str:
        """One-line CLI summary after ``list_active_content_refs`` (``content_count`` is ``len(refs)``)."""
        ...


class FilesystemCorpusManager:
    """
    Scan a file or directory tree (same rules as :func:`rechunk.corpus.scan_filesystem_corpus`).
    Paths are used only inside this class.
    """

    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        self._refs: list[ContentRef] | None = None
        self._doc_ids: list[str] | None = None

    def _ensure_loaded(self) -> None:
        if self._refs is None:
            self._refs, self._doc_ids = scan_filesystem_corpus(self._path)

    def list_active_content_refs(self) -> list[ContentRef]:
        self._ensure_loaded()
        assert self._refs is not None
        return list(self._refs)

    def temporal_ingest_hints(self) -> TemporalIngestHints | None:
        self._ensure_loaded()
        assert self._doc_ids is not None
        root = self._path.parent if self._path.is_file() else self._path
        return TemporalIngestHints(docs_root=root, doc_ids=list(self._doc_ids))

    def summary_message(self, content_count: int) -> str:
        return (
            f"Corpus scan: {content_count} unique content object(s) by hash from {self._path} "
            "(full text not kept in memory for Q&A)."
        )


class HashManifestCorpusManager:
    """Load active corpus from a hash-only JSON manifest (no paths on wire)."""

    def __init__(self, manifest_path: Path) -> None:
        self._manifest_path = manifest_path.resolve()
        self._refs: list[ContentRef] | None = None

    def _ensure_loaded(self) -> None:
        if self._refs is None:
            self._refs = load_content_refs_from_manifest(self._manifest_path)

    def list_active_content_refs(self) -> list[ContentRef]:
        self._ensure_loaded()
        assert self._refs is not None
        return list(self._refs)

    def temporal_ingest_hints(self) -> TemporalIngestHints | None:
        return None

    def summary_message(self, content_count: int) -> str:
        return (
            f"Manifest (hash-only): {content_count} content hash(es) from {self._manifest_path} "
            "(no filesystem corpus scan for Q&A)."
        )


class EcsActiveCorpusManager:
    """
    Active corpus = distinct content hashes in :class:`FilesystemExtractedContentService` (``active_logical``).

    No filesystem scan and no re-ingest from the CLI; use after Temporal or another process populated ECS.
    """

    def __init__(self, ecs: FilesystemExtractedContentService | None = None) -> None:
        self._ecs = ecs if ecs is not None else FilesystemExtractedContentService()

    def list_active_content_refs(self) -> list[ContentRef]:
        return self._ecs.list_active_content_refs()

    def temporal_ingest_hints(self) -> TemporalIngestHints | None:
        return None

    def summary_message(self, content_count: int) -> str:
        return (
            f"ECS active corpus: {content_count} distinct content hash(es) "
            "(no docs path; ingest separately via Temporal or sync)."
        )

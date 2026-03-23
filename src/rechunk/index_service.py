"""
Top-level orchestration for ECS ingest + vectorization planning (v12 Phase C).

Query/retrieval assembly stays in existing CLI until a later phase; this module
focuses on ingest, chunk planning, and Temporal dispatch.
"""

from __future__ import annotations

import os
from pathlib import Path

from rechunk.active_corpus_manifest import write_active_manifest_exact
from rechunk.chunker import Chunker, VectorizationWorkItem
from rechunk.extracted_content.models import SourceDocumentRef
from rechunk.extracted_content.protocol import ExtractedContentService
from rechunk.fingerprints import compute_embedding_fingerprint
from rechunk.strategies import Strategy, load_strategies
from rechunk.vector_store.protocol import VectorStore
from rechunk.vectorization_config import (
    EMBEDDER_KIND_OPENAI,
    OPENAI_EMBEDDING_MODEL,
    VECTOR_SCHEMA_VERSION,
)


class IndexService:
    def __init__(
        self,
        *,
        ecs: ExtractedContentService,
        vector_store: VectorStore,
        strategies_path: Path | None = None,
    ) -> None:
        self._ecs = ecs
        self._vs = vector_store
        self._strategies_path = strategies_path
        self._chunker = Chunker(ecs, vector_store)

    def embedding_fingerprint(self) -> str:
        return compute_embedding_fingerprint(
            embedder_kind=EMBEDDER_KIND_OPENAI,
            model=os.environ.get("RECHUNK_OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL),
        )

    def ingest_filesystem_docs(self, docs_root: Path, doc_ids: list[str], *, source_kind: str = "filesystem") -> None:
        """Register each file under ``docs_root`` in ECS (active membership follows logical_doc_id)."""
        root = docs_root.resolve()
        for rel in doc_ids:
            path = root / rel
            if not path.is_file():
                continue
            ref = SourceDocumentRef(
                logical_doc_id=rel.replace("\\", "/"),
                source_kind=source_kind,
                path=path,
            )
            self._ecs.ensure_content(ref)

    def sync_active_manifest_file(self) -> list[str]:
        """Write ``corpus_content_hashes.json`` to match current ECS active set."""
        return write_active_manifest_exact(self._ecs.list_active_hashes())

    def load_strategies_from_file(self) -> list[Strategy]:
        if self._strategies_path is None or not self._strategies_path.exists():
            return []
        loaded = load_strategies(self._strategies_path)
        return loaded or []

    def list_pending_vectorization(
        self,
        strategies: list[Strategy],
    ) -> list[VectorizationWorkItem]:
        return self._chunker.list_pending(
            strategies,
            embedding_fingerprint=self.embedding_fingerprint(),
            vector_schema_version=VECTOR_SCHEMA_VERSION,
        )


def build_strategy_from_cli(
    *,
    strategy_id: str,
    kind: str,
    instruction: str,
    splitter: str = "sentence",
    model: str | None = None,
) -> Strategy:
    """Construct a :class:`Strategy` from CLI flags (``kind`` is ``llm`` or ``builtin_splitter``)."""
    if kind == "builtin_splitter":
        return Strategy(
            id=strategy_id,
            kind="builtin_splitter",
            instruction=instruction,
            splitter=splitter,
        )
    return Strategy(
        id=strategy_id,
        kind="llm",
        instruction=instruction,
        model=model,
    )


def resolve_strategy_for_job(
    *,
    strategies_path: Path,
    strategy_id: str,
    cli_strategy: Strategy,
) -> Strategy:
    """
    Prefer a strategy definition from ``rechunk_strategies.json`` when present;
    otherwise use ``cli_strategy`` (built from argparse).
    """
    strategies = load_strategies(strategies_path)
    if strategies:
        for s in strategies:
            if s.id == strategy_id:
                return s
    return cli_strategy

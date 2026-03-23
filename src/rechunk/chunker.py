"""
Diff active ECS hashes against vectorized rows and plan per-document work (v12 Phase C).
"""

from __future__ import annotations

from dataclasses import dataclass

from rechunk.extracted_content.protocol import ExtractedContentService
from rechunk.fingerprints import compute_strategy_fingerprint
from rechunk.strategies import Strategy, strategy_to_dict
from rechunk.vector_store.protocol import VectorStore
from rechunk.vectorization_config import VECTOR_SCHEMA_VERSION


@dataclass(frozen=True)
class VectorizationWorkItem:
    """One unit of work: vectorize ``content_hash`` with ``strategy``."""

    content_hash: str
    strategy: Strategy


class Chunker:
    """Plans vectorization work; does not start workflows."""

    def __init__(self, ecs: ExtractedContentService, vector_store: VectorStore) -> None:
        self._ecs = ecs
        self._vs = vector_store

    def list_pending(
        self,
        strategies: list[Strategy],
        *,
        embedding_fingerprint: str,
        vector_schema_version: str = VECTOR_SCHEMA_VERSION,
    ) -> list[VectorizationWorkItem]:
        """
        For each strategy, find active content hashes that have no rows yet for
        (strategy_fingerprint, embedding_fingerprint, schema).
        """
        active = self._ecs.list_active_hashes()
        active_set = set(active)
        pending: list[VectorizationWorkItem] = []

        for strategy in strategies:
            sfp = compute_strategy_fingerprint(strategy_to_dict(strategy))
            done = set(
                self._vs.list_vectorized_hashes(
                    strategy_fingerprint=sfp,
                    embedding_fingerprint=embedding_fingerprint,
                    vector_schema_version=vector_schema_version,
                )
            )
            for h in sorted(active_set):
                if h not in done:
                    pending.append(VectorizationWorkItem(content_hash=h, strategy=strategy))

        return pending

"""
Mtime snapshots for VectorStore row bundles (ECS-first Q&A path).

Used by ``run_with_docs`` interactive reload instead of JSONL chunk-cache mtimes.
"""

from __future__ import annotations

from collections.abc import Sequence

from rechunk.corpus import ContentRef
from rechunk.fingerprints import compute_strategy_fingerprint
from rechunk.strategies import Strategy, strategy_to_dict
from rechunk.vector_store.protocol import VectorStore


def get_vector_store_strategy_mtimes(
    vector_store: VectorStore,
    strategies: list[Strategy],
    content_refs: Sequence[ContentRef],
    *,
    embedding_fingerprint: str,
    vector_schema_version: str,
) -> dict[str, float]:
    """
    For each strategy id, the max ``st_mtime`` among row-bundle JSON files for the
    given content hashes (0.0 if none exist).
    """
    hashes = [ref.content_hash.lower() for ref in content_refs]
    out: dict[str, float] = {}
    for s in strategies:
        sfp = compute_strategy_fingerprint(strategy_to_dict(s))
        m = 0.0
        for h in hashes:
            stat = vector_store.row_bundle_stat(
                content_hash=h,
                strategy_fingerprint=sfp,
                embedding_fingerprint=embedding_fingerprint,
                vector_schema_version=vector_schema_version,
            )
            if stat is not None:
                m = max(m, stat[0])
        out[s.id] = m
    return out


def vector_store_cache_updated_since(
    vector_store: VectorStore,
    strategies: list[Strategy],
    content_refs: Sequence[ContentRef],
    last_known: dict[str, float],
    *,
    embedding_fingerprint: str,
    vector_schema_version: str,
) -> bool:
    current = get_vector_store_strategy_mtimes(
        vector_store,
        strategies,
        content_refs,
        embedding_fingerprint=embedding_fingerprint,
        vector_schema_version=vector_schema_version,
    )
    for s in strategies:
        if current.get(s.id, 0.0) > last_known.get(s.id, 0.0):
            return True
    return False

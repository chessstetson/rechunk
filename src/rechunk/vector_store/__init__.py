"""
:class:`VectorStore` protocol — vector rows + corpus-scoped collections (v12).

Phase A: contract only. Implementations (e.g. filesystem dev adapter) come in later phases.

**Rows** (reusable across corpus states): embedding + character spans + metadata,
keyed logically by ``(content_hash, strategy_fingerprint, embedding_fingerprint, vector_schema_version)``.

**Collections**: materialized retrieval index for one corpus view
(``corpus_snapshot_id``) and one retrieval configuration.
"""

from rechunk.vector_store.filesystem import FilesystemVectorStore
from rechunk.vector_store.protocol import VectorStore

__all__ = ["FilesystemVectorStore", "VectorStore"]

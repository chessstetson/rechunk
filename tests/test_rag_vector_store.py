"""Assemble VectorStoreIndex from VectorStore row bundles (ECS-first path)."""

from __future__ import annotations

from llama_index.core.embeddings import MockEmbedding

from rechunk.corpus import ContentRef
from rechunk.fingerprints import compute_embedding_fingerprint, compute_strategy_fingerprint
from rechunk.rag_index import build_vector_index_from_vector_store
from rechunk.retrieval import retrieve_top_k
from rechunk.strategies import Strategy, strategy_to_dict
from rechunk.vector_store import FilesystemVectorStore
from rechunk.vectorization_config import EMBEDDER_KIND_OPENAI, OPENAI_EMBEDDING_MODEL, VECTOR_SCHEMA_VERSION


def test_build_index_from_vector_rows_roundtrip(tmp_path) -> None:
    emb = MockEmbedding(embed_dim=8)
    vs = FilesystemVectorStore(tmp_path / "vs")
    strategy = Strategy(id="s_test", kind="builtin_splitter", instruction="x", splitter="sentence")
    sfp = compute_strategy_fingerprint(strategy_to_dict(strategy))
    efp = compute_embedding_fingerprint(embedder_kind=EMBEDDER_KIND_OPENAI, model=OPENAI_EMBEDDING_MODEL)
    h = "a" * 64
    rows = [
        {
            "content_hash": h,
            "embedding": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "metadata": {
                "strategy": "s_test",
                "source_doc": "doc1.txt",
                "source_spans": [{"start_char": 0, "end_char": 5}],
            },
            "chunk_text": "hello",
        },
        {
            "content_hash": h,
            "embedding": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "metadata": {
                "strategy": "s_test",
                "source_doc": "doc1.txt",
                "source_spans": [{"start_char": 6, "end_char": 11}],
            },
            "chunk_text": "world",
        },
    ]
    vs.upsert_rows(
        strategy_fingerprint=sfp,
        embedding_fingerprint=efp,
        vector_schema_version=VECTOR_SCHEMA_VERSION,
        rows=rows,
    )

    ref = ContentRef(content_hash=h, source_hint="doc1.txt")
    index, nodes = build_vector_index_from_vector_store(
        vs,
        [strategy],
        [ref],
        embedding_fingerprint=efp,
        vector_schema_version=VECTOR_SCHEMA_VERSION,
        quiet=True,
        embed_model=emb,
    )
    assert len(nodes) == 2
    hits = retrieve_top_k(index, "hello", top_k=2)
    assert len(hits) >= 1

"""Phase C: vectorize_content_for_strategy activity + manifest exact write."""

from __future__ import annotations

import pytest
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from temporalio.testing import ActivityEnvironment

from rechunk.extracted_content import FilesystemExtractedContentService, SourceDocumentRef
from rechunk.fingerprints import compute_embedding_fingerprint, compute_strategy_fingerprint
from rechunk.strategies import Strategy, strategy_to_dict
from rechunk.vector_store import FilesystemVectorStore
from rechunk.vectorization_config import (
    EMBEDDER_KIND_OPENAI,
    OPENAI_EMBEDDING_MODEL,
    VECTOR_SCHEMA_VERSION,
)
from rechunk.worker_runtime import configure_worker_runtime, reset_worker_runtime
from temporal_activities import vectorize_content_for_strategy
from temporal_vectorization_inputs import DocumentVectorizationInput


@pytest.mark.asyncio
async def test_vectorize_builtin_writes_vector_rows_and_jsonl_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RECHUNK_STRATEGY_CACHE_DIR", str(tmp_path / "strat_cache"))
    monkeypatch.setenv("RECHUNK_ECS_ROOT", str(tmp_path / "ecs"))
    monkeypatch.setenv("RECHUNK_VECTOR_STORE_DEV_ROOT", str(tmp_path / "vs"))

    emb = MockEmbedding(embed_dim=8)
    Settings.embed_model = emb
    ecs = FilesystemExtractedContentService(tmp_path / "ecs")
    vs = FilesystemVectorStore(tmp_path / "vs", embed_model=emb)
    configure_worker_runtime(ecs, vs)
    try:
        f = tmp_path / "doc.txt"
        f.write_text("First sentence here. Second sentence there.", encoding="utf-8")
        ec = ecs.ensure_content(
            SourceDocumentRef(logical_doc_id="doc.txt", source_kind="fs", path=f),
        )
        s = Strategy(
            id="s_phase_c",
            kind="builtin_splitter",
            instruction="builtin",
            splitter="sentence",
        )
        sd = strategy_to_dict(s)
        sfp = compute_strategy_fingerprint(sd)
        efp = compute_embedding_fingerprint(embedder_kind=EMBEDDER_KIND_OPENAI, model=OPENAI_EMBEDDING_MODEL)
        inp = DocumentVectorizationInput(
            content_hash=ec.content_hash,
            strategy_id=s.id,
            strategy_definition=sd,
            strategy_fingerprint=sfp,
            embedding_fingerprint=efp,
            vector_schema_version=VECTOR_SCHEMA_VERSION,
        )
        env = ActivityEnvironment()
        result = await env.run(vectorize_content_for_strategy, inp)
        assert result["status"] == "processed"
        assert result["rows"] >= 1

        rows = vs.read_rows_for_hash(
            content_hash=ec.content_hash,
            strategy_fingerprint=sfp,
            embedding_fingerprint=efp,
            vector_schema_version=VECTOR_SCHEMA_VERSION,
        )
        assert len(rows) >= 1
        assert all(len(r["embedding"]) == 8 for r in rows)

        result2 = await env.run(vectorize_content_for_strategy, inp)
        assert result2["status"] == "skipped"

        from rechunk.cache import load_chunk_cache

        cached = load_chunk_cache("s_phase_c")
        assert ec.content_hash in cached
    finally:
        reset_worker_runtime()


def test_write_active_manifest_exact_replaces(tmp_path, monkeypatch) -> None:
    import json

    from rechunk.active_corpus_manifest import write_active_manifest_exact

    p = tmp_path / "m.json"
    monkeypatch.setenv("RECHUNK_ACTIVE_CORPUS_MANIFEST", str(p))
    old_hash = "aa" * 32
    p.write_text(json.dumps([old_hash]) + "\n", encoding="utf-8")
    h1 = "bb" * 32
    h2 = "cc" * 32
    out = write_active_manifest_exact([h1, h2])
    assert out == sorted([h1.lower(), h2.lower()])
    data = __import__("json").loads(p.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert old_hash not in data

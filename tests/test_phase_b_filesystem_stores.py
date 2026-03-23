"""Phase B (v12): filesystem ExtractedContentService + VectorStore dev adapters."""

from __future__ import annotations

import pytest

from rechunk.cache import compute_content_hash
from rechunk.corpus_snapshot_id import compute_corpus_snapshot_id
from rechunk.extracted_content import (
    ExtractedContentService,
    FilesystemExtractedContentService,
    SourceDocumentRef,
)
from rechunk.fingerprints import compute_embedding_fingerprint, compute_strategy_fingerprint
from rechunk.vector_store import FilesystemVectorStore, VectorStore


@pytest.fixture
def ecs(tmp_path) -> FilesystemExtractedContentService:
    return FilesystemExtractedContentService(tmp_path / "ecs")


@pytest.fixture
def vs(tmp_path) -> FilesystemVectorStore:
    return FilesystemVectorStore(tmp_path / "vs")


def test_ecs_ensure_from_path_roundtrip(ecs: FilesystemExtractedContentService, tmp_path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello phase b\n", encoding="utf-8")
    ref = SourceDocumentRef(
        logical_doc_id="doc-a",
        source_kind="filesystem",
        path=f,
    )
    ec = ecs.ensure_content(ref)
    assert ec.content_hash == compute_content_hash("hello phase b")
    assert ecs.list_active_hashes() == [ec.content_hash]
    got = ecs.get_content(ec.content_hash)
    assert got.extracted_text == "hello phase b"
    assert ecs.has_content(ec.content_hash)


def test_ecs_idempotent_re_ensure(ecs: FilesystemExtractedContentService, tmp_path) -> None:
    f = tmp_path / "b.txt"
    f.write_text("same", encoding="utf-8")
    ref = SourceDocumentRef(logical_doc_id="b", source_kind="fs", path=f)
    h1 = ecs.ensure_content(ref).content_hash
    h2 = ecs.ensure_content(ref).content_hash
    assert h1 == h2
    assert ecs.list_active_hashes() == [h1]


def test_ecs_logical_replacement_deactivates_old_hash(ecs: FilesystemExtractedContentService, tmp_path) -> None:
    f = tmp_path / "c.txt"
    f.write_text("v1", encoding="utf-8")
    ecs.ensure_content(SourceDocumentRef(logical_doc_id="c", source_kind="fs", path=f))
    h1 = compute_content_hash("v1")
    assert ecs.list_active_hashes() == [h1]

    f.write_text("v2", encoding="utf-8")
    h2 = compute_content_hash("v2")
    ecs.ensure_content(SourceDocumentRef(logical_doc_id="c", source_kind="fs", path=f))
    assert ecs.list_active_hashes() == [h2]
    assert ecs.has_content(h1)
    assert ecs.get_content(h1).extracted_text == "v1"


def test_ecs_deactivate_logical_doc(ecs: FilesystemExtractedContentService, tmp_path) -> None:
    f = tmp_path / "d.txt"
    f.write_text("d", encoding="utf-8")
    ecs.ensure_content(SourceDocumentRef(logical_doc_id="d", source_kind="fs", path=f))
    assert len(ecs.list_active_hashes()) == 1
    ecs.deactivate_logical_doc("d", reason="test")
    assert ecs.list_active_hashes() == []


def test_ecs_apply_source_inventory(ecs: FilesystemExtractedContentService, tmp_path) -> None:
    for name, lid in [("e1.txt", "e1"), ("e2.txt", "e2")]:
        p = tmp_path / name
        p.write_text(lid, encoding="utf-8")
        ecs.ensure_content(
            SourceDocumentRef(logical_doc_id=lid, source_kind="sharepoint", path=p)
        )
    assert len(ecs.list_active_hashes()) == 2
    ecs.apply_source_inventory("sharepoint", observed_logical_doc_ids=["e1"])
    active = ecs.list_active_hashes()
    assert len(active) == 1
    assert ecs.get_content(active[0]).logical_doc_id == "e1"


def test_ecs_raw_bytes_text(ecs: FilesystemExtractedContentService) -> None:
    ref = SourceDocumentRef(
        logical_doc_id="rb",
        source_kind="api",
        raw_bytes=b"plain bytes text",
        mime_type="text/plain",
    )
    ec = ecs.ensure_content(ref)
    assert ec.extracted_text == "plain bytes text"


def test_ecs_raw_bytes_binary_without_path_errors(ecs: FilesystemExtractedContentService) -> None:
    ref = SourceDocumentRef(
        logical_doc_id="x",
        source_kind="api",
        raw_bytes=b"%PDF-1.4",
        mime_type="application/pdf",
    )
    with pytest.raises(ValueError, match="raw_bytes"):
        ecs.ensure_content(ref)


def test_vector_store_upsert_and_list(vs: FilesystemVectorStore) -> None:
    sfp = compute_strategy_fingerprint({"id": "s1", "kind": "builtin"})
    efp = compute_embedding_fingerprint(embedder_kind="mock", model="m1")
    schema = "v1"
    h = "a" * 64
    vs.upsert_rows(
        strategy_fingerprint=sfp,
        embedding_fingerprint=efp,
        vector_schema_version=schema,
        rows=[
            {
                "content_hash": h,
                "span_start": 0,
                "span_end": 5,
                "embedding": [0.1, 0.2],
                "metadata": {"k": 1},
            }
        ],
    )
    assert vs.list_vectorized_hashes(
        strategy_fingerprint=sfp,
        embedding_fingerprint=efp,
        vector_schema_version=schema,
    ) == [h]
    rows = vs.read_rows_for_hash(
        content_hash=h,
        strategy_fingerprint=sfp,
        embedding_fingerprint=efp,
        vector_schema_version=schema,
    )
    assert len(rows) == 1
    assert rows[0]["span_end"] == 5


def test_vector_store_upsert_replaces_same_span(vs: FilesystemVectorStore) -> None:
    sfp = compute_strategy_fingerprint({"id": "s2"})
    efp = compute_embedding_fingerprint(embedder_kind="mock", model="m1")
    h = "b" * 64
    base = dict(
        strategy_fingerprint=sfp,
        embedding_fingerprint=efp,
        vector_schema_version="v1",
    )
    vs.upsert_rows(
        rows=[{"content_hash": h, "span_start": 0, "span_end": 3, "embedding": [1.0]}],
        **base,
    )
    vs.upsert_rows(
        rows=[{"content_hash": h, "span_start": 0, "span_end": 3, "embedding": [2.0]}],
        **base,
    )
    rows = vs.read_rows_for_hash(content_hash=h, **base)
    assert rows[0]["embedding"] == [2.0]


def test_vector_store_put_get_collection_roundtrip(vs: FilesystemVectorStore, tmp_path) -> None:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.schema import TextNode

    emb = MockEmbedding(embed_dim=4)
    vs_emb = FilesystemVectorStore(vs._root, embed_model=emb)
    nodes = [TextNode(text="hello", id_="1")]
    idx = VectorStoreIndex(nodes, embed_model=emb)
    cid = compute_corpus_snapshot_id(["deadbeef" * 8])
    sfp = compute_strategy_fingerprint({"id": "s"})
    efp = compute_embedding_fingerprint(embedder_kind="mock", model="m")
    vs_emb.put_collection(
        corpus_snapshot_id=cid,
        strategy_fingerprint=sfp,
        embedding_fingerprint=efp,
        vector_schema_version="v1",
        index_obj=idx,
        metadata={"n": 1},
    )
    loaded = vs_emb.get_collection(
        corpus_snapshot_id=cid,
        strategy_fingerprint=sfp,
        embedding_fingerprint=efp,
        vector_schema_version="v1",
    )
    assert loaded is not None
    assert len(loaded.docstore.docs) >= 1


def test_vector_store_get_collection_without_embed_returns_path(vs: FilesystemVectorStore, tmp_path) -> None:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.schema import TextNode

    emb = MockEmbedding(embed_dim=2)
    nodes = [TextNode(text="x", id_="1")]
    idx = VectorStoreIndex(nodes, embed_model=emb)
    cid = compute_corpus_snapshot_id([])
    sfp = "s" * 64
    efp = "e" * 64
    vs.put_collection(
        corpus_snapshot_id=cid,
        strategy_fingerprint=sfp,
        embedding_fingerprint=efp,
        vector_schema_version="v1",
        index_obj=idx,
    )
    out = vs.get_collection(
        corpus_snapshot_id=cid,
        strategy_fingerprint=sfp,
        embedding_fingerprint=efp,
        vector_schema_version="v1",
    )
    assert out is not None
    from pathlib import Path

    assert isinstance(out, Path)


def test_isinstance_protocols(ecs: FilesystemExtractedContentService, vs: FilesystemVectorStore) -> None:
    assert isinstance(ecs, ExtractedContentService)
    assert isinstance(vs, VectorStore)

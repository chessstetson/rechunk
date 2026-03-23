"""Phase A (v12): corpus_snapshot_id, fingerprints, models — no service impl yet."""

from __future__ import annotations

import hashlib

import pytest

from rechunk.corpus_snapshot_id import compute_corpus_snapshot_id
from rechunk.extracted_content import ExtractedContent, SourceDocumentRef
from rechunk.extracted_content.protocol import ExtractedContentService
from rechunk.fingerprints import compute_embedding_fingerprint, compute_strategy_fingerprint
from rechunk.vector_store import VectorStore


def test_compute_corpus_snapshot_id_empty() -> None:
    expected = hashlib.sha256(b"").hexdigest()
    assert compute_corpus_snapshot_id([]) == expected


def test_compute_corpus_snapshot_id_order_invariant() -> None:
    a = "aa" * 32
    b = "bb" * 32
    assert compute_corpus_snapshot_id([a, b]) == compute_corpus_snapshot_id([b, a])


def test_compute_corpus_snapshot_id_case_normalized() -> None:
    h = "Ab" * 32
    assert compute_corpus_snapshot_id([h.upper()]) == compute_corpus_snapshot_id([h.lower()])


def test_compute_corpus_snapshot_id_strips_whitespace() -> None:
    h = "a" * 64
    assert compute_corpus_snapshot_id([f"  {h}  "]) == compute_corpus_snapshot_id([h])


def test_compute_strategy_fingerprint_key_order_invariant() -> None:
    fp1 = compute_strategy_fingerprint({"z": 1, "a": 2})
    fp2 = compute_strategy_fingerprint({"a": 2, "z": 1})
    assert fp1 == fp2


def test_compute_strategy_fingerprint_value_change() -> None:
    assert compute_strategy_fingerprint({"id": "s1", "kind": "llm"}) != compute_strategy_fingerprint(
        {"id": "s1", "kind": "builtin"}
    )


def test_builtin_chunking_cli_fallback_matches_default_baseline_strategy_fingerprint() -> None:
    """start_strategy_chunking (no JSON file) must match run_with_docs default baseline rows path."""
    from rechunk.strategies import DEFAULT_BASELINE_STRATEGY, Strategy, strategy_to_dict

    cli_builtin = Strategy(
        id="s_default",
        kind="builtin_splitter",
        instruction=DEFAULT_BASELINE_STRATEGY.instruction,
        splitter="sentence",
    )
    assert compute_strategy_fingerprint(strategy_to_dict(cli_builtin)) == compute_strategy_fingerprint(
        strategy_to_dict(DEFAULT_BASELINE_STRATEGY)
    )
    # Historical bug: old fallback used instruction "builtin splitter", which changed the fingerprint.
    assert compute_strategy_fingerprint(strategy_to_dict(cli_builtin)) != compute_strategy_fingerprint(
        strategy_to_dict(
            Strategy(
                id="s_default",
                kind="builtin_splitter",
                instruction="builtin splitter",
                splitter="sentence",
            )
        )
    )


def test_compute_embedding_fingerprint_stable() -> None:
    fp = compute_embedding_fingerprint(embedder_kind="openai", model="text-embedding-3-small")
    assert len(fp) == 64
    assert fp == compute_embedding_fingerprint(embedder_kind="openai", model="text-embedding-3-small")


def test_compute_embedding_fingerprint_extra_changes() -> None:
    base = compute_embedding_fingerprint(embedder_kind="openai", model="text-embedding-3-small")
    with_extra = compute_embedding_fingerprint(
        embedder_kind="openai",
        model="text-embedding-3-small",
        extra={"dims": 1536},
    )
    assert base != with_extra


def test_extracted_content_frozen() -> None:
    ec = ExtractedContent(
        content_hash="a" * 64,
        logical_doc_id="doc1",
        mime_type="text/plain",
        extracted_text="hi",
    )
    with pytest.raises(Exception):  # FrozenInstanceError on 3.11+
        ec.content_hash = "b" * 64  # type: ignore[misc]


def test_source_document_ref_defaults() -> None:
    ref = SourceDocumentRef(logical_doc_id="x", source_kind="filesystem")
    assert ref.path is None
    assert ref.metadata == {}


class _DummyECS:
    """Minimal object satisfying ExtractedContentService for isinstance checks."""

    def ensure_content(self, source_doc_ref):
        raise NotImplementedError

    def deactivate_logical_doc(self, logical_doc_id: str, *, reason=None):
        raise NotImplementedError

    def apply_source_inventory(self, source_kind: str, observed_logical_doc_ids: list[str]):
        raise NotImplementedError

    def get_content(self, content_hash: str):
        raise NotImplementedError

    def has_content(self, content_hash: str):
        raise NotImplementedError

    def list_active_hashes(self) -> list[str]:
        return []


class _DummyVS:
    def get_collection(self, **kwargs):
        return None

    def put_collection(self, **kwargs):
        raise NotImplementedError

    def list_vectorized_hashes(self, **kwargs):
        return []

    def upsert_rows(self, **kwargs):
        raise NotImplementedError

    def read_rows_for_hash(self, **kwargs):
        return []

    def row_bundle_stat(self, **kwargs):
        return None


def test_extracted_content_service_protocol_runtime_check() -> None:
    assert isinstance(_DummyECS(), ExtractedContentService)


def test_vector_store_protocol_runtime_check() -> None:
    assert isinstance(_DummyVS(), VectorStore)

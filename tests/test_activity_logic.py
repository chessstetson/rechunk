"""
Cache + ingest snapshot checks and isolated chunk-to-cache pipeline tests (no legacy Temporal activities).

Older workflow-level tests targeted ``StrategyChunkingWorkflow`` and file-path chunk activities; vectorization
is covered by Phase C tests (``vectorize_content_for_strategy`` / batch workflow).
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode

from rechunk.cache import (
    append_chunk_cache,
    compute_content_hash,
    load_chunk_cache,
)


def test_ingest_snapshot_build_and_read_roundtrip(tmp_path):
    """Snapshot written at trigger time matches disk when read in activity path."""
    (tmp_path / "a.txt").write_text("hello snapshot", encoding="utf-8")
    os.environ["RECHUNK_INGEST_SNAPSHOT_DIR"] = str(tmp_path / "snap_out")
    try:
        from rechunk.ingest_snapshot import build_and_write_ingest_snapshot, read_ingest_snapshot

        snap = build_and_write_ingest_snapshot(tmp_path, ["a.txt"], strategy_id="t_roundtrip")
        root, manifest = read_ingest_snapshot(snap)
        assert root == tmp_path.resolve()
        assert len(manifest) == 1
        assert manifest[0]["doc_id"] == "a.txt"
        assert manifest[0]["content_hash"] == compute_content_hash("hello snapshot")
    finally:
        os.environ.pop("RECHUNK_INGEST_SNAPSHOT_DIR", None)


def test_compute_content_hash():
    h = compute_content_hash("hello world")
    assert isinstance(h, str)
    assert len(h) == 64
    assert h == compute_content_hash("hello world")
    assert h != compute_content_hash("hello world.")


def test_cache_round_trip(tmp_path):
    """Append nodes to cache, load back, assert they match."""
    os.environ["RECHUNK_STRATEGY_CACHE_DIR"] = str(tmp_path)
    try:
        strategy_id = "test_strategy_round_trip"
        content_hash = "abc123"
        nodes = [
            TextNode(
                id_="n1",
                text="chunk one",
                metadata={"strategy": strategy_id, "source_doc": "doc1"},
                ref_doc_id="doc1",
            ),
            TextNode(
                id_="n2",
                text="chunk two",
                metadata={"strategy": strategy_id, "source_doc": "doc1"},
                ref_doc_id="doc1",
            ),
        ]
        append_chunk_cache(strategy_id, content_hash, nodes)
        loaded = load_chunk_cache(strategy_id)
        assert content_hash in loaded
        got = loaded[content_hash]
        assert len(got) == 2
        assert got[0].text == "chunk one"
        assert got[0].metadata.get("strategy") == strategy_id
        assert got[0].metadata.get("source_doc") == "doc1"
        assert got[1].text == "chunk two"
    finally:
        os.environ.pop("RECHUNK_STRATEGY_CACHE_DIR", None)


def test_activity_logic_in_isolation(tmp_path):
    """
    Run the exact pipeline vectorization uses for LLM: Document → parse (mocked) → append cache → load.
    Proves the code path works without Temporal.
    """
    os.environ["RECHUNK_STRATEGY_CACHE_DIR"] = str(tmp_path)
    try:
        doc_content = "First paragraph.\n\nSecond paragraph."
        doc_file = tmp_path / "doc1.txt"
        doc_file.write_text(doc_content, encoding="utf-8")
        doc_id = "doc1.txt"
        strategy_id = "test_activity_logic"
        content_hash = compute_content_hash(doc_content)

        text = doc_file.read_text(encoding="utf-8")
        doc = Document(text=text, id_=doc_id)

        mock_nodes = [
            TextNode(
                id_="doc1_chunk_1",
                text="First paragraph.",
                metadata={"strategy": strategy_id, "source_doc": doc_id},
                ref_doc_id=doc_id,
            ),
            TextNode(
                id_="doc1_chunk_2",
                text="Second paragraph.",
                metadata={"strategy": strategy_id, "source_doc": doc_id},
                ref_doc_id=doc_id,
            ),
        ]

        with patch("rechunk.node_parser.LLMNodeParser.get_nodes_from_documents", return_value=mock_nodes):
            from rechunk import LLMNodeParser

            parser = LLMNodeParser(
                strategy_id=strategy_id,
                strategy_instruction="Split by paragraph.",
            )
            nodes = parser.get_nodes_from_documents([doc])

        assert len(nodes) == 2
        append_chunk_cache(strategy_id, content_hash, nodes)
        loaded = load_chunk_cache(strategy_id)
        assert content_hash in loaded
        got = loaded[content_hash]
        assert len(got) == 2
        assert got[0].text == "First paragraph."
        assert got[1].text == "Second paragraph."
        assert got[0].metadata.get("source_doc") == doc_id
    finally:
        os.environ.pop("RECHUNK_STRATEGY_CACHE_DIR", None)

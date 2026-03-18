"""
Step 0.1 & 0.2: Verify activity logic, cache round-trip, and activity run inside Temporal.

0.1: Code path and cache work without Temporal. 0.2: Real activity runs in Temporal's
ActivityEnvironment and writes to cache. No Temporal server or real LLM.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode
from temporalio.testing import ActivityEnvironment

from rechunk.cache import (
    append_chunk_cache,
    compute_content_hash,
    load_chunk_cache,
)


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
    Run the exact pipeline the activity will use: read doc from path → parse (mocked) → append cache → load.
    Proves the code path works before it lives inside a Temporal activity.
    """
    os.environ["RECHUNK_STRATEGY_CACHE_DIR"] = str(tmp_path)
    try:
        doc_content = "First paragraph.\n\nSecond paragraph."
        doc_file = tmp_path / "doc1.txt"
        doc_file.write_text(doc_content, encoding="utf-8")
        doc_id = "doc1.txt"
        strategy_id = "test_activity_logic"
        content_hash = compute_content_hash(doc_content)

        # What the activity will do: read text, build Document, run parser, append cache.
        text = doc_file.read_text(encoding="utf-8")
        doc = Document(text=text, id_=doc_id)

        # Mock LLMNodeParser.get_nodes_from_documents so we don't call the real LLM.
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


@pytest.mark.asyncio
async def test_activity_runs_inside_temporal(tmp_path):
    """
    Step 0.2: Run the real chunk_doc_with_strategy activity in Temporal's activity context.
    Verifies serialization, async execution, and that the activity writes to the cache.
    """
    os.environ["RECHUNK_STRATEGY_CACHE_DIR"] = str(tmp_path)
    try:
        doc_content = "Section A.\n\nSection B."
        doc_file = tmp_path / "doc.txt"
        doc_file.write_text(doc_content, encoding="utf-8")
        doc_id = "doc.txt"
        strategy_id = "test_temporal_activity"
        content_hash = compute_content_hash(doc_content)

        mock_nodes = [
            TextNode(
                id_="doc_chunk_1",
                text="Section A.",
                metadata={"strategy": strategy_id, "source_doc": doc_id},
                ref_doc_id=doc_id,
            ),
            TextNode(
                id_="doc_chunk_2",
                text="Section B.",
                metadata={"strategy": strategy_id, "source_doc": doc_id},
                ref_doc_id=doc_id,
            ),
        ]

        from temporal_activities import ChunkDocInput, chunk_doc_with_strategy

        activity_input = ChunkDocInput(
            strategy_id=strategy_id,
            strategy_instruction="Split by section.",
            model=None,
            docs_root=str(tmp_path),
            doc_id=doc_id,
            content_hash=content_hash,
        )

        with patch("rechunk.node_parser.LLMNodeParser.get_nodes_from_documents", return_value=mock_nodes):
            env = ActivityEnvironment()
            await env.run(chunk_doc_with_strategy, activity_input)

        loaded = load_chunk_cache(strategy_id)
        assert content_hash in loaded
        got = loaded[content_hash]
        assert len(got) == 2
        assert got[0].text == "Section A."
        assert got[1].text == "Section B."
        assert got[0].metadata.get("source_doc") == doc_id
    finally:
        os.environ.pop("RECHUNK_STRATEGY_CACHE_DIR", None)


@pytest.mark.asyncio
async def test_builtin_chunking_runs_inside_temporal(tmp_path):
    """
    Run the real chunk_doc_with_builtin_splitter activity in Temporal's activity context.
    Uses real LlamaIndex SentenceSplitter (no mock). Verifies baseline chunking works in Temporal.
    """
    os.environ["RECHUNK_STRATEGY_CACHE_DIR"] = str(tmp_path)
    try:
        # Content that SentenceSplitter will split into multiple chunks (multiple sentences).
        doc_content = "First sentence here. Second sentence there. Third one for good measure."
        doc_file = tmp_path / "builtin_doc.txt"
        doc_file.write_text(doc_content, encoding="utf-8")
        doc_id = "builtin_doc.txt"
        strategy_id = "test_builtin_temporal"
        content_hash = compute_content_hash(doc_content)

        from temporal_activities import BuiltinChunkInput, chunk_doc_with_builtin_splitter

        activity_input = BuiltinChunkInput(
            strategy_id=strategy_id,
            splitter="sentence",
            docs_root=str(tmp_path),
            doc_id=doc_id,
            content_hash=content_hash,
        )

        env = ActivityEnvironment()
        await env.run(chunk_doc_with_builtin_splitter, activity_input)

        loaded = load_chunk_cache(strategy_id)
        assert content_hash in loaded
        got = loaded[content_hash]
        assert len(got) >= 1
        # All chunks should be tagged with strategy and source_doc (LlamaIndex baseline path).
        for node in got:
            assert node.metadata.get("strategy") == strategy_id
            assert node.metadata.get("source_doc") == doc_id
        # Reassembled text should match original (minus possible whitespace normalization).
        reassembled = " ".join(n.text.strip() for n in got)
        assert "First sentence" in reassembled and "Third one" in reassembled
    finally:
        os.environ.pop("RECHUNK_STRATEGY_CACHE_DIR", None)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run real LLM chunking test (run manually)",
)
@pytest.mark.asyncio
async def test_real_llm_chunking_in_activity(tmp_path):
    """
    Real LLM chunking inside a Temporal activity (no mock). Requires OPENAI_API_KEY.
    Run manually: OPENAI_API_KEY=sk-... pytest tests/test_activity_logic.py::test_real_llm_chunking_in_activity -v
    """
    os.environ["RECHUNK_STRATEGY_CACHE_DIR"] = str(tmp_path)
    try:
        from llama_index.core import Settings
        from llama_index.llms.openai import OpenAI

        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

        doc_content = (
            "The company was founded in 2020. It builds developer tools. "
            "The flagship product is a chunking library for RAG."
        )
        doc_file = tmp_path / "real_llm_doc.txt"
        doc_file.write_text(doc_content, encoding="utf-8")
        doc_id = "real_llm_doc.txt"
        strategy_id = "test_real_llm"
        content_hash = compute_content_hash(doc_content)

        from temporal_activities import ChunkDocInput, chunk_doc_with_strategy

        activity_input = ChunkDocInput(
            strategy_id=strategy_id,
            strategy_instruction="Split by sentence. One chunk per sentence.",
            model=None,
            docs_root=str(tmp_path),
            doc_id=doc_id,
            content_hash=content_hash,
        )

        env = ActivityEnvironment()
        await env.run(chunk_doc_with_strategy, activity_input)

        loaded = load_chunk_cache(strategy_id)
        assert content_hash in loaded
        got = loaded[content_hash]
        assert len(got) >= 1
        for node in got:
            assert node.metadata.get("strategy") == strategy_id
            assert node.metadata.get("source_doc") == doc_id
        all_text = " ".join(n.text for n in got)
        assert "founded in 2020" in all_text or "developer tools" in all_text or "chunking" in all_text
    finally:
        os.environ.pop("RECHUNK_STRATEGY_CACHE_DIR", None)

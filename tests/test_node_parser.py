"""Tests for LLMNodeParser (v0.1)."""

import pytest
from llama_index.core import Document
from unittest.mock import MagicMock, patch

from rechunk import LLMNodeParser
from rechunk.node_parser import _extract_json_array, CHUNKING_PROMPT


def test_extract_json_array_raw():
    assert _extract_json_array('[{"a": 1}]') == [{"a": 1}]


def test_extract_json_array_markdown_fence():
    text = '```json\n[{"chunk_id": "x", "content": "hi"}]\n```'
    assert _extract_json_array(text) == [{"chunk_id": "x", "content": "hi"}]


def test_chunking_prompt_placeholders():
    assert "{strategy_instruction}" in CHUNKING_PROMPT
    assert "{strategy_id}" in CHUNKING_PROMPT
    assert "{doc_id}" in CHUNKING_PROMPT
    assert "{document_text}" in CHUNKING_PROMPT


def test_llm_node_parser_instantiation():
    parser = LLMNodeParser(
        strategy_id="s_test",
        strategy_instruction="Split by sentences.",
    )
    assert parser.strategy_id == "s_test"
    assert parser.strategy_instruction == "Split by sentences."
    assert parser.llm is None


@patch("rechunk.node_parser.LLMNodeParser._parse_nodes")
def test_get_nodes_from_documents_integration(mock_parse):
    from llama_index.core.schema import TextNode

    mock_parse.return_value = [
        TextNode(text="chunk1", metadata={"strategy": "s_test", "source_doc": "doc1"}),
    ]
    parser = LLMNodeParser(
        strategy_id="s_test",
        strategy_instruction="Split by sentences.",
    )
    parser._parse_nodes = mock_parse
    docs = [Document(text="Hello. World.", id_="doc1")]
    nodes = parser.get_nodes_from_documents(docs)
    assert len(nodes) == 1
    assert nodes[0].text == "chunk1"
    assert nodes[0].metadata.get("strategy") == "s_test"

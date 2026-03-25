"""Tests for DerivedNodeParser (mocked LLM)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llama_index.core import Document

from rechunk.node_parser import DerivedNodeParser


def _mock_llm(response: str) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = response
    return m


def test_derived_parser_mock_llm_two_nodes():
    doc_text = "abcdefghijklmnop"
    response_json = """[
      {
        "node_id": "n1",
        "content": "Summary A",
        "source_spans": [{"start_char": 0, "end_char": 4, "quote": "abcd"}],
        "metadata": {}
      },
      {
        "node_id": "n2",
        "content": "Summary B",
        "source_spans": [{"start_char": 10, "end_char": 14}],
        "metadata": {}
      }
    ]"""
    llm = _mock_llm(response_json)

    parser = DerivedNodeParser(
        strategy_id="s_derived_test",
        strategy_instruction="Summarize in two nodes.",
        llm=None,
        max_doc_chars_for_llm=50_000,
    )
    with patch("llama_index.core.Settings", SimpleNamespace(llm=llm)):
        nodes = parser.get_nodes_from_documents([Document(text=doc_text, id_="doc1.txt")])
    assert len(nodes) == 2
    assert nodes[0].text == "Summary A"
    assert nodes[0].metadata.get("derived") is True
    assert nodes[0].metadata.get("source_spans") == [{"start_char": 0, "end_char": 4, "quote": "abcd"}]
    assert nodes[1].text == "Summary B"


def test_derived_parser_dedupes_duplicate_span_key():
    doc_text = "abcdefghij"
    response_json = """[
      {"node_id": "a", "content": "First", "source_spans": [{"start_char": 0, "end_char": 3}], "metadata": {}},
      {"node_id": "b", "content": "Second", "source_spans": [{"start_char": 0, "end_char": 3}], "metadata": {}}
    ]"""
    llm = _mock_llm(response_json)

    parser = DerivedNodeParser(
        strategy_id="s_dedup",
        strategy_instruction="x",
        llm=None,
    )
    with patch("llama_index.core.Settings", SimpleNamespace(llm=llm)):
        nodes = parser.get_nodes_from_documents([Document(text=doc_text, id_="d")])
    assert len(nodes) == 1
    assert nodes[0].text == "First"

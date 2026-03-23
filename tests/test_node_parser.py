"""Tests for LLMNodeParser (v0.1) and fallbacks (length pre-check, windowed fallback)."""

import os
import pytest
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from unittest.mock import MagicMock, patch

from rechunk import LLMNodeParser
from rechunk.node_parser import (
    CHUNKING_PROMPT,
    DEFAULT_MAX_DOC_CHARS_FOR_LLM,
    FALLBACK_OVERLAP_CHARS,
    FALLBACK_WINDOW_CHARS,
    _extract_json_array,
    _extract_spans_from_llm_chunk,
    _reconstruct_content_from_spans,
    _windowed_fallback,
)
from rechunk.node_span_utils import char_spans_for_nodes
from llama_index.core.schema import TextNode


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
    assert "spans" in CHUNKING_PROMPT


def test_extract_spans_multi_region():
    doc_len = 200
    c = {
        "spans": [{"start_char": 0, "end_char": 3}, {"start_char": 10, "end_char": 12}],
    }
    spans = _extract_spans_from_llm_chunk(c, doc_len=doc_len)
    assert spans == [(0, 3), (10, 12)]
    doc = "0123456789XX"
    assert _reconstruct_content_from_spans(doc, spans) == "012XX"


def test_extract_spans_single_start_end():
    c = {"start_char": 1, "end_char": 4}
    assert _extract_spans_from_llm_chunk(c, doc_len=10) == [(1, 4)]


def test_char_spans_uses_metadata_span_ranges_bbox():
    full = "aaabbbcccddd"
    node = TextNode(
        text="aaabbb",
        metadata={"span_ranges": [[0, 3], [6, 9]]},
        start_char_idx=0,
        end_char_idx=9,
    )
    assert char_spans_for_nodes(full, [node]) == [(0, 9)]


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


def test_windowed_fallback_semi_overlapping():
    """Fallback produces overlapping windows and sets start/end char indices."""
    # 30k chars: one full window (24k) + one overlapping (6k) with step 20k
    text = "x" * 30_000
    nodes = _windowed_fallback(
        text,
        "doc1",
        "s_test",
        max_chars=FALLBACK_WINDOW_CHARS,
        overlap_chars=FALLBACK_OVERLAP_CHARS,
        id_prefix="test",
    )
    assert len(nodes) >= 2
    assert nodes[0].text == "x" * 24_000
    assert nodes[0].start_char_idx == 0
    assert nodes[0].end_char_idx == 24_000
    assert nodes[1].start_char_idx == 20_000  # step = 24k - 4k
    assert nodes[1].end_char_idx == 30_000
    assert nodes[1].metadata["strategy"] == "s_test"
    assert nodes[1].metadata["source_doc"] == "doc1"


def test_over_length_doc_uses_fallback_without_llm():
    """Doc over max_doc_chars_for_llm is windowed without calling the LLM (pre-check path)."""
    long_text = "A" * (DEFAULT_MAX_DOC_CHARS_FOR_LLM + 1)
    doc = Document(text=long_text, id_="long_doc.txt")
    parser = LLMNodeParser(
        strategy_id="s_len_test",
        strategy_instruction="Split by section.",
        llm=None,
        max_doc_chars_for_llm=DEFAULT_MAX_DOC_CHARS_FOR_LLM,
    )
    nodes = parser.get_nodes_from_documents([doc])
    # Pre-check path: we get semi-overlapping window nodes, no LLM invoked
    assert len(nodes) >= 1
    for n in nodes:
        assert n.metadata.get("strategy") == "s_len_test"
        assert n.metadata.get("source_doc") == "long_doc.txt"
    assert nodes[0].start_char_idx == 0
    assert nodes[0].end_char_idx == FALLBACK_WINDOW_CHARS
    # IDs indicate length_fallback, not error_fallback or LLM chunks
    assert "length_fallback" in nodes[0].id_


# Small doc used for LLM vs windowed comparison (fits in context; we then window it for B).
_COMPARISON_DOC = """
Our company was founded in 2020. We build developer tools for data pipelines.
The flagship product is a chunking library for RAG. It supports multiple strategies.

Pricing is based on usage. Contact sales for enterprise licensing.
We offer a free tier for projects under 10k documents.

Support is available via email and our documentation site. Response time is under 24 hours.
""".strip()


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run fallback vs LLM comparison (run manually)",
)
def test_windowed_fallback_substantially_similar_to_llm_chunking():
    """
    Compare LLM chunking vs windowed fallback on the same short doc: chunk with LLM (A),
    then chunk the same doc into parts via _windowed_fallback (B). Use the LLM to judge
    whether A and B are substantially similar in content coverage and RAG usefulness.

    Run manually: OPENAI_API_KEY=sk-... pytest tests/test_node_parser.py::test_windowed_fallback_substantially_similar_to_llm_chunking -v
    """
    assert len(_COMPARISON_DOC) < DEFAULT_MAX_DOC_CHARS_FOR_LLM
    doc_id = "comparison_doc.txt"
    strategy_id = "s_compare"
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)

    # A: LLM chunking (non-windowed path)
    parser_llm = LLMNodeParser(
        strategy_id=strategy_id,
        strategy_instruction="Split by topic or paragraph. Keep related content together.",
        max_doc_chars_for_llm=DEFAULT_MAX_DOC_CHARS_FOR_LLM,
    )
    doc = Document(text=_COMPARISON_DOC, id_=doc_id)
    nodes_llm = parser_llm.get_nodes_from_documents([doc])
    chunks_a = [n.text.strip() for n in nodes_llm if n.text.strip()]

    # B: Windowed fallback on same doc (small windows so we get multiple parts)
    window_size = 350
    overlap = 80
    nodes_windowed = _windowed_fallback(
        _COMPARISON_DOC,
        doc_id,
        strategy_id,
        max_chars=window_size,
        overlap_chars=overlap,
        id_prefix="compare_test",
    )
    chunks_b = [n.text.strip() for n in nodes_windowed if n.text.strip()]

    assert len(chunks_a) >= 1
    assert len(chunks_b) >= 2, "windowed fallback should yield multiple chunks for this doc length"

    prompt = f"""You are evaluating a fallback chunking strategy for RAG. When a document is too long for the model to chunk semantically, we use fixed-size overlapping windows (chunking B) instead of LLM semantic chunks (chunking A).

Document (full text):
---
{_COMPARISON_DOC}
---

Chunking A (LLM semantic chunks — used when doc fits in context):
---
{chr(10).join(f"[{i+1}] {t}" for i, t in enumerate(chunks_a))}
---

Chunking B (fixed-size overlapping windows — fallback when doc is too long):
---
{chr(10).join(f"[{i+1}] {t}" for i, t in enumerate(chunks_b))}
---

Question: Does chunking B preserve all the key information from the document (facts, topics, contact/support, pricing) so that a RAG system could still retrieve relevant chunks and answer user questions? B may split sentences at window boundaries; we only need to know if the same information is still present and findable.
Reply with exactly YES or NO on the first line, then one short sentence. Answer YES if the windowed fallback still contains the document's key information and is usable for RAG."""

    response = Settings.llm.complete(prompt)
    answer = str(response).strip().upper()
    # Accept YES anywhere in first line or clearly in response
    first_line = answer.split("\n")[0].strip() if answer else ""
    is_yes = "YES" in first_line or (
        "YES" in answer and "NO" not in answer.split("YES")[0]
    )
    assert is_yes, (
        f"LLM judged chunkings not substantially similar. Response: {response!r}"
    )

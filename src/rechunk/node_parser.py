"""
LLM-based node parser that chunks documents according to a strategy instruction.

Strategy is parameterized natural language; the LLM performs the chunking.
Implements ReChunk v0.1 (single-strategy chunking).

Documents that exceed the model's context length are never sent to the LLM;
they are chunked with a semi-overlapping window fallback (same as after parse errors).
"""

import json
import re
from typing import Any, List, Optional, Sequence

from llama_index.core.llms import LLM
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, MetadataMode, TextNode

# Conservative limit so prompt + document stay under typical 16k context (~4 chars/token).
# Override via LLMNodeParser(max_doc_chars_for_llm=...) if using a larger context model.
DEFAULT_MAX_DOC_CHARS_FOR_LLM = 50_000

# Window size and overlap for fallback chunking (used when doc is too long or LLM fails).
FALLBACK_WINDOW_CHARS = 24_000
FALLBACK_OVERLAP_CHARS = 4_000


CHUNKING_PROMPT = """You are a document chunking engine. Given the document below, apply the following
chunking strategy and return a JSON array of chunks.

Strategy: {strategy_instruction}

Rules:
- Each chunk must be self-contained enough to answer a question without requiring adjacent context
- Include enough surrounding context in each chunk to make it meaningful in isolation
- For each chunk you MUST provide start_char and end_char: the 0-based character indices of that chunk in the document. start_char is the index of the first character (inclusive), end_char is the index of the character after the last (exclusive). So document.substring(start_char, end_char) must equal the chunk content.
- Return ONLY valid JSON. No preamble, no explanation.

Response format:
[
  {{
    "chunk_id": "unique string",
    "content": "the chunk text (must exactly match document from start_char to end_char)",
    "start_char": 0,
    "end_char": 142,
    "metadata": {{ "strategy": "{strategy_id}", "source_doc": "{doc_id}" }}
  }}
]

Document:
{document_text}
"""


def _extract_json_array(text: str) -> List[dict]:
    """Extract a JSON array from LLM output, tolerating markdown code fences."""
    text = text.strip()
    # Strip optional markdown code block
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text)


def _windowed_fallback(
    text: str,
    doc_id: str,
    strategy_id: str,
    *,
    max_chars: int = FALLBACK_WINDOW_CHARS,
    overlap_chars: int = FALLBACK_OVERLAP_CHARS,
    id_prefix: str = "window_fallback",
) -> List[TextNode]:
    """
    Chunk long text into semi-overlapping windows. Used when the doc exceeds
    model context length (pre-check) or when the LLM call fails (post-error).
    """
    text_str = text if isinstance(text, str) else str(text)
    if not text_str.strip():
        return []
    step = max(1, max_chars - overlap_chars)
    out: List[TextNode] = []
    start = 0
    part_idx = 1
    while start < len(text_str):
        end = min(start + max_chars, len(text_str))
        chunk_text = text_str[start:end]
        fallback_id = f"{doc_id}_{id_prefix}_part{part_idx}"
        node = TextNode(
            id_=fallback_id,
            text=chunk_text,
            metadata={"strategy": strategy_id, "source_doc": doc_id},
            ref_doc_id=doc_id,
        )
        node.start_char_idx = start
        node.end_char_idx = end
        out.append(node)
        if end >= len(text_str):
            break
        start += step
        part_idx += 1
    return out


class LLMNodeParser(NodeParser):
    """
    Node parser that uses an LLM to chunk documents according to a strategy instruction.

    Each chunk is tagged with strategy_id and source_doc in metadata so that
    multi-strategy layers can be attributed (v0.2+).

    Documents longer than max_doc_chars_for_llm are never sent to the LLM; they
    are chunked with semi-overlapping windows (same as the post-error fallback).
    """

    strategy_id: str
    strategy_instruction: str
    llm: Optional[LLM] = None
    max_doc_chars_for_llm: int = DEFAULT_MAX_DOC_CHARS_FOR_LLM

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        import sys
        from llama_index.core.utils import get_tqdm_iterable

        out: List[BaseNode] = []
        node_list = list(nodes)
        total = len(node_list)
        items = get_tqdm_iterable(node_list, show_progress, "ReChunk (LLM)")
        for i, node in enumerate(items):
            doc_id = getattr(node, "id_", None) or getattr(node, "node_id", None) or ""
            print(f"      [{i + 1}/{total}] {doc_id}", file=sys.stderr, flush=True)
            if hasattr(node, "text"):
                text = node.text
            else:
                text = node.get_content(metadata_mode=MetadataMode.NONE)
            if not text.strip():
                continue
            # Pre-check: if doc exceeds model context length, use windowed fallback
            # instead of calling the LLM (avoids a guaranteed failure and retries).
            if len(text) > self.max_doc_chars_for_llm:
                import sys

                print(
                    f"      [SKIP] Doc over {self.max_doc_chars_for_llm} chars ({doc_id!r}). Using semi-overlapping window fallback.",
                    file=sys.stderr,
                    flush=True,
                )
                out.extend(
                    _windowed_fallback(
                        text,
                        doc_id,
                        self.strategy_id,
                        id_prefix="length_fallback",
                    )
                )
                continue
            try:
                prompt = CHUNKING_PROMPT.format(
                    strategy_instruction=self.strategy_instruction,
                    strategy_id=self.strategy_id,
                    doc_id=doc_id,
                    document_text=text,
                )
                llm = self.llm
                if llm is None:
                    from llama_index.core import Settings

                    llm = Settings.llm
                response = llm.complete(prompt)
                raw = str(response).strip()
                try:
                    chunks = _extract_json_array(raw)
                except (json.JSONDecodeError, TypeError):
                    # If the model returned malformed JSON, fall back to a
                    # single chunk. We may later choose to re-ask the model
                    # or apply a repair heuristic instead.
                    chunks = [
                        {
                            "chunk_id": f"{doc_id}_fallback",
                            "content": text,
                            "start_char": 0,
                            "end_char": len(text),
                            "metadata": {"strategy": self.strategy_id, "source_doc": doc_id},
                        }
                    ]
                doc_len = len(text)
                for c in chunks:
                    content = c.get("content", "")
                    meta = c.get("metadata") or {}
                    meta.setdefault("strategy", self.strategy_id)
                    meta.setdefault("source_doc", doc_id)
                    start_char_idx = c.get("start_char")
                    end_char_idx = c.get("end_char")
                    if start_char_idx is not None and end_char_idx is not None:
                        try:
                            start_char_idx = int(start_char_idx)
                            end_char_idx = int(end_char_idx)
                            if start_char_idx < 0 or end_char_idx > doc_len or start_char_idx >= end_char_idx:
                                start_char_idx = end_char_idx = None
                        except (TypeError, ValueError):
                            start_char_idx = end_char_idx = None
                    else:
                        start_char_idx = end_char_idx = None
                    temp_node = TextNode(text=content)
                    chunk_id = c.get("chunk_id") or self.id_func(temp_node)
                    new_node = TextNode(
                        id_=chunk_id,
                        text=content,
                        metadata=meta,
                        ref_doc_id=doc_id,
                    )
                    if start_char_idx is not None and end_char_idx is not None:
                        new_node.start_char_idx = start_char_idx
                        new_node.end_char_idx = end_char_idx
                    out.append(new_node)
            except Exception as e:
                # If an individual document fails (timeout, API error, context_length_exceeded, etc.),
                # fall back to semi-overlapping windows so we don't lose the doc.
                print(
                    f"      [SKIP] Failed for {doc_id!r}: {e}. Using semi-overlapping window fallback.",
                    file=sys.stderr,
                    flush=True,
                )
                out.extend(
                    _windowed_fallback(
                        text,
                        doc_id,
                        self.strategy_id,
                        id_prefix="error_fallback",
                    )
                )
        return out

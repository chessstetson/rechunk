"""
LLM-based node parser that chunks documents according to a strategy instruction.

Strategy is parameterized natural language; the LLM performs the chunking.
Implements ReChunk v0.1 (single-strategy chunking).
"""

import json
import re
from typing import Any, List, Optional, Sequence

from llama_index.core.llms import LLM
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, MetadataMode, TextNode


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


class LLMNodeParser(NodeParser):
    """
    Node parser that uses an LLM to chunk documents according to a strategy instruction.

    Each chunk is tagged with strategy_id and source_doc in metadata so that
    multi-strategy layers can be attributed (v0.2+).
    """

    strategy_id: str
    strategy_instruction: str
    llm: Optional[LLM] = None

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
                # If an individual document fails (timeout, API error, etc.),
                # we currently fall back to a simple character-window split to
                # avoid losing the doc entirely. This is intentionally ad hoc
                # and may be replaced later with a more principled policy.
                print(
                    f"      [SKIP] Failed for {doc_id!r}: {e}. Using simple windowed fallback.",
                    file=sys.stderr,
                    flush=True,
                )
                max_chars = 24000
                text_str = text if isinstance(text, str) else str(text)
                start = 0
                part_idx = 1
                while start < len(text_str):
                    chunk_text = text_str[start : start + max_chars]
                    fallback_id = f"{doc_id}_error_fallback_part{part_idx}"
                    out.append(
                        TextNode(
                            id_=fallback_id,
                            text=chunk_text,
                            metadata={"strategy": self.strategy_id, "source_doc": doc_id},
                            ref_doc_id=doc_id,
                        )
                    )
                    start += max_chars
                    part_idx += 1
        return out

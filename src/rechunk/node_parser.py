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
- Return ONLY valid JSON. No preamble, no explanation.

Response format:
[
  {{
    "chunk_id": "unique string",
    "content": "the chunk text",
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
        from llama_index.core.utils import get_tqdm_iterable

        out: List[BaseNode] = []
        items = get_tqdm_iterable(nodes, show_progress, "ReChunk (LLM)")
        for node in items:
            doc_id = getattr(node, "id_", None) or getattr(node, "node_id", None) or ""
            if hasattr(node, "text"):
                text = node.text
            else:
                text = node.get_content(metadata_mode=MetadataMode.NONE)
            if not text.strip():
                continue
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
                # Fallback: treat whole document as one chunk
                chunks = [
                    {
                        "chunk_id": f"{doc_id}_fallback",
                        "content": text,
                        "metadata": {"strategy": self.strategy_id, "source_doc": doc_id},
                    }
                ]
            for c in chunks:
                content = c.get("content", "")
                meta = c.get("metadata") or {}
                meta.setdefault("strategy", self.strategy_id)
                meta.setdefault("source_doc", doc_id)
                temp_node = TextNode(text=content)
                chunk_id = c.get("chunk_id") or self.id_func(temp_node)
                out.append(
                    TextNode(
                        id_=chunk_id,
                        text=content,
                        metadata=meta,
                        ref_doc_id=doc_id,
                    )
                )
        return out

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

from rechunk.derived_metadata import build_sorted_source_spans_metadata


def _metadata_source_spans_from_pairs(pairs: list[tuple[int, int]]) -> list[dict[str, int]]:
    return [{"start_char": int(s), "end_char": int(e)} for s, e in sorted(pairs, key=lambda t: (t[0], t[1]))]


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
- **Character positions:** indices are 0-based into the document string below. ``end_char`` is exclusive (the index after the last included character).
- **Single contiguous excerpt:** provide ``start_char`` and ``end_char`` so that the chunk text equals ``document[start_char:end_char]``.
- **Multiple non-contiguous excerpts (same logical chunk):** use a ``spans`` array instead of a single range. Each element is ``{{"start_char": i, "end_char": j}}`` in **document order** (non-overlapping or abutting ranges). The chunk ``content`` must be the **exact concatenation** of ``document[start_char:end_char]`` for each span in order (no extra spaces or separators unless they appear between spans in the document).
- Do not use both ``spans`` and top-level ``start_char``/``end_char`` for the same chunk; prefer ``spans`` when the chunk pulls text from more than one region.
- Return ONLY valid JSON. No preamble, no explanation.

Response format (single-span example):
[
  {{
    "chunk_id": "unique string",
    "content": "the chunk text (must exactly match the referenced span(s) in the document)",
    "start_char": 0,
    "end_char": 142,
    "metadata": {{ "strategy": "{strategy_id}", "source_doc": "{doc_id}" }}
  }}
]

Multi-span example (one chunk, two disjoint regions):
[
  {{
    "chunk_id": "another id",
    "content": "first region textsecond region text",
    "spans": [{{"start_char": 10, "end_char": 25}}, {{"start_char": 100, "end_char": 120}}],
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


def _parse_one_span_bounds(
    item: object,
    doc_len: int,
) -> tuple[int, int] | None:
    """Parse start/end from a dict (LLM span object). Returns None if invalid."""
    if not isinstance(item, dict):
        return None
    s = item.get("start_char")
    if s is None and "start" in item:
        s = item.get("start")
    e = item.get("end_char")
    if e is None and "end" in item:
        e = item.get("end")
    if s is None or e is None:
        return None
    try:
        si, ei = int(s), int(e)
    except (TypeError, ValueError):
        return None
    if si < 0 or ei > doc_len or si >= ei:
        return None
    return si, ei


def _extract_spans_from_llm_chunk(chunk: dict, *, doc_len: int) -> list[tuple[int, int]] | None:
    """
    Return ordered, validated (start, end) spans for one chunk dict.

    Prefer ``spans`` (multi-region). Otherwise a single ``start_char``/``end_char``.
    Returns None if no valid span data (caller may fall back to content-only).
    """
    raw_spans = chunk.get("spans")
    if raw_spans is not None:
        if not isinstance(raw_spans, list) or len(raw_spans) == 0:
            return None
        out: list[tuple[int, int]] = []
        for item in raw_spans:
            pair = _parse_one_span_bounds(item, doc_len)
            if pair is None:
                return None
            out.append(pair)
        # Optional: enforce document order / no overlap — trust model; merge abutting is OK
        return out

    s = chunk.get("start_char")
    e = chunk.get("end_char")
    if s is None or e is None:
        return None
    try:
        si, ei = int(s), int(e)
    except (TypeError, ValueError):
        return None
    if si < 0 or ei > doc_len or si >= ei:
        return None
    return [(si, ei)]


def _reconstruct_content_from_spans(document: str, spans: list[tuple[int, int]]) -> str:
    return "".join(document[s:e] for s, e in spans)


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
            metadata={
                "strategy": strategy_id,
                "source_doc": doc_id,
                "source_spans": [{"start_char": start, "end_char": end}],
            },
            ref_doc_id=doc_id,
        )
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
                    meta = dict(c.get("metadata") or {})
                    meta.setdefault("strategy", self.strategy_id)
                    meta.setdefault("source_doc", doc_id)

                    span_list = _extract_spans_from_llm_chunk(c, doc_len=doc_len)
                    content = c.get("content", "")
                    if not isinstance(content, str):
                        content = str(content or "")

                    if span_list:
                        reconstructed = _reconstruct_content_from_spans(text, span_list)
                        # Trust span positions if content drifts (whitespace / model slip)
                        if content.strip() != reconstructed.strip():
                            content = reconstructed
                        meta["source_spans"] = _metadata_source_spans_from_pairs(span_list)
                    else:
                        sc = c.get("start_char")
                        ec = c.get("end_char")
                        if sc is not None and ec is not None:
                            try:
                                sci, eci = int(sc), int(ec)
                                if 0 <= sci < eci <= doc_len:
                                    meta["source_spans"] = _metadata_source_spans_from_pairs([(sci, eci)])
                            except (TypeError, ValueError):
                                pass

                    temp_node = TextNode(text=content)
                    chunk_id = c.get("chunk_id") or self.id_func(temp_node)
                    new_node = TextNode(
                        id_=chunk_id,
                        text=content,
                        metadata=meta,
                        ref_doc_id=doc_id,
                    )
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


DERIVED_CHUNKING_PROMPT = """You are a document analysis engine. Given the document below, apply the strategy
and return a JSON array of **derived nodes**.

Strategy: {strategy_instruction}

Rules:
- Each node has **freely written** ``content`` (synthetic text optimized for retrieval — summaries, inventories, profiles, etc.). It does **not** need to be a verbatim substring of the document.
- Every node **must** include ``source_spans``: a non-empty list of regions in the document this node is grounded in. Each element: ``{{"start_char": i, "end_char": j}}`` with 0-based indices and **exclusive** ``end_char``. You may add an optional short ``quote`` (≤120 chars, verbatim from the document within that span) to anchor the UI.
- Order of ``source_spans`` does not matter for identity; cite every region the node depends on.
- Return ONLY valid JSON. No preamble, no explanation.

Response format:
[
  {{
    "node_id": "unique string",
    "content": "generated analysis text — not necessarily verbatim from the document",
    "source_spans": [
      {{"start_char": 0, "end_char": 200, "quote": "optional short verbatim anchor"}},
      {{"start_char": 500, "end_char": 620}}
    ],
    "metadata": {{ "strategy": "{strategy_id}", "source_doc": "{doc_id}" }}
  }}
]

Document:
{document_text}
"""


class DerivedNodeParser(NodeParser):
    """
    LLM produces **derived** text plus ``source_spans`` for provenance (see ``DERIVED_CHUNKS.md``).

    Same length pre-check and windowed fallback as :class:`LLMNodeParser` when the document is too
    long for one model call (fallback is **not** derived-quality; it is overlapping windows).
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
        items = get_tqdm_iterable(node_list, show_progress, "ReChunk (derived)")
        for i, node in enumerate(items):
            doc_id = getattr(node, "id_", None) or getattr(node, "node_id", None) or ""
            print(f"      [{i + 1}/{total}] derived {doc_id}", file=sys.stderr, flush=True)
            if hasattr(node, "text"):
                text = node.text
            else:
                text = node.get_content(metadata_mode=MetadataMode.NONE)
            if not text.strip():
                continue
            if len(text) > self.max_doc_chars_for_llm:
                print(
                    f"      [SKIP] Doc over {self.max_doc_chars_for_llm} chars ({doc_id!r}). "
                    "Using semi-overlapping window fallback (not derived-quality).",
                    file=sys.stderr,
                    flush=True,
                )
                out.extend(
                    _windowed_fallback(
                        text,
                        doc_id,
                        self.strategy_id,
                        id_prefix="length_fallback_derived",
                    )
                )
                continue
            try:
                prompt = DERIVED_CHUNKING_PROMPT.format(
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
                    chunks = [
                        {
                            "node_id": f"{doc_id}_fallback",
                            "content": text[:8000] + ("…" if len(text) > 8000 else ""),
                            "source_spans": [{"start_char": 0, "end_char": len(text)}],
                            "metadata": {"strategy": self.strategy_id, "source_doc": doc_id},
                        }
                    ]
                doc_len = len(text)
                seen_keys: set[tuple[tuple[int, int], ...]] = set()
                for c in chunks:
                    meta = dict(c.get("metadata") or {})
                    meta.setdefault("strategy", self.strategy_id)
                    meta.setdefault("source_doc", doc_id)
                    meta["derived"] = True

                    raw_spans = c.get("source_spans")
                    if not isinstance(raw_spans, list):
                        print(
                            f"      [WARN] derived node missing source_spans; skipping ({doc_id!r})",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    stored_spans = build_sorted_source_spans_metadata(raw_spans, doc_len=doc_len)
                    if not stored_spans:
                        print(
                            f"      [WARN] derived node invalid source_spans; skipping ({doc_id!r})",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue

                    pair_key = tuple(
                        (int(d["start_char"]), int(d["end_char"])) for d in stored_spans
                    )
                    if pair_key in seen_keys:
                        print(
                            f"      [WARN] duplicate derived canonical span key {pair_key!r}; "
                            f"keeping one node ({doc_id!r})",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    seen_keys.add(pair_key)

                    content = c.get("content", "")
                    if not isinstance(content, str):
                        content = str(content or "")
                    if not content.strip():
                        print(
                            f"      [WARN] derived node empty content; skipping ({doc_id!r})",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue

                    meta["source_spans"] = stored_spans

                    nid = c.get("node_id") or c.get("chunk_id")
                    temp_node = TextNode(text=content)
                    chunk_id = nid or self.id_func(temp_node)
                    new_node = TextNode(
                        id_=str(chunk_id),
                        text=content,
                        metadata=meta,
                        ref_doc_id=doc_id,
                    )
                    out.append(new_node)
            except Exception as e:
                print(
                    f"      [SKIP] Derived parse failed for {doc_id!r}: {e}. Window fallback.",
                    file=sys.stderr,
                    flush=True,
                )
                out.extend(
                    _windowed_fallback(
                        text,
                        doc_id,
                        self.strategy_id,
                        id_prefix="error_fallback_derived",
                    )
                )
        return out

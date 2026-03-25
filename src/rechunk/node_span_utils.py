"""Map chunk nodes to character spans in the source extracted text."""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import MetadataMode

from rechunk.derived_metadata import bbox_from_source_spans


def char_spans_for_nodes(full_text: str, nodes: list[Any]) -> list[tuple[int, int]]:
    """
    Return ``(start, end)`` character ranges for each node (end exclusive).

    Uses ``metadata['source_spans']`` when present (bounding box over all regions).
    Otherwise searches for node text in ``full_text`` from a moving cursor.
    """
    out: list[tuple[int, int]] = []
    search_from = 0
    n = len(full_text)

    for node in nodes:
        if hasattr(node, "get_content"):
            t = node.get_content(metadata_mode=MetadataMode.NONE)
        else:
            t = getattr(node, "text", "") or ""
        meta = getattr(node, "metadata", None) or {}
        if isinstance(meta, dict):
            db = bbox_from_source_spans(meta, doc_len=n)
            if db is not None:
                lo, hi = db
                out.append((lo, hi))
                search_from = max(search_from, hi)
                continue

        if not (t or "").strip():
            out.append((0, 0))
            continue

        idx = full_text.find(t, search_from)
        if idx < 0:
            idx = full_text.find(t.strip(), search_from)
        if idx < 0:
            idx = search_from
            end = min(n, idx + len(t))
        else:
            end = idx + len(t)
        out.append((idx, end))
        search_from = max(search_from, idx + 1)

    return out


def ensure_metadata_source_spans_for_nodes(full_text: str, nodes: list[Any]) -> None:
    """
    Set ``metadata['source_spans']`` to ``[{{start_char, end_char}}]`` when missing.

    Uses :func:`char_spans_for_nodes` (sentence/token splitters, fallbacks, etc.).
    Does not overwrite existing ``source_spans`` (LLM / derived).
    """
    spans = char_spans_for_nodes(full_text, nodes)
    for node, (lo, hi) in zip(nodes, spans, strict=True):
        meta = dict(getattr(node, "metadata", None) or {})
        if meta.get("source_spans"):
            node.metadata = meta
            continue
        meta["source_spans"] = [{"start_char": int(lo), "end_char": int(hi)}]
        node.metadata = meta

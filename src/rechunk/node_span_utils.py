"""Map chunk nodes to character spans in the source extracted text."""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import MetadataMode


def _bbox_from_span_ranges(meta: dict | None, *, doc_len: int) -> tuple[int, int] | None:
    """If metadata has ``span_ranges`` (list of [start, end]), return inclusive bbox."""
    if not meta:
        return None
    raw = meta.get("span_ranges")
    if not isinstance(raw, list) or not raw:
        return None
    pairs: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                s, e = int(item[0]), int(item[1])
            except (TypeError, ValueError):
                return None
            if s < 0 or e > doc_len or s >= e:
                return None
            pairs.append((s, e))
        else:
            return None
    if not pairs:
        return None
    return min(s for s, _ in pairs), max(e for _, e in pairs)


def char_spans_for_nodes(full_text: str, nodes: list[Any]) -> list[tuple[int, int]]:
    """
    Return ``(start, end)`` character ranges for each node (end exclusive).

    If node metadata has ``span_ranges`` (multi-region LLM chunks), returns the **bounding
    box** ``(min_start, max_end)`` so ``full_text[start:end]`` may include gaps; the exact
    regions are in metadata.

    Otherwise uses :attr:`start_char_idx` / :attr:`end_char_idx` when valid, or searches
    for node text in ``full_text`` from a moving cursor.
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
        bbox = _bbox_from_span_ranges(meta if isinstance(meta, dict) else None, doc_len=n)
        if bbox is not None:
            lo, hi = bbox
            out.append((lo, hi))
            search_from = max(search_from, hi)
            continue

        sc = getattr(node, "start_char_idx", None)
        ec = getattr(node, "end_char_idx", None)
        if sc is not None and ec is not None:
            try:
                si, ei = int(sc), int(ec)
                if 0 <= si < ei <= n:
                    out.append((si, ei))
                    search_from = max(search_from, ei)
                    continue
            except (TypeError, ValueError):
                pass

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

"""
``source_spans`` provenance helpers (all strategy kinds).

Parsing/sorting for LLM output, bbox for storage, and vector row merge keys.
Derived strategies (synthetic ``content`` + ``source_spans``) are summarized in **README.md**.
"""

from __future__ import annotations

from typing import Any


def parse_source_spans_raw(raw: object, *, doc_len: int) -> list[tuple[int, int]] | None:
    """
    Parse ``source_spans`` from LLM JSON into validated (start, end) pairs (end exclusive).

    Each item may be a dict with ``start_char``/``end_char`` (or ``start``/``end``).
    Returns ``None`` if missing, empty, or invalid.
    """
    if not isinstance(raw, list) or len(raw) == 0:
        return None
    out: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        s = item.get("start_char")
        if s is None:
            s = item.get("start")
        e = item.get("end_char")
        if e is None:
            e = item.get("end")
        if s is None or e is None:
            return None
        try:
            si, ei = int(s), int(e)
        except (TypeError, ValueError):
            return None
        if si < 0 or ei > doc_len or si >= ei:
            return None
        out.append((si, ei))
    return out


def build_sorted_source_spans_metadata(raw: list[Any], *, doc_len: int) -> list[dict[str, Any]] | None:
    """
    Validate ``source_spans`` from LLM output and return **sorted** list of
    ``{"start_char", "end_char", "quote"?}`` for storage and merge keys.
    """
    pairs = parse_source_spans_raw(raw, doc_len=doc_len)
    if not pairs:
        return None
    quote_by: dict[tuple[int, int], str] = {}
    for item, (s, e) in zip(raw, pairs, strict=True):
        if isinstance(item, dict):
            q = item.get("quote")
            if isinstance(q, str) and q.strip():
                quote_by.setdefault((s, e), q.strip()[:500])
    out: list[dict[str, Any]] = []
    for s, e in sorted(pairs, key=lambda t: (t[0], t[1])):
        d: dict[str, Any] = {"start_char": s, "end_char": e}
        if (s, e) in quote_by:
            d["quote"] = quote_by[(s, e)]
        out.append(d)
    return out


def bbox_from_source_spans(meta: dict[str, Any] | None, *, doc_len: int) -> tuple[int, int] | None:
    """Bounding box over ``metadata['source_spans']`` (all strategy kinds)."""
    if not meta:
        return None
    raw = meta.get("source_spans")
    if not isinstance(raw, list) or not raw:
        return None
    pairs: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, dict):
            s, e = item.get("start_char"), item.get("end_char")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            s, e = item[0], item[1]
        else:
            return None
        try:
            si, ei = int(s), int(e)
        except (TypeError, ValueError):
            return None
        if si < 0 or ei > doc_len or si >= ei:
            return None
        pairs.append((si, ei))
    if not pairs:
        return None
    return min(s for s, _ in pairs), max(e for _, e in pairs)


def canonical_source_spans_merge_key(meta: dict[str, Any] | None) -> tuple[tuple[int, int], ...] | None:
    """
    Stable merge key from ``metadata['source_spans']``: sorted tuple of ``(start, end)`` pairs.

    Returns ``None`` if missing or invalid ``source_spans``.
    """
    if not meta:
        return None
    raw = meta.get("source_spans")
    if not isinstance(raw, list) or not raw:
        return None
    pair_list: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, dict):
            s, e = item.get("start_char"), item.get("end_char")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            s, e = item[0], item[1]
        else:
            return None
        try:
            pair_list.append((int(s), int(e)))
        except (TypeError, ValueError):
            return None
    if not pair_list:
        return None
    return tuple(sorted(pair_list, key=lambda t: (t[0], t[1])))

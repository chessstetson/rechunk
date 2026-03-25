"""Tests for derived chunk metadata helpers."""

from rechunk.derived_metadata import (
    bbox_from_source_spans,
    build_sorted_source_spans_metadata,
    canonical_source_spans_merge_key,
    parse_source_spans_raw,
)
from rechunk.strategies import normalize_strategy_kind, strategy_definition_uses_derived


def test_normalize_strategy_kind_derived():
    assert normalize_strategy_kind("derived") == "derived"
    assert normalize_strategy_kind("DERIVED") == "derived"


def test_strategy_definition_uses_derived():
    assert strategy_definition_uses_derived({"kind": "derived", "id": "x", "instruction": "y"})
    assert not strategy_definition_uses_derived({"kind": "llm", "id": "x", "instruction": "y"})


def test_parse_source_spans_raw():
    raw = [{"start_char": 10, "end_char": 20}, {"start_char": 0, "end_char": 5}]
    assert parse_source_spans_raw(raw, doc_len=25) == [(10, 20), (0, 5)]


def test_build_sorted_source_spans_metadata_sorts_and_quotes():
    raw = [
        {"start_char": 10, "end_char": 12, "quote": "hi"},
        {"start_char": 0, "end_char": 3},
    ]
    out = build_sorted_source_spans_metadata(raw, doc_len=100)
    assert out == [
        {"start_char": 0, "end_char": 3},
        {"start_char": 10, "end_char": 12, "quote": "hi"},
    ]


def test_canonical_source_spans_merge_key_order_invariant():
    meta_a = {
        "source_spans": [{"start_char": 10, "end_char": 20}, {"start_char": 0, "end_char": 5}],
    }
    meta_b = {
        "source_spans": [{"start_char": 0, "end_char": 5}, {"start_char": 10, "end_char": 20}],
    }
    assert canonical_source_spans_merge_key(meta_a) == canonical_source_spans_merge_key(meta_b)
    assert canonical_source_spans_merge_key(meta_a) == ((0, 5), (10, 20))


def test_bbox_from_source_spans():
    meta = {
        "source_spans": [{"start_char": 10, "end_char": 20}, {"start_char": 0, "end_char": 5}],
    }
    assert bbox_from_source_spans(meta, doc_len=100) == (0, 20)


def test_canonical_source_spans_merge_key_none_when_missing():
    assert canonical_source_spans_merge_key({}) is None
    assert canonical_source_spans_merge_key({"source_spans": []}) is None

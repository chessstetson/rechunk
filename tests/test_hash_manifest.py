"""Hash-only manifest load/write (retrieval boundary)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from rechunk.cache import compute_content_hash
from rechunk.corpus import ContentRef
from rechunk.hash_manifest import (
    load_content_refs_from_manifest,
    write_hash_manifest,
    write_manifest_from_filesystem_scan,
)


def test_load_array_format() -> None:
    h1 = "a" * 64
    h2 = "b" * 64
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.json"
        p.write_text(json.dumps([h1, h2]), encoding="utf-8")
        refs = load_content_refs_from_manifest(p)
        assert refs == [
            ContentRef(content_hash=h1, source_hint=None),
            ContentRef(content_hash=h2, source_hint=None),
        ]


def test_load_content_hashes_key() -> None:
    h = "c" * 64
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.json"
        p.write_text(json.dumps({"content_hashes": [h]}), encoding="utf-8")
        refs = load_content_refs_from_manifest(p)
        assert len(refs) == 1
        assert refs[0].content_hash == h


def test_load_dedupes_and_normalizes_case() -> None:
    h = "d" * 64
    upper = "D" * 64
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.json"
        p.write_text(json.dumps([h, upper]), encoding="utf-8")
        refs = load_content_refs_from_manifest(p)
        assert len(refs) == 1


def test_load_rejects_invalid_hex() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.json"
        p.write_text(json.dumps(["not-a-hash"]), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid content_hash"):
            load_content_refs_from_manifest(p)


def test_write_hash_manifest_roundtrip() -> None:
    hashes = [compute_content_hash("x"), compute_content_hash("y")]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.json"
        write_hash_manifest(p, hashes)
        loaded = load_content_refs_from_manifest(p)
        assert [r.content_hash for r in loaded] == [h.lower() for h in hashes]


def test_write_manifest_from_filesystem_scan() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.txt").write_text("hello manifest", encoding="utf-8")
        out = root / "hashes.json"
        write_manifest_from_filesystem_scan(root, out)
        refs = load_content_refs_from_manifest(out)
        assert len(refs) == 1
        assert refs[0].content_hash == compute_content_hash("hello manifest")

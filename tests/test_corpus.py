"""Corpus scan: content refs without retaining full text in lists."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rechunk.cache import compute_content_hash
from rechunk.corpus import ContentRef, scan_filesystem_corpus


def test_scan_single_file_content_ref_aligns_with_hash() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "note.txt"
        p.write_text("hello corpus", encoding="utf-8")
        refs, doc_ids = scan_filesystem_corpus(p)
        assert len(refs) == 1
        assert doc_ids == ["note.txt"]
        assert refs[0] == ContentRef(
            content_hash=compute_content_hash("hello corpus"),
            source_hint="note.txt",
        )


def test_scan_directory_dedupes_by_content_hash() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.txt").write_text("same", encoding="utf-8")
        (root / "b.txt").write_text("same", encoding="utf-8")
        refs, doc_ids = scan_filesystem_corpus(root)
        assert len(refs) == 1
        assert doc_ids == ["a.txt"]

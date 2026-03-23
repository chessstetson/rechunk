"""CorpusManager protocol implementations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rechunk.cache import compute_content_hash
from rechunk.corpus import ContentRef
from rechunk.corpus_manager import (
    FilesystemCorpusManager,
    HashManifestCorpusManager,
    TemporalIngestHints,
)


def test_filesystem_manager_lists_refs_and_hints(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("hello", encoding="utf-8")
    m = FilesystemCorpusManager(tmp_path)
    refs = m.list_active_content_refs()
    assert len(refs) == 1
    assert refs[0].content_hash == compute_content_hash("hello")
    hints = m.temporal_ingest_hints()
    assert isinstance(hints, TemporalIngestHints)
    assert hints.docs_root == tmp_path.resolve()
    assert hints.doc_ids == ["x.txt"]
    assert "Corpus scan" in m.summary_message(len(refs))


def test_filesystem_single_file_hints_root_is_parent(tmp_path: Path) -> None:
    f = tmp_path / "only.txt"
    f.write_text("a", encoding="utf-8")
    m = FilesystemCorpusManager(f)
    m.list_active_content_refs()
    hints = m.temporal_ingest_hints()
    assert hints is not None
    assert hints.docs_root == tmp_path.resolve()
    assert hints.doc_ids == ["only.txt"]


def test_hash_manifest_manager_no_ingest_hints(tmp_path: Path) -> None:
    h = "a" * 64
    p = tmp_path / "m.json"
    p.write_text(json.dumps([h]), encoding="utf-8")
    m = HashManifestCorpusManager(p)
    refs = m.list_active_content_refs()
    assert refs == [ContentRef(content_hash=h, source_hint=None)]
    assert m.temporal_ingest_hints() is None
    assert "Manifest (hash-only)" in m.summary_message(len(refs))

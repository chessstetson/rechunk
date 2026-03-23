"""Active corpus manifest merge (ingest-owned hash list)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from rechunk.active_corpus_manifest import (
    active_corpus_manifest_path,
    merge_content_hashes_into_active_manifest,
)


def test_merge_creates_file_sorted_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "active.json"
        monkeypatch.setenv("RECHUNK_ACTIVE_CORPUS_MANIFEST", str(p))
        h1 = "a" * 64
        h2 = "b" * 64
        out = merge_content_hashes_into_active_manifest([h2, h1, h2])
        assert out == [h1, h2]
        assert json.loads(p.read_text(encoding="utf-8")) == [h1, h2]


def test_merge_unions_with_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "active.json"
        p.write_text(json.dumps(["c" * 64]), encoding="utf-8")
        monkeypatch.setenv("RECHUNK_ACTIVE_CORPUS_MANIFEST", str(p))
        h2 = "d" * 64
        out = merge_content_hashes_into_active_manifest([h2])
        assert len(out) == 2
        assert "c" * 64 in out and h2 in out


def test_active_corpus_manifest_path_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECHUNK_ACTIVE_CORPUS_MANIFEST", "/tmp/custom.json")
    assert active_corpus_manifest_path() == Path("/tmp/custom.json").resolve()

"""Filesystem → ECS ingest activity (decoupled from vectorization)."""

from __future__ import annotations

import json

import pytest
from temporalio.testing import ActivityEnvironment

from temporal_activities import LoadIngestSnapshotInput, ingest_filesystem_corpus_from_snapshot


@pytest.mark.asyncio
async def test_ingest_activity_populates_ecs_and_exact_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RECHUNK_ECS_ROOT", str(tmp_path / "ecs"))
    monkeypatch.setenv("RECHUNK_VECTOR_STORE_DEV_ROOT", str(tmp_path / "vs"))
    manifest_path = tmp_path / "corpus_hashes.json"
    monkeypatch.setenv("RECHUNK_ACTIVE_CORPUS_MANIFEST", str(manifest_path))

    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("hello ingest", encoding="utf-8")

    from rechunk.ingest_snapshot import build_and_write_ingest_snapshot

    snap = build_and_write_ingest_snapshot(root, ["a.txt"], strategy_id="t_ingest")

    env = ActivityEnvironment()
    result = await env.run(
        ingest_filesystem_corpus_from_snapshot,
        LoadIngestSnapshotInput(snapshot_path=str(snap)),
    )
    assert result["ingested_logical_docs"] == 1
    assert result["active_unique_hashes"] == 1

    from rechunk.extracted_content import FilesystemExtractedContentService

    ecs = FilesystemExtractedContentService(tmp_path / "ecs")
    assert len(ecs.list_active_hashes()) == 1

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 1

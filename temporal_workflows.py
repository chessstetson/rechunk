"""
Temporal workflows for ReChunk. Sandbox-safe: no I/O, no activity/LLM imports.

Workflows only orchestrate; they call activities by name and pass serializable input.
"""

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

from temporalio import workflow

from temporal_vectorization_inputs import BatchDocumentVectorizationInput, DocumentVectorizationInput


# Mirror activity input dataclasses (workflow cannot import temporal_activities).
@dataclass
class LogWorkflowSummaryInput:
    strategy_id: str
    total: int
    skipped: int
    processed: int


@dataclass
class LoadIngestSnapshotInput:
    snapshot_path: str


@dataclass
class GetCachedHashesInput:
    strategy_id: str


@dataclass
class ChunkDocInput:
    strategy_id: str
    strategy_instruction: str
    model: Optional[str]
    docs_root: str
    doc_id: str
    content_hash: str


@dataclass
class BuiltinChunkInput:
    strategy_id: str
    splitter: str
    docs_root: str
    doc_id: str
    content_hash: str


@dataclass
class MergeActiveCorpusManifestInput:
    content_hashes: List[str]


@dataclass
class FilesystemCorpusIngestInput:
    """
    Ingest-only workflow: path to snapshot JSON (see :mod:`rechunk.ingest_snapshot`).
    Worker on task queue ``rechunk-ingest`` executes the activity.
    """

    ingest_snapshot_path: str


@dataclass
class StrategyChunkingInput:
    """Input for StrategyChunkingWorkflow. One strategy, many docs. Supports LLM and built-in."""

    strategy_id: str
    kind: str  # "llm" or "builtin_splitter"
    ingest_snapshot_path: str  # JSON: docs_root + per-doc hashes (see rechunk.ingest_snapshot)
    strategy_instruction: Optional[str] = None  # for kind=llm
    model: Optional[str] = None  # for kind=llm
    splitter: str = "sentence"  # for kind=builtin_splitter: "sentence" or "token"


@workflow.defn
class StrategyChunkingWorkflow:
    """Chunk documents for one strategy. Phase 2: skip-if-cached + parallel execution."""

    @workflow.run
    async def run(self, input: StrategyChunkingInput) -> dict:
        # 2.1: Resolve manifest from ingest snapshot (I/O in activity; workflow history keeps path only).
        loaded: dict = await workflow.execute_activity(
            "load_manifest_from_ingest_snapshot",
            LoadIngestSnapshotInput(snapshot_path=input.ingest_snapshot_path),
            start_to_close_timeout=timedelta(minutes=10),
        )
        docs_root: str = loaded["docs_root"]
        manifest: List[dict] = loaded["manifest"]
        cached_hashes_list: List[str] = await workflow.execute_activity(
            "get_cached_hashes",
            GetCachedHashesInput(strategy_id=input.strategy_id),
            start_to_close_timeout=timedelta(seconds=60),
        )
        cached_set = set(cached_hashes_list)
        to_process = [m for m in manifest if m["content_hash"] not in cached_set]
        total = len(manifest)
        skipped = total - len(to_process)
        processed = len(to_process)

        # 2.2: Run chunking for uncached docs in parallel (LLM or built-in by kind).
        if to_process:
            if input.kind == "builtin_splitter":
                async def chunk_one_builtin(m: dict) -> None:
                    await workflow.execute_activity(
                        "chunk_doc_with_builtin_splitter",
                        BuiltinChunkInput(
                            strategy_id=input.strategy_id,
                            splitter=input.splitter,
                            docs_root=docs_root,
                            doc_id=m["doc_id"],
                            content_hash=m["content_hash"],
                        ),
                        start_to_close_timeout=timedelta(minutes=5),
                    )
                await asyncio.gather(*[chunk_one_builtin(m) for m in to_process])
            else:
                async def chunk_one_llm(m: dict) -> None:
                    await workflow.execute_activity(
                        "chunk_doc_with_strategy",
                        ChunkDocInput(
                            strategy_id=input.strategy_id,
                            strategy_instruction=input.strategy_instruction or "",
                            model=input.model,
                            docs_root=docs_root,
                            doc_id=m["doc_id"],
                            content_hash=m["content_hash"],
                        ),
                        start_to_close_timeout=timedelta(minutes=5),
                    )
                await asyncio.gather(*[chunk_one_llm(m) for m in to_process])

        # Always log summary so the worker terminal shows feedback (e.g. "0 chunked" on re-run).
        await workflow.execute_activity(
            "log_workflow_summary",
            LogWorkflowSummaryInput(
                strategy_id=input.strategy_id,
                total=total,
                skipped=skipped,
                processed=processed,
            ),
            start_to_close_timeout=timedelta(seconds=10),
        )
        # Ingest-owned active corpus (hash-only); merged on every successful workflow completion.
        if manifest:
            unique_hashes = sorted({m["content_hash"] for m in manifest})
            await workflow.execute_activity(
                "merge_active_corpus_manifest",
                MergeActiveCorpusManifestInput(content_hashes=unique_hashes),
                start_to_close_timeout=timedelta(minutes=2),
            )
        return {"total": total, "skipped": skipped, "processed": processed}


@workflow.defn
class FilesystemCorpusIngestWorkflow:
    """
    Durable filesystem → ECS ingest. Does **not** chunk or embed; run vectorization separately.
    """

    @workflow.run
    async def run(self, input: FilesystemCorpusIngestInput) -> dict:
        result: dict = await workflow.execute_activity(
            "ingest_filesystem_corpus_from_snapshot",
            LoadIngestSnapshotInput(snapshot_path=input.ingest_snapshot_path),
            start_to_close_timeout=timedelta(hours=2),
        )
        return result


@workflow.defn
class DocumentVectorizationWorkflow:
    """
    Phase C: vectorize a single ``content_hash`` with one strategy via ECS + VectorStore.

    Legacy :class:`StrategyChunkingWorkflow` remains registered for older tests/tools.
    """

    @workflow.run
    async def run(self, input: DocumentVectorizationInput) -> dict:
        result: dict = await workflow.execute_activity(
            "vectorize_content_for_strategy",
            input,
            start_to_close_timeout=timedelta(minutes=15),
        )
        status = result.get("status", "unknown")
        skipped = 1 if status == "skipped" else 0
        processed = 1 if status == "processed" else 0
        await workflow.execute_activity(
            "log_workflow_summary",
            LogWorkflowSummaryInput(
                strategy_id=input.strategy_id,
                total=1,
                skipped=skipped,
                processed=processed,
            ),
            start_to_close_timeout=timedelta(seconds=10),
        )
        return result


@workflow.defn
class BatchDocumentVectorizationWorkflow:
    """
    Single orchestration workflow: one ``vectorize_content_for_strategy`` activity per content hash.

    Waved ``asyncio.gather`` (see ``fanout_batch_size`` on input) keeps each **workflow task**
    from replaying hundreds of activity events at once, which otherwise hits
    ``Workflow Task Timed Out`` (start-to-close). Within a wave, the worker's
    ``max_concurrent_activities`` caps simultaneous activity execution.
    """

    @workflow.run
    async def run(self, input: BatchDocumentVectorizationInput) -> dict:
        async def vectorize_one(content_hash: str) -> dict:
            return await workflow.execute_activity(
                "vectorize_content_for_strategy",
                DocumentVectorizationInput(
                    content_hash=content_hash,
                    strategy_id=input.strategy_id,
                    strategy_definition=input.strategy_definition,
                    strategy_fingerprint=input.strategy_fingerprint,
                    embedding_fingerprint=input.embedding_fingerprint,
                    vector_schema_version=input.vector_schema_version,
                ),
                start_to_close_timeout=timedelta(minutes=15),
            )

        results: list[dict] = []
        if input.content_hashes:
            fanout = max(1, int(input.fanout_batch_size))
            hashes = input.content_hashes
            for i in range(0, len(hashes), fanout):
                chunk = hashes[i : i + fanout]
                batch = await asyncio.gather(*[vectorize_one(h) for h in chunk])
                results.extend(batch)

        processed = 0
        skipped = 0
        total_rows = 0
        for result in results:
            st = result.get("status", "")
            if st == "processed":
                processed += 1
            elif st == "skipped":
                skipped += 1
            total_rows += int(result.get("rows", 0))

        n = len(input.content_hashes)
        await workflow.execute_activity(
            "log_workflow_summary",
            LogWorkflowSummaryInput(
                strategy_id=input.strategy_id,
                total=n,
                skipped=skipped,
                processed=processed,
            ),
            start_to_close_timeout=timedelta(seconds=30),
        )
        return {
            "total_hashes": n,
            "processed": processed,
            "skipped": skipped,
            "total_rows": total_rows,
        }

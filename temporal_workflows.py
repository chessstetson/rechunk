"""
Temporal workflows for ReChunk. Sandbox-safe: no I/O, no activity/LLM imports.

Workflows only orchestrate; they call activities by name and pass serializable input.
"""

import asyncio
from dataclasses import dataclass
from datetime import timedelta

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
class FilesystemCorpusIngestInput:
    """
    Ingest-only workflow: path to snapshot JSON (see :mod:`rechunk.ingest_snapshot`).
    Worker on task queue ``rechunk-ingest`` executes the activity.
    """

    ingest_snapshot_path: str


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

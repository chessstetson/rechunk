"""
Temporal client helpers: filesystem ingest workflow vs document vectorization (Phase C).

Ingest (ECS + manifest) and vectorization are **decoupled**: run
:class:`~temporal_workflows.FilesystemCorpusIngestWorkflow` on ``rechunk-ingest`` first, then start
:class:`~temporal_workflows.BatchDocumentVectorizationWorkflow` on ``rechunk-strategy-chunking``.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

from rechunk.fingerprints import compute_strategy_fingerprint
from rechunk.index_service import IndexService
from rechunk.strategies import Strategy, strategy_to_dict
from rechunk.temporal_queues import TASK_QUEUE_INGEST, TASK_QUEUE_VECTORIZATION
from rechunk.vector_store import FilesystemVectorStore
from rechunk.repo_paths import project_root
from rechunk.vectorization_config import VECTOR_SCHEMA_VERSION

# Default strategies file (run_with_docs uses same path).
_DEFAULT_STRATEGIES_PATH = project_root() / "rechunk_strategies.json"


def _temporal_address(explicit: str | None = None) -> str:
    return (explicit or os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")).strip()


async def _enqueue_batch_vectorization_for_strategy(
    client: "object",  # temporalio.client.Client
    strategy: Strategy,
    strategies_path: Path | None,
) -> str | None:
    """
    Start :class:`BatchDocumentVectorizationWorkflow` for all ECS-active hashes pending for ``strategy``.

    Returns workflow id, or ``None`` if nothing to do.
    """
    from temporalio.common import WorkflowIDReusePolicy

    from rechunk.extracted_content import FilesystemExtractedContentService

    from temporal_vectorization_inputs import BatchDocumentVectorizationInput
    from temporal_workflows import BatchDocumentVectorizationWorkflow

    spath = strategies_path or _DEFAULT_STRATEGIES_PATH
    ecs = FilesystemExtractedContentService()
    vs = FilesystemVectorStore()
    index = IndexService(ecs=ecs, vector_store=vs, strategies_path=spath)

    efp = index.embedding_fingerprint()
    pending = index.list_pending_vectorization([strategy])
    if not pending:
        print(
            f"  No pending vectorization for strategy {strategy.id!r} (all active docs already have rows).",
            file=sys.stderr,
        )
        return None

    hashes = sorted({item.content_hash for item in pending})
    sd = strategy_to_dict(strategy)
    sfp = compute_strategy_fingerprint(sd)
    batch_inp = BatchDocumentVectorizationInput(
        content_hashes=hashes,
        strategy_id=str(strategy.id),
        strategy_definition=sd,
        strategy_fingerprint=sfp,
        embedding_fingerprint=efp,
        vector_schema_version=VECTOR_SCHEMA_VERSION,
    )
    wid = f"rechunk-batch-v12-{strategy.id}-{uuid.uuid4().hex[:12]}"
    await client.start_workflow(
        BatchDocumentVectorizationWorkflow,
        batch_inp,
        id=wid,
        task_queue=TASK_QUEUE_VECTORIZATION,
        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
    )
    print(
        f"  Started BatchDocumentVectorizationWorkflow {wid!r} ({len(hashes)} hash(es)) "
        f"for strategy {strategy.id!r}. Worker: {TASK_QUEUE_VECTORIZATION!r}.",
        file=sys.stderr,
        flush=True,
    )
    return wid


def trigger_pending_vectorization_sync(
    strategy: Strategy,
    *,
    temporal_address: str | None = None,
    strategies_path: Path | None = None,
) -> str | None:
    """
    Enqueue vectorization for one strategy over the **current ECS active corpus** (no docs path).

    Use after adding a strategy in ``--ecs`` / manifest mode, or anytime you need to backfill
    VectorStore rows. Requires a worker on ``TASK_QUEUE_VECTORIZATION``.
    """
    try:
        from temporalio.client import Client
    except ImportError:
        print(
            "  (Temporal not available; run: python scripts/start_strategy_chunking.py "
            f"{strategy.id!r})",
            file=sys.stderr,
        )
        return None

    addr = _temporal_address(temporal_address)

    async def _run() -> str | None:
        client = await Client.connect(addr)
        return await _enqueue_batch_vectorization_for_strategy(client, strategy, strategies_path)

    try:
        return asyncio.run(_run())
    except Exception as e:
        print(
            f"  Could not start vectorization workflow: {e}. "
            f"Try: python scripts/start_strategy_chunking.py {strategy.id!r}",
            file=sys.stderr,
        )
        return None


def trigger_filesystem_ingest_sync(
    docs_root: Path,
    doc_ids: list[str],
    *,
    temporal_address: str = "localhost:7233",
    wait_for_result: bool = False,
) -> str | None:
    """
    Write an ingest snapshot and start :class:`FilesystemCorpusIngestWorkflow` on the ingest queue.

    Returns workflow id, or ``None`` if Temporal is unavailable. Requires an **ingest** worker
    polling ``TASK_QUEUE_INGEST``.
    """
    try:
        from temporalio.client import Client

        from rechunk.ingest_snapshot import build_and_write_ingest_snapshot

        from temporal_workflows import FilesystemCorpusIngestInput, FilesystemCorpusIngestWorkflow
    except ImportError:
        print(
            "  (Temporal not available; use IndexService.ingest locally or install temporalio.)",
            file=sys.stderr,
        )
        return None

    snap = build_and_write_ingest_snapshot(
        docs_root.resolve(),
        doc_ids,
        strategy_id="ingest",
    )

    async def _run() -> str:
        client = await Client.connect(temporal_address)
        wid = f"rechunk-ingest-{uuid.uuid4().hex[:12]}"
        handle = await client.start_workflow(
            FilesystemCorpusIngestWorkflow,
            FilesystemCorpusIngestInput(ingest_snapshot_path=str(snap)),
            id=wid,
            task_queue=TASK_QUEUE_INGEST,
        )
        if wait_for_result:
            result = await handle.result()
            print(f"  Ingest workflow {wid} completed: {result}", file=sys.stderr, flush=True)
        else:
            print(
                f"  Started ingest workflow {wid} (snapshot {snap.name}). "
                f"Ingest worker must poll {TASK_QUEUE_INGEST!r}.",
                file=sys.stderr,
                flush=True,
        )
        return wid

    try:
        return asyncio.run(_run())
    except Exception as e:
        print(f"  Could not start ingest workflow: {e}", file=sys.stderr)
        return None


def trigger_strategy_chunking_sync(
    docs_root: Path,
    doc_ids: list[str],
    strategy: Strategy,
    *,
    temporal_address: str | None = None,
    strategies_path: Path | None = None,
    ingest_first: bool = False,
) -> None:
    """
    Start one :class:`BatchDocumentVectorizationWorkflow` for all pending content hashes (single strategy).

    Does **not** ingest into ECS unless ``ingest_first=True`` (escape hatch). Normally ingest runs
    separately via :func:`trigger_filesystem_ingest_sync` or ``scripts/start_corpus_ingest.py``.
    """
    try:
        from temporalio.client import Client
    except ImportError:
        print(
            "  (Temporal not available; run scripts/start_strategy_chunking.py manually to backfill.)",
            file=sys.stderr,
        )
        return

    addr = _temporal_address(temporal_address)

    async def _run() -> None:
        client = await Client.connect(addr)
        if ingest_first:
            from rechunk.ingest_snapshot import build_and_write_ingest_snapshot

            from temporal_workflows import FilesystemCorpusIngestInput, FilesystemCorpusIngestWorkflow

            snap = build_and_write_ingest_snapshot(
                docs_root.resolve(),
                doc_ids,
                strategy_id="ingest",
            )
            iwid = f"rechunk-ingest-inline-{uuid.uuid4().hex[:10]}"
            h = await client.start_workflow(
                FilesystemCorpusIngestWorkflow,
                FilesystemCorpusIngestInput(ingest_snapshot_path=str(snap)),
                id=iwid,
                task_queue=TASK_QUEUE_INGEST,
            )
            await h.result()

        await _enqueue_batch_vectorization_for_strategy(client, strategy, strategies_path)

    try:
        asyncio.run(_run())
    except Exception as e:
        print(
            f"  Could not start workflow(s): {e}. Run scripts/start_strategy_chunking.py manually if needed.",
            file=sys.stderr,
        )

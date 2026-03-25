"""
Temporal workers for ReChunk (ingest vs vectorization on separate task queues).

Run with a Temporal server (e.g. ``temporal server start-dev``), set ``OPENAI_API_KEY`` for the
vectorization worker, then::

  python temporal_workers.py              # both workers / both queues (local dev default)
  python temporal_workers.py ingest       # ECS ingest only (no embed model required)
  python temporal_workers.py vectorization
  python temporal_workers.py both

Override with env ``RECHUNK_TEMPORAL_WORKER_ROLE`` (ingest | vectorization | both).

Phase C vectorization activities use :func:`rechunk.worker_runtime.configure_worker_runtime`
(ECS + VectorStore on local disk by default).
"""

from __future__ import annotations

import asyncio
import os
import sys

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

from rechunk.extracted_content import FilesystemExtractedContentService
from rechunk.temporal_queues import TASK_QUEUE_INGEST, TASK_QUEUE_VECTORIZATION
from rechunk.vector_store import FilesystemVectorStore
from rechunk.vectorization_config import OPENAI_EMBEDDING_MODEL
from rechunk.worker_runtime import configure_worker_runtime

from temporalio.client import Client
from temporalio.worker import Worker

from temporal_activities import (
    chunk_doc_with_builtin_splitter,
    chunk_doc_with_strategy,
    get_cached_hashes,
    ingest_filesystem_corpus_from_snapshot,
    load_doc_manifest,
    load_manifest_from_ingest_snapshot,
    log_workflow_summary,
    merge_active_corpus_manifest,
    vectorize_content_for_strategy,
)
from temporal_workflows import (
    BatchDocumentVectorizationWorkflow,
    DocumentVectorizationWorkflow,
    FilesystemCorpusIngestWorkflow,
    StrategyChunkingWorkflow,
)


def _max_concurrent_activities() -> int:
    """
    Cap simultaneous activity executions per vectorization worker process.

    OpenAI / disk-heavy vectorization benefits from >1 but not unbounded parallelism.
    Set ``RECHUNK_MAX_CONCURRENT_ACTIVITIES`` (integer). When unset or invalid, defaults to ``8``.
    """
    raw = os.environ.get("RECHUNK_MAX_CONCURRENT_ACTIVITIES", "").strip()
    if not raw:
        return 8
    try:
        n = int(raw, 10)
    except ValueError:
        return 8
    return max(1, n)


def _roles_from_argv_and_env() -> set[str]:
    env = os.environ.get("RECHUNK_TEMPORAL_WORKER_ROLE", "").strip().lower()
    if env in ("ingest", "vectorization", "both"):
        raw = env
    elif len(sys.argv) > 1:
        raw = sys.argv[1].strip().lower()
    else:
        raw = "both"
    if raw == "both":
        return {"ingest", "vectorization"}
    if raw in ("ingest", "vectorization"):
        return {raw}
    raise SystemExit(
        "Usage: python temporal_workers.py [ingest|vectorization|both]\n"
        "Or set RECHUNK_TEMPORAL_WORKER_ROLE to ingest, vectorization, or both."
    )


async def _run_ingest_worker(client: Client) -> None:
    worker = Worker(
        client,
        task_queue=TASK_QUEUE_INGEST,
        workflows=[FilesystemCorpusIngestWorkflow],
        activities=[ingest_filesystem_corpus_from_snapshot],
    )
    await worker.run()


async def _run_vectorization_worker(client: Client) -> None:
    embed_model_name = os.environ.get("RECHUNK_OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=embed_model_name)
    ecs = FilesystemExtractedContentService()
    vs = FilesystemVectorStore(embed_model=Settings.embed_model)
    configure_worker_runtime(ecs, vs)

    max_act = _max_concurrent_activities()
    print(
        f"  max_concurrent_activities={max_act} (override with RECHUNK_MAX_CONCURRENT_ACTIVITIES)",
        file=sys.stderr,
        flush=True,
    )
    worker = Worker(
        client,
        task_queue=TASK_QUEUE_VECTORIZATION,
        workflows=[
            StrategyChunkingWorkflow,
            DocumentVectorizationWorkflow,
            BatchDocumentVectorizationWorkflow,
        ],
        activities=[
            chunk_doc_with_strategy,
            chunk_doc_with_builtin_splitter,
            load_doc_manifest,
            load_manifest_from_ingest_snapshot,
            vectorize_content_for_strategy,
            get_cached_hashes,
            log_workflow_summary,
            merge_active_corpus_manifest,
        ],
        max_concurrent_activities=max_act,
    )
    await worker.run()


async def main() -> None:
    roles = _roles_from_argv_and_env()
    addr = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    client = await Client.connect(addr)

    if roles == {"ingest", "vectorization"}:
        print(
            f"ReChunk Temporal: TWO Worker instances in ONE process (dev default), "
            f"connected to {addr!r}:\n"
            f"  - queue {TASK_QUEUE_INGEST!r}  → ingest only (no OpenAI embed on this side)\n"
            f"  - queue {TASK_QUEUE_VECTORIZATION!r} → chunk + embed + VectorStore\n"
            f"For TWO separate OS processes, use two terminals:\n"
            f"  python temporal_workers.py ingest\n"
            f"  python temporal_workers.py vectorization",
            file=sys.stderr,
            flush=True,
        )
        await asyncio.gather(
            _run_ingest_worker(client),
            _run_vectorization_worker(client),
        )
    elif roles == {"ingest"}:
        print(
            f"ReChunk Temporal: one Worker on {TASK_QUEUE_INGEST!r} @ {addr!r} (ingest only).",
            file=sys.stderr,
            flush=True,
        )
        await _run_ingest_worker(client)
    else:
        print(
            f"ReChunk Temporal: one Worker on {TASK_QUEUE_VECTORIZATION!r} @ {addr!r} (vectorization).",
            file=sys.stderr,
            flush=True,
        )
        await _run_vectorization_worker(client)


if __name__ == "__main__":
    asyncio.run(main())

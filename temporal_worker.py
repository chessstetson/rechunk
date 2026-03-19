"""
Temporal worker for ReChunk: runs StrategyChunkingWorkflow and chunking activities.

Run with a Temporal server (e.g. temporal server start-dev), set OPENAI_API_KEY, then:
  python temporal_worker.py
"""

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from temporal_activities import (
    chunk_doc_with_builtin_splitter,
    chunk_doc_with_strategy,
    get_cached_hashes,
    load_doc_manifest,
    log_workflow_summary,
)
from temporal_workflows import StrategyChunkingWorkflow


TASK_QUEUE = "rechunk-strategy-chunking"


async def main() -> None:
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[StrategyChunkingWorkflow],
        activities=[
            chunk_doc_with_strategy,
            chunk_doc_with_builtin_splitter,
            load_doc_manifest,
            get_cached_hashes,
            log_workflow_summary,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

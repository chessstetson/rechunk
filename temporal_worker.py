import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from temporal_activities import chunk_doc_with_strategy, chunk_doc_with_builtin_splitter
from temporal_workflows import StrategyChunkingWorkflow


TASK_QUEUE = "rechunk-strategy-chunking"


async def main() -> None:
    """Temporal worker entrypoint.

    Connects to a Temporal server and hosts the StrategyChunkingWorkflow and
    chunk_doc_with_strategy Activity. This is a minimal v1 scaffold; we may
    later add configuration (namespaces, TLS, etc.).
    """
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[StrategyChunkingWorkflow],
        activities=[chunk_doc_with_strategy, chunk_doc_with_builtin_splitter],
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())


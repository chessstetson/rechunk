"""
Temporal client helpers: start chunking workflows from synchronous callers (e.g. CLI).

Keeps Temporal-specific imports and asyncio bridging out of ``run_with_docs.py``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rechunk.strategies import Strategy

# Must match ``temporal_worker.py`` task queue for strategy chunking.
TASK_QUEUE_STRATEGY_CHUNKING = "rechunk-strategy-chunking"


def trigger_strategy_chunking_sync(
    docs_root: Path,
    doc_ids: list[str],
    strategy: Strategy,
    *,
    temporal_address: str = "localhost:7233",
) -> None:
    """
    Start the strategy chunking workflow so the worker backfills cache for ``strategy``.

    If Temporal packages or the server are unavailable, prints a hint to stderr and returns.
    """
    try:
        from temporalio.client import Client
        from temporal_workflows import StrategyChunkingInput, StrategyChunkingWorkflow
    except ImportError:
        print(
            "  (Temporal not available; run scripts/start_strategy_chunking.py manually to backfill.)",
            file=sys.stderr,
        )
        return

    async def _run() -> None:
        client = await Client.connect(temporal_address)
        workflow_id = f"rechunk-{strategy.id}"
        await client.start_workflow(
            StrategyChunkingWorkflow,
            StrategyChunkingInput(
                strategy_id=strategy.id,
                kind=strategy.kind,
                docs_root=str(docs_root.resolve()),
                doc_ids=doc_ids,
                strategy_instruction=strategy.instruction if strategy.kind == "llm" else None,
                model=getattr(strategy, "model", None),
                splitter=getattr(strategy, "splitter", "sentence"),
            ),
            id=workflow_id,
            task_queue=TASK_QUEUE_STRATEGY_CHUNKING,
        )
        print(
            f"  Started workflow {workflow_id}. Worker will backfill chunks for strategy {strategy.id!r}.",
            file=sys.stderr,
        )

    try:
        asyncio.run(_run())
    except Exception as e:
        print(
            f"  Could not start workflow: {e}. Run scripts/start_strategy_chunking.py manually if needed.",
            file=sys.stderr,
        )

"""
Temporal workflows for ReChunk. Sandbox-safe: no I/O, no activity/LLM imports.

Workflows only orchestrate; they call activities by name and pass serializable input.
"""

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

from temporalio import workflow


# Mirror of ChunkDocInput for workflow use (workflow code cannot import temporal_activities).
@dataclass
class ChunkDocInput:
    strategy_id: str
    strategy_instruction: str
    model: Optional[str]
    docs_root: str
    doc_id: str
    content_hash: str


@dataclass
class StrategyChunkingInput:
    """Input for StrategyChunkingWorkflow. One strategy, many docs."""

    strategy_id: str
    strategy_instruction: str
    model: Optional[str]
    docs_root: str
    doc_ids: List[str]


@workflow.defn
class StrategyChunkingWorkflow:
    """Chunk all documents for one strategy. Phase 1: sequential execution."""

    @workflow.run
    async def run(self, input: StrategyChunkingInput) -> None:
        for doc_id in input.doc_ids:
            await workflow.execute_activity(
                "chunk_doc_with_strategy",
                ChunkDocInput(
                    strategy_id=input.strategy_id,
                    strategy_instruction=input.strategy_instruction,
                    model=input.model,
                    docs_root=input.docs_root,
                    doc_id=doc_id,
                    content_hash="",  # activity computes from doc text when empty
                ),
                start_to_close_timeout=timedelta(minutes=5),
            )

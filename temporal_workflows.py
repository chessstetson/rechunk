import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

from temporalio import workflow


@dataclass
class StrategyChunkingInput:
    """Workflow input describing a single strategy + corpus.

    For v1 we keep this intentionally simple: the Workflow receives a docs_root
    and an explicit list of doc_ids (paths relative to that root). Later we can
    factor out a manifest-loading Activity.
    """

    strategy_id: str
    kind: str
    strategy_instruction: str
    model: Optional[str]
    splitter: Optional[str]
    docs_root: str
    doc_ids: List[str]


@dataclass
class ChunkDocInput:
    """Activity input definition kept sandbox-safe for workflows.

    NOTE: This mirrors the structure used in the Activity module but avoids
    importing non-sandbox-safe libraries into workflow code. The Activity
    implementation lives in temporal_activities.py and is referenced by name.
    """

    strategy_id: str
    strategy_instruction: str
    model: Optional[str]
    docs_root: str
    doc_id: str


@dataclass
class BuiltinChunkInput:
    """Activity input for built-in splitters kept workflow-safe."""

    strategy_id: str
    splitter: str
    docs_root: str
    doc_id: str


@workflow.defn
class StrategyChunkingWorkflow:
    """Orchestrate chunking for a single strategy across a set of documents.

    v1: sequential execution. A later phase will fan out Activities in
    parallel and add skip-if-cached logic.
    """

    @workflow.run
    async def run(self, input: StrategyChunkingInput) -> None:
        tasks = []
        for doc_id in input.doc_ids:
            if input.kind == "builtin_splitter":
                tasks.append(
                    workflow.execute_activity(
                        "chunk_doc_with_builtin_splitter",
                        BuiltinChunkInput(
                            strategy_id=input.strategy_id,
                            splitter=input.splitter or "sentence",
                            docs_root=input.docs_root,
                            doc_id=doc_id,
                        ),
                        start_to_close_timeout=timedelta(minutes=5),
                    )
                )
                continue

            tasks.append(
                workflow.execute_activity(
                    "chunk_doc_with_strategy",
                    ChunkDocInput(
                        strategy_id=input.strategy_id,
                        strategy_instruction=input.strategy_instruction,
                        model=input.model,
                        docs_root=input.docs_root,
                        doc_id=doc_id,
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                )
            )

        # For v1 we simply await all tasks; the Temporal SDK will sequence them.
        # In a later phase we may choose to run them truly in parallel with
        # asyncio.gather and tune the concurrency.
        await asyncio.gather(*tasks)


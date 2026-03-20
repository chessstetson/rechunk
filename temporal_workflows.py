"""
Temporal workflows for ReChunk. Sandbox-safe: no I/O, no activity/LLM imports.

Workflows only orchestrate; they call activities by name and pass serializable input.
"""

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

from temporalio import workflow


# Mirror activity input dataclasses (workflow cannot import temporal_activities).
@dataclass
class LogWorkflowSummaryInput:
    strategy_id: str
    total: int
    skipped: int
    processed: int


@dataclass
class LoadDocManifestInput:
    docs_root: str
    doc_ids: List[str]


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
class StrategyChunkingInput:
    """Input for StrategyChunkingWorkflow. One strategy, many docs. Supports LLM and built-in."""

    strategy_id: str
    kind: str  # "llm" or "builtin_splitter"
    docs_root: str
    doc_ids: List[str]
    strategy_instruction: Optional[str] = None  # for kind=llm
    model: Optional[str] = None  # for kind=llm
    splitter: str = "sentence"  # for kind=builtin_splitter: "sentence" or "token"


@workflow.defn
class StrategyChunkingWorkflow:
    """Chunk documents for one strategy. Phase 2: skip-if-cached + parallel execution."""

    @workflow.run
    async def run(self, input: StrategyChunkingInput) -> dict:
        # 2.1: Load manifest (doc_id -> content_hash) and cached hashes; skip already-cached docs.
        manifest: List[dict] = await workflow.execute_activity(
            "load_doc_manifest",
            LoadDocManifestInput(docs_root=input.docs_root, doc_ids=input.doc_ids),
            start_to_close_timeout=timedelta(minutes=10),
        )
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
                            docs_root=input.docs_root,
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
                            docs_root=input.docs_root,
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

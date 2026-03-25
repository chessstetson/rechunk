#!/usr/bin/env python3
"""
Start **one** :class:`BatchDocumentVectorizationWorkflow` for all pending ECS hashes (single strategy).

Each hash is its own **activity**; the workflow runs them in **waves** of parallel
``asyncio.gather`` (size from ``RECHUNK_BATCH_VECTORIZATION_FANOUT``, default 32) so workflow
tasks do not time out. Within a wave, ``max_concurrent_activities`` on the worker caps concurrency.

Set ``RECHUNK_BATCH_WORKFLOW_TASK_TIMEOUT_SECONDS`` if workflow tasks still time out (default 600s).

Uses **ECS active hashes** vs **VectorStore** row presence only — no corpus path argument.
Run ``scripts/start_corpus_ingest.py <docs_root>`` first so ECS is populated.

Usage:
  python scripts/start_strategy_chunking.py <strategy_id> [options]

Requires Temporal server, **long-running** ``python temporal_workers.py`` in another terminal (that
process **executes** activities; this script only **starts** a workflow on the server), and
``OPENAI_API_KEY`` (embeddings always use OpenAI unless you change the worker).

:class:`DocumentVectorizationWorkflow` (one workflow per hash) remains available for advanced use;
this script uses the batch workflow only.
"""

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

# Project root and ``src`` on path (package lives under ``src/``).
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from temporalio.client import Client
from temporalio.common import WorkflowIDReusePolicy

from rechunk.extracted_content import FilesystemExtractedContentService
from rechunk.fingerprints import compute_strategy_fingerprint
from rechunk.index_service import (
    IndexService,
    build_strategy_from_cli,
    resolve_strategy_for_job,
)
from rechunk.strategies import DEFAULT_BASELINE_STRATEGY, Strategy, strategy_to_dict
from rechunk.temporal_queues import TASK_QUEUE_VECTORIZATION
from rechunk.vector_store import FilesystemVectorStore
from rechunk.vectorization_config import (
    VECTOR_SCHEMA_VERSION,
    batch_vectorization_fanout_batch_size,
    batch_vectorization_workflow_task_timeout,
)
from temporal_vectorization_inputs import BatchDocumentVectorizationInput
from temporal_workflows import BatchDocumentVectorizationWorkflow

STRATEGIES_FILE = _project_root / "rechunk_strategies.json"

DEFAULT_INSTRUCTION = (
    "Identify all named entities (people, places, organizations) and create one chunk "
    "per entity, including all text that directly describes or relates to that entity."
)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start one batch vectorization workflow from ECS + VectorStore pending diff."
    )
    parser.add_argument("strategy_id", help="Strategy ID (e.g. s_entities or s_default)")
    parser.add_argument(
        "--kind",
        choices=("llm", "builtin", "derived"),
        default="builtin",
        help=(
            "Fallback if strategy_id is missing from rechunk_strategies.json: llm, derived, or builtin. "
            "If the id exists in the file, that file entry is used (see printed kind below)."
        ),
    )
    parser.add_argument(
        "--splitter",
        choices=("sentence", "token"),
        default="sentence",
        help="For --kind builtin: sentence (default) or token",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="For --kind llm: chunking instruction")
    parser.add_argument("--address", default="localhost:7233", help="Temporal server address")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until the batch workflow completes",
    )
    args = parser.parse_args()

    kind = "builtin_splitter" if args.kind == "builtin" else "llm"
    if kind == "llm":
        cli_strategy = build_strategy_from_cli(
            strategy_id=args.strategy_id,
            kind=kind,
            instruction=args.instruction,
            splitter=args.splitter,
            model=None,
        )
    else:
        # Must match DEFAULT_BASELINE_STRATEGY / run_with_docs when no JSON file exists,
        # or VectorStore row paths (strategy fingerprint includes full strategy dict).
        cli_strategy = Strategy(
            id=args.strategy_id,
            kind="builtin_splitter",
            instruction=DEFAULT_BASELINE_STRATEGY.instruction,
            splitter=args.splitter,
        )
    strategy = resolve_strategy_for_job(
        strategies_path=STRATEGIES_FILE,
        strategy_id=args.strategy_id,
        cli_strategy=cli_strategy,
    )
    if strategy.kind == "derived":
        chunking_mode = "derived nodes (LLM synthetic text + source_spans; slow)"
    elif strategy.kind == "llm":
        chunking_mode = "LLM verbatim chunking (slow: one strategy call per doc)"
    else:
        chunking_mode = (
            f"built-in splitter ({getattr(strategy, 'splitter', 'sentence')!r}, no chunking LLM)"
        )
    print(
        f"Resolved strategy {strategy.id!r}: kind={strategy.kind!r} → {chunking_mode}.",
        flush=True,
    )

    ecs = FilesystemExtractedContentService()
    vs = FilesystemVectorStore()
    index = IndexService(ecs=ecs, vector_store=vs, strategies_path=STRATEGIES_FILE)
    active = ecs.list_active_hashes()
    if not active:
        print(
            "ECS has no active content hashes. Ingest first, e.g.\n"
            "  python scripts/start_corpus_ingest.py <docs_root> --wait",
            file=sys.stderr,
        )
        sys.exit(1)

    efp = index.embedding_fingerprint()
    pending = index.list_pending_vectorization([strategy])

    if not pending:
        sfp = compute_strategy_fingerprint(strategy_to_dict(strategy))
        rows_prefix = (
            vs.root / "rows" / sfp / efp / VECTOR_SCHEMA_VERSION
        )
        print(
            f"Nothing to vectorize for strategy {strategy.id!r}: "
            f"all {len(active)} ECS-active content hash(es) already have VectorStore row bundles "
            f"for this strategy + embedding model.",
            flush=True,
        )
        print(
            "\n"
            "Why: chunking is keyed by **content hash**, not ingest run. If these are the same "
            "`.txt` bytes as an earlier experiment (e.g. same Wikipedia sample with default `--seed`), "
            "existing rows under `storage/vector_store_dev/rows/` still apply — "
            "`start_strategy_chunking` is correctly a no-op.\n"
            "\n"
            "To actually re-embed: delete the row bundles for this strategy (or all of "
            f"`{vs.root / 'rows'}`), then run this script again.\n"
            "To get **new** hashes without wiping the store: use different source files or "
            "`python scripts/prepare_hf_benchmark_corpus.py wikipedia --seed <other>`.\n"
            f"\nRow bundle directory for this strategy/embed/schema:\n  {rows_prefix}\n",
            flush=True,
        )
        return

    hashes = sorted({item.content_hash for item in pending})
    sd = strategy_to_dict(strategy)
    sfp = compute_strategy_fingerprint(sd)
    batch_inp = BatchDocumentVectorizationInput(
        content_hashes=hashes,
        strategy_id=strategy.id,
        strategy_definition=sd,
        strategy_fingerprint=sfp,
        embedding_fingerprint=efp,
        vector_schema_version=VECTOR_SCHEMA_VERSION,
        fanout_batch_size=batch_vectorization_fanout_batch_size(),
    )
    wid = f"rechunk-batch-v12-{strategy.id}-{uuid.uuid4().hex[:12]}"

    print(
        f"ECS: {len(active)} active hash(es); "
        f"starting 1 batch workflow for {len(hashes)} pending hash(es), strategy {strategy.id!r}.\n"
        f"  workflow_id={wid}",
        flush=True,
    )

    client = await Client.connect(args.address)
    handle = await client.start_workflow(
        BatchDocumentVectorizationWorkflow,
        batch_inp,
        id=wid,
        task_queue=TASK_QUEUE_VECTORIZATION,
        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        task_timeout=batch_vectorization_workflow_task_timeout(),
    )

    print(
        "—\n"
        "Scheduled the workflow on Temporal only. Actual chunk+embed work runs in another process:\n"
        "  python temporal_workers.py\n"
        "Keep that running (polls queue 'rechunk-strategy-chunking'). Without it, the workflow waits.\n"
        "Built-in splitting is still slow for large corpora: every chunk is embedded via OpenAI API.\n"
        "—",
        flush=True,
    )
    if args.wait:
        result = await handle.result()
        print(f"Batch completed: {result}")


if __name__ == "__main__":
    asyncio.run(main())

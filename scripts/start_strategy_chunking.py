#!/usr/bin/env python3
"""
Start a StrategyChunkingWorkflow for any strategy (LLM or built-in). Worker must be running.

Usage:
  python scripts/start_strategy_chunking.py <docs_root> <strategy_id> [options]

Example (LLM — extract named entities):
  python scripts/start_strategy_chunking.py ./docs s_entities

Example (built-in SentenceSplitter):
  python scripts/start_strategy_chunking.py ./docs s_default --kind builtin --splitter sentence

Example (built-in TokenTextSplitter):
  python scripts/start_strategy_chunking.py ./docs s_token --kind builtin --splitter token

Example (wait until workflow finishes):
  python scripts/start_strategy_chunking.py ./docs s_entities --wait

Requires Temporal server (e.g. temporal server start-dev) and a worker (python temporal_worker.py).
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temporalio.client import Client
from temporal_workflows import StrategyChunkingInput, StrategyChunkingWorkflow


TASK_QUEUE = "rechunk-strategy-chunking"

# Default: extract named entities (people, places, organizations) — one chunk per entity.
DEFAULT_INSTRUCTION = (
    "Identify all named entities (people, places, organizations) and create one chunk "
    "per entity, including all text that directly describes or relates to that entity."
)


def get_doc_ids(docs_root: Path) -> list[str]:
    """Relative paths for supported files under docs_root."""
    exts = {".txt", ".md", ".pdf", ".docx"}
    ids = []
    for f in sorted(docs_root.rglob("*")):
        if f.is_file() and f.suffix.lower() in exts:
            ids.append(str(f.relative_to(docs_root)))
    return ids


async def main() -> None:
    parser = argparse.ArgumentParser(description="Start ReChunk strategy chunking workflow")
    parser.add_argument("docs_root", type=Path, help="Root directory of documents")
    parser.add_argument("strategy_id", help="Strategy ID (e.g. s_entities or s_default)")
    parser.add_argument(
        "--kind",
        choices=("llm", "builtin"),
        default="llm",
        help="Strategy kind: llm (default) or builtin (Sentence/Token splitter)",
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
        help="Wait for the workflow to finish, then print 'Workflow completed.' (run worker in another terminal)",
    )
    args = parser.parse_args()

    docs_root = args.docs_root.resolve()
    if not docs_root.is_dir():
        print(f"Not a directory: {docs_root}", file=sys.stderr)
        sys.exit(1)

    doc_ids = get_doc_ids(docs_root)
    if not doc_ids:
        print(f"No .txt/.md/.pdf/.docx files under {docs_root}", file=sys.stderr)
        sys.exit(1)

    kind = "builtin_splitter" if args.kind == "builtin" else "llm"
    client = await Client.connect(args.address)
    workflow_id = f"rechunk-{args.strategy_id}"
    handle = await client.start_workflow(
        StrategyChunkingWorkflow,
        StrategyChunkingInput(
            strategy_id=args.strategy_id,
            kind=kind,
            docs_root=str(docs_root),
            doc_ids=doc_ids,
            strategy_instruction=args.instruction if kind == "llm" else None,
            model=None,
            splitter=args.splitter,
        ),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    print(f"Started workflow {workflow_id} ({len(doc_ids)} docs). Run the worker to process.")
    if args.wait:
        result = await handle.result()
        print("Workflow completed.")
        if isinstance(result, dict):
            print(
                f"  {result.get('processed', 0)} chunked, "
                f"{result.get('skipped', 0)} already cached, "
                f"{result.get('total', 0)} total."
            )


if __name__ == "__main__":
    asyncio.run(main())

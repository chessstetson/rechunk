#!/usr/bin/env python3
"""
Interactive entrypoint: **corpus path → local ECS ingest → queue initial embeddings** (Temporal), then Q&A.

Delegates to ``run_with_docs.py --interactive``. It **does not** prompt for your first question until
the vector index has at least one chunk (embeddings from the worker / cache).

Requires ``OPENAI_API_KEY`` and **``python temporal_workers.py``** so vectorization completes.

With no corpus argument, you can **press Enter** or type **demo** to opt in to a small English
Wikipedia subset (Hugging Face + ``pip install -e '.[benchmark-corpora]'``), or type your own path.

Examples::

    python scripts/run_interactive.py
    python scripts/run_interactive.py path/to/docs

Before running, checks that Temporal is reachable at ``TEMPORAL_ADDRESS`` (default
``localhost:7233``) and that workers are polling ReChunk's task queues; if not, prints
copy-paste steps and exits. Use ``--skip-temporal-check`` to bypass (advanced).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_RUN_WITH_DOCS = _SCRIPT_DIR / "run_with_docs.py"
_PREPARE_BENCHMARK = _SCRIPT_DIR / "prepare_hf_benchmark_corpus.py"

_DEV_NAMESPACE = "default"

# Same default out dir as ``prepare_hf_benchmark_corpus.py wikipedia`` (200 docs matches README presets).
_DEMO_N = 200


def _temporal_address() -> str:
    return (os.environ.get("TEMPORAL_ADDRESS") or "localhost:7233").strip()


def _ensure_src_on_path(project_root: Path) -> None:
    src = str((project_root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


async def _try_connect_temporal(addr: str, *, timeout: float) -> bool:
    try:
        from temporalio.client import Client

        await asyncio.wait_for(Client.connect(addr), timeout=timeout)
        return True
    except Exception:
        return False


async def _pollers_on_queue(client, queue_name: str) -> int:
    from temporalio.api.enums.v1 import TaskQueueType
    from temporalio.api.taskqueue.v1 import TaskQueue
    from temporalio.api.workflowservice.v1 import DescribeTaskQueueRequest

    total = 0
    for tqt in (
        TaskQueueType.TASK_QUEUE_TYPE_WORKFLOW,
        TaskQueueType.TASK_QUEUE_TYPE_ACTIVITY,
    ):
        req = DescribeTaskQueueRequest(
            namespace=_DEV_NAMESPACE,
            task_queue=TaskQueue(name=queue_name),
            task_queue_type=tqt,
            report_pollers=True,
        )
        try:
            resp = await client.workflow_service.describe_task_queue(
                req, timeout=timedelta(seconds=10)
            )
            total += len(resp.pollers)
        except Exception:
            continue
    return total


async def _rechunk_workers_poll_ready(project_root: Path, client) -> bool:
    _ensure_src_on_path(project_root)
    from rechunk.temporal_queues import TASK_QUEUE_INGEST, TASK_QUEUE_VECTORIZATION

    n_i = await _pollers_on_queue(client, TASK_QUEUE_INGEST)
    n_v = await _pollers_on_queue(client, TASK_QUEUE_VECTORIZATION)
    return n_i >= 1 and n_v >= 1


async def _preflight_temporal_async(project_root: Path, addr: str) -> str:
    """
    Returns ``\"ok\"`` if Temporal is up and ReChunk workers are polling both queues.
    Otherwise ``\"server\"`` or ``\"workers\"``.
    """
    try:
        from temporalio.client import Client

        client = await asyncio.wait_for(Client.connect(addr), timeout=3.0)
    except Exception:
        return "server"
    if not await _rechunk_workers_poll_ready(project_root, client):
        return "workers"
    return "ok"


def _print_temporal_server_help(addr: str) -> None:
    print(
        f"Cannot reach a Temporal server at {addr!r}.\n",
        file=sys.stderr,
    )
    print(
        "Start the dev server in its own terminal, then run this script again.\n\n"
        "  1. Install the Temporal CLI if needed: https://docs.temporal.io/cli\n\n"
        "  2. In a new terminal (any directory is fine):\n\n"
        "       temporal server start-dev\n\n"
        "     Default frontend: localhost:7233  (override with TEMPORAL_ADDRESS)\n\n"
        "  3. In another terminal, from the **repository root** with your venv active:\n\n"
        "       python temporal_workers.py\n\n"
        "     Set OPENAI_API_KEY in that environment for embeddings / vectorization.\n",
        file=sys.stderr,
    )


def _print_workers_help(addr: str) -> None:
    print(
        f"Temporal at {addr!r} is up, but no ReChunk workers are polling the "
        f"ingest / vectorization task queues.\n",
        file=sys.stderr,
    )
    print(
        "In a **new terminal**, from the **repository root** with your venv active:\n\n"
        "  python temporal_workers.py\n\n"
        "Ensure OPENAI_API_KEY is set (required for the vectorization worker).\n"
        "Then run this script again.\n",
        file=sys.stderr,
    )


def verify_temporal_ready(project_root: Path) -> int:
    """Return 0 if Temporal + workers look good; 1 after printing help."""
    addr = _temporal_address()
    status = asyncio.run(_preflight_temporal_async(project_root, addr))
    if status == "server":
        _print_temporal_server_help(addr)
        return 1
    if status == "workers":
        _print_workers_help(addr)
        return 1
    return 0


def _demo_corpus_dir(project_root: Path) -> Path:
    return project_root / "docs" / "benchmark_corpora" / "wikipedia"


def _ensure_demo_corpus(project_root: Path) -> Path | None:
    out = _demo_corpus_dir(project_root)
    if out.is_dir():
        existing = list(out.glob("*.txt"))
        if existing:
            print(f"Using existing demo corpus ({len(existing)} .txt): {out}")
            return out.resolve()

    print(
        "Demo corpus: downloading a small English Wikipedia subset (Hugging Face; needs network).\n"
        "If this fails, run:  pip install -e '.[benchmark-corpora]'"
    )
    cmd = [
        sys.executable,
        str(_PREPARE_BENCHMARK),
        "wikipedia",
        "--n",
        str(_DEMO_N),
    ]
    if subprocess.run(cmd, cwd=project_root).returncode != 0:
        print("Demo download failed (see errors above).", file=sys.stderr)
        return None
    if not out.is_dir() or not any(out.glob("*.txt")):
        print(f"Demo download did not produce .txt files under {out}", file=sys.stderr)
        return None
    return out.resolve()


def _prompt_corpus_path(project_root: Path) -> Path | None:
    demo = _demo_corpus_dir(project_root)
    print(
        "Corpus — enter a path to your documents (directory or file),\n"
        f"  or press Enter / type demo for a small Wikipedia subset → {demo}\n"
        "  (network + datasets; install: pip install -e '.[benchmark-corpora]')."
    )
    try:
        raw = input("Path (or Enter for demo): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.", file=sys.stderr)
        sys.exit(130)
    if not raw or raw.casefold() == "demo":
        return _ensure_demo_corpus(project_root)
    return Path(raw).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest a document tree into ECS, queue strategy embeddings (Temporal), "
            "wait until chunks exist, then prompt for questions interactively."
        )
    )
    parser.add_argument(
        "corpus",
        nargs="?",
        type=Path,
        default=None,
        help="Root directory or single file; omit for prompt (Enter/demo = Wikipedia subset)",
    )
    parser.add_argument("--model", default=None, help="OpenAI chat model (forwarded to run_with_docs)")
    parser.add_argument("--top-k", type=int, default=None, metavar="K", help="Retrieval top-k")
    parser.add_argument(
        "--no-vector-index-cache",
        action="store_true",
        help="Forward to run_with_docs (rebuild disk vector cache)",
    )
    parser.add_argument(
        "--skip-temporal-check",
        action="store_true",
        help="Do not verify Temporal server / ReChunk workers before running (advanced)",
    )
    args = parser.parse_args()

    if not args.skip_temporal_check:
        v = verify_temporal_ready(_PROJECT_ROOT)
        if v != 0:
            return v

    path = args.corpus
    if path is None:
        resolved = _prompt_corpus_path(_PROJECT_ROOT)
        if resolved is None:
            return 1
        path = resolved
    path = path.expanduser().resolve()
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    cmd: list[str] = [sys.executable, str(_RUN_WITH_DOCS), str(path), "--interactive"]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.no_vector_index_cache:
        cmd.append("--no-vector-index-cache")

    return int(subprocess.call(cmd, cwd=_PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())

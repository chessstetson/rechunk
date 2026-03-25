"""
Shared constants for ECS vectorization (Phase C) — client and worker must agree.
"""

from __future__ import annotations

import os
from datetime import timedelta

# Bump when VectorStore row / collection layout changes.
VECTOR_SCHEMA_VERSION = "v1"


def batch_vectorization_workflow_task_timeout() -> timedelta:
    """
    Start-to-close timeout for a **single workflow task** (replay + command generation).

    Large batches of parallel activities produce large history chunks; the default
    server limit (often 10s) is too low. Override with ``RECHUNK_BATCH_WORKFLOW_TASK_TIMEOUT_SECONDS``.
    """
    raw = os.environ.get("RECHUNK_BATCH_WORKFLOW_TASK_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return timedelta(minutes=10)
    try:
        sec = int(raw, 10)
    except ValueError:
        return timedelta(minutes=10)
    return max(timedelta(seconds=30), timedelta(seconds=min(sec, 3600)))


def batch_vectorization_fanout_batch_size() -> int:
    """
    How many ``vectorize_content_for_strategy`` activities the batch workflow awaits per wave.

    Smaller waves keep each workflow-task replay bounded (avoids workflow task timeouts).
    Override with ``RECHUNK_BATCH_VECTORIZATION_FANOUT`` (integer, capped at 256).
    """
    raw = os.environ.get("RECHUNK_BATCH_VECTORIZATION_FANOUT", "").strip()
    if not raw:
        return 32
    try:
        n = int(raw, 10)
    except ValueError:
        return 32
    return max(1, min(n, 256))

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDER_KIND_OPENAI = "openai"

"""
Process-global references for Temporal worker activities (Phase C).

Set via :func:`configure_worker_runtime` from ``temporal_workers.py`` before the worker runs.
Tests call the same configure function before driving activities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rechunk.extracted_content.protocol import ExtractedContentService
    from rechunk.vector_store.protocol import VectorStore

_runtime_ecs: Any = None
_runtime_vs: Any = None


def configure_worker_runtime(
    ecs: ExtractedContentService,
    vector_store: VectorStore,
) -> None:
    """Wire ECS and VectorStore for activities (idempotent for tests)."""
    global _runtime_ecs, _runtime_vs
    _runtime_ecs = ecs
    _runtime_vs = vector_store


def get_worker_ecs() -> ExtractedContentService:
    if _runtime_ecs is None:
        raise RuntimeError(
            "Worker runtime not configured: call configure_worker_runtime() from temporal_workers.py"
        )
    return _runtime_ecs


def get_worker_vector_store() -> VectorStore:
    if _runtime_vs is None:
        raise RuntimeError(
            "Worker runtime not configured: call configure_worker_runtime() from temporal_workers.py"
        )
    return _runtime_vs


def reset_worker_runtime() -> None:
    """Clear runtime (tests)."""
    global _runtime_ecs, _runtime_vs
    _runtime_ecs = None
    _runtime_vs = None

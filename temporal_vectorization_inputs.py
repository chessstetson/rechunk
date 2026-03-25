"""
Serializable workflow/activity payloads for ECS document vectorization (Phase C).

Standalone module (stdlib + dataclasses only) so Temporal workflow sandbox can import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DocumentVectorizationInput:
    """One document × one strategy; worker loads text from ECS."""

    content_hash: str
    strategy_id: str
    strategy_definition: dict[str, Any]
    strategy_fingerprint: str
    embedding_fingerprint: str
    vector_schema_version: str


@dataclass
class BatchDocumentVectorizationInput:
    """
    One workflow run: fan out ``vectorize_content_for_strategy`` for each ``content_hash``.

    Activities are started in waves of ``fanout_batch_size`` (``asyncio.gather`` per wave) so
    workflow-task replay stays bounded; within a wave, worker ``max_concurrent_activities`` limits
    how many run at once.

    Easier to inspect in Temporal UI than N separate workflow executions.
    """

    content_hashes: list[str]
    strategy_id: str
    strategy_definition: dict[str, Any]
    strategy_fingerprint: str
    embedding_fingerprint: str
    vector_schema_version: str
    fanout_batch_size: int = 32

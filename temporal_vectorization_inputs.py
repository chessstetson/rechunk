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
    One workflow run: call ``vectorize_content_for_strategy`` once per ``content_hash`` (sequential).

    Easier to inspect in Temporal UI than N separate workflow executions.
    """

    content_hashes: list[str]
    strategy_id: str
    strategy_definition: dict[str, Any]
    strategy_fingerprint: str
    embedding_fingerprint: str
    vector_schema_version: str

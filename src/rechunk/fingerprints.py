"""
Stable fingerprints for strategy definitions and embedding configuration (v12).

Used for :class:`VectorStore` row/collection keys and Temporal workflow payloads.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def compute_strategy_fingerprint(strategy_definition: dict[str, Any]) -> str:
    """
    SHA-256 hex of a **canonical JSON** encoding of the strategy definition.

    Keys are sorted; separators are fixed so the same logical strategy always
    yields the same fingerprint. Values must be JSON-serializable.
    """
    canonical = json.dumps(strategy_definition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_embedding_fingerprint(
    *,
    embedder_kind: str,
    model: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """
    SHA-256 hex fingerprint for an embedding configuration.

    ``embedder_kind`` names the provider or class family (e.g. ``"openai"``).
    ``model`` is the model id (e.g. ``text-embedding-3-small``).
    Optional ``extra`` is merged into the payload for dimensions / API version, etc.
    """
    payload_obj: dict[str, Any] = {
        "embedder_kind": embedder_kind,
        "model": model,
    }
    if extra:
        payload_obj["extra"] = extra
    canonical = json.dumps(payload_obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

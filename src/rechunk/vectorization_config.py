"""
Shared constants for ECS vectorization (Phase C) — client and worker must agree.
"""

from __future__ import annotations

# Bump when VectorStore row / collection layout changes.
VECTOR_SCHEMA_VERSION = "v1"

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDER_KIND_OPENAI = "openai"

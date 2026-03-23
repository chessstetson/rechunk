"""
On-disk cache for :class:`VectorStoreIndex` embeddings (LlamaIndex storage_context persist).

Avoids re-embedding every ``run_with_docs`` launch when strategies, corpus hashes, chunk caches,
and embedding model are unchanged.

Disable with ``--no-vector-index-cache`` or env ``RECHUNK_NO_VECTOR_INDEX_CACHE=1``.
Override directory with ``RECHUNK_VECTOR_INDEX_CACHE_DIR``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from rechunk.repo_paths import project_root

# Bump when split/embed pipeline changes so stale caches are not reused.
_VECTOR_INDEX_CACHE_FORMAT = 2


def vector_index_cache_root() -> Path:
    env = os.environ.get("RECHUNK_VECTOR_INDEX_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / "storage" / "vector_index_cache"


def disk_cache_disabled() -> bool:
    v = os.environ.get("RECHUNK_NO_VECTOR_INDEX_CACHE", "").strip().lower()
    return v in ("1", "true", "yes")


def embed_model_fingerprint(embed_model: Any) -> str:
    """Stable string for cache key (must change if embedding space changes)."""
    cls = embed_model.__class__.__name__
    model = getattr(embed_model, "model", None) or getattr(embed_model, "_model", None)
    if model is not None:
        return f"{cls}:{model}"
    return cls


def compute_vector_index_cache_key(
    *,
    strategy_ids: list[str],
    content_hashes: list[str],
    strategy_cache_mtimes: dict[str, float],
    embed_model_fp: str,
    cache_source: str = "jsonl",
) -> str:
    """
    ``cache_source``: ``"jsonl"`` = strategy chunk JSONL mtimes (legacy);
    ``"vector_rows"`` = per-strategy max mtime of VectorStore row bundles under ``storage/vector_store_dev``.
    """
    payload = {
        "v": _VECTOR_INDEX_CACHE_FORMAT,
        "source": cache_source,
        "strategies": sorted(strategy_ids),
        "hashes": sorted(content_hashes),
        "mtimes": {k: strategy_cache_mtimes[k] for k in sorted(strategy_cache_mtimes.keys())},
        "embed": embed_model_fp,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def persist_dir_for_cache_key(key: str) -> Path:
    return vector_index_cache_root() / key


def cache_dir_looks_ready(persist_dir: Path) -> bool:
    if not persist_dir.is_dir():
        return False
    # LlamaIndex defaults from StorageContext.persist
    return (persist_dir / "index_store.json").is_file()


def try_load_vector_index_from_disk(persist_dir: Path, embed_model: Any) -> Any | None:
    """
    Load :class:`VectorStoreIndex` from ``persist_dir`` or return ``None``.
    """
    if not cache_dir_looks_ready(persist_dir):
        return None
    try:
        from llama_index.core import StorageContext, load_index_from_storage

        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        return load_index_from_storage(storage_context, embed_model=embed_model)
    except Exception:
        return None


def persist_vector_index(index: Any, persist_dir: Path) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))

"""
Shared strategy chunk cache for ReChunk.

Used by the CLI and Temporal activities so both read/write the same
storage/strategies/{strategy_id}_chunks.jsonl format (per-node ``id``, ``text``, ``metadata`` with
``source_spans``, ``ref_doc_id``). Storage directory can be
overridden via env RECHUNK_STRATEGY_CACHE_DIR for tests.

Freshness: the cache exposes whether it has been updated since a previous
snapshot. Today this is implemented by comparing file mtimes; in principle a
background process (e.g. a Temporal worker or another thread) could set an
"I've been updated" flag and the cache could report that instead, so the CLI
never touches the filesystem directly.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

from llama_index.core.schema import MetadataMode, TextNode

from rechunk.repo_paths import project_root


def _storage_dir() -> Path:
    """Strategy cache root; overridable via RECHUNK_STRATEGY_CACHE_DIR for tests."""
    env = os.environ.get("RECHUNK_STRATEGY_CACHE_DIR")
    if env:
        return Path(env).resolve()
    return project_root() / "storage" / "strategies"


def compute_content_hash(content: str) -> str:
    """SHA-256 hash of document content for deduplication and cache keys."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _strategy_cache_path(strategy_id: str) -> Path:
    return _storage_dir() / f"{strategy_id}_chunks.jsonl"


def _node_to_dict(node: TextNode) -> dict:
    if hasattr(node, "get_content"):
        text = node.get_content(metadata_mode=MetadataMode.NONE)
    else:
        text = getattr(node, "text", "")
    return {
        "id": getattr(node, "id_", None) or "",
        "text": text,
        "metadata": getattr(node, "metadata", None) or {},
        "ref_doc_id": getattr(node, "ref_doc_id", None),
    }


def _dict_to_node(d: dict) -> TextNode:
    return TextNode(
        id_=d.get("id", ""),
        text=d.get("text", ""),
        metadata=d.get("metadata") or {},
        ref_doc_id=d.get("ref_doc_id"),
    )


def append_chunk_cache(strategy_id: str, content_hash: str, nodes: List[TextNode]) -> None:
    """
    Append one doc's chunk results to the per-strategy JSONL cache.
    Same format as the CLI's _append_strategy_chunk_cache.
    """
    root = _storage_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = _strategy_cache_path(strategy_id)
    rec = {
        "content_hash": content_hash,
        "nodes": [_node_to_dict(n) for n in nodes],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def load_chunk_cache(strategy_id: str) -> Dict[str, List[TextNode]]:
    """
    Load cached chunks for a strategy, keyed by document content_hash.
    Same format as the CLI's _load_strategy_chunk_cache.
    """
    path = _strategy_cache_path(strategy_id)
    if not path.exists():
        return {}
    cache: Dict[str, List[TextNode]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    h = rec.get("content_hash")
                    nodes_data = rec.get("nodes") or []
                    if not h or not isinstance(nodes_data, list):
                        continue
                    cache[h] = [_dict_to_node(n) for n in nodes_data]
                except json.JSONDecodeError:
                    continue
    except OSError:
        return {}
    return cache


# --- Freshness: has the cache been updated? (CLI asks cache; cache does the checking) ---


def get_strategy_cache_mtimes(strategy_ids: List[str]) -> Dict[str, float]:
    """
    Return current mtime (or 0.0 if missing) for each strategy's cache file.
    Caller can store this and later ask cache_updated_since() if anything changed.
    """
    out: Dict[str, float] = {}
    for sid in strategy_ids:
        path = _strategy_cache_path(sid)
        out[sid] = path.stat().st_mtime if path.exists() else 0.0
    return out


def cache_updated_since(strategy_ids: List[str], last_known: Dict[str, float]) -> bool:
    """
    True if any of the given strategies' cache files have been modified since
    the last_known mtimes (e.g. from get_strategy_cache_mtimes).
    The cache does the filesystem check; the CLI just asks. In the future,
    a background process could set an "updated" flag and we could check that
    instead so the CLI never touches the filesystem.
    """
    for sid in strategy_ids:
        path = _strategy_cache_path(sid)
        current = path.stat().st_mtime if path.exists() else 0.0
        if current > last_known.get(sid, 0.0):
            return True
    return False

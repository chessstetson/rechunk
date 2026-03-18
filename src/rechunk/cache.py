"""
Shared strategy chunk cache for ReChunk.

Used by the CLI and (later) Temporal activities so both read/write the same
storage/strategies/{strategy_id}_chunks.jsonl format. Storage directory can be
overridden via env RECHUNK_STRATEGY_CACHE_DIR for tests.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

from llama_index.core.schema import MetadataMode, TextNode


def _storage_dir() -> Path:
    """Strategy cache root; overridable via RECHUNK_STRATEGY_CACHE_DIR for tests."""
    env = os.environ.get("RECHUNK_STRATEGY_CACHE_DIR")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2] / "storage" / "strategies"


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
        "start_char_idx": getattr(node, "start_char_idx", None),
        "end_char_idx": getattr(node, "end_char_idx", None),
    }


def _dict_to_node(d: dict) -> TextNode:
    node = TextNode(
        id_=d.get("id", ""),
        text=d.get("text", ""),
        metadata=d.get("metadata") or {},
        ref_doc_id=d.get("ref_doc_id"),
    )
    start = d.get("start_char_idx")
    end = d.get("end_char_idx")
    if start is not None and end is not None:
        node.start_char_idx = start
        node.end_char_idx = end
    return node


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

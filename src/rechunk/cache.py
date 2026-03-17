import hashlib
import json
from pathlib import Path
from typing import Dict, List

from llama_index.core.schema import MetadataMode, TextNode

# NOTE: This module mirrors some of the cache helpers used in the CLI script.
# It exists so Temporal workers can write/read the same cache format without
# depending directly on the CLI module. We may later refactor to a single
# shared implementation.


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_STORAGE_DIR = PROJECT_ROOT / "storage" / "strategies"


def compute_content_hash(content: str) -> str:
    """SHA-256 hash of document content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _strategy_cache_path(strategy_id: str) -> Path:
    return STRATEGY_STORAGE_DIR / f"{strategy_id}_chunks.jsonl"


def _node_to_dict(node: TextNode) -> Dict:
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


def _dict_to_node(d: Dict) -> TextNode:
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


def append_llm_chunk_cache(strategy_id: str, content_hash: str, nodes: List[TextNode]) -> None:
    """Append per-doc chunk results to the LLM strategy cache (JSONL)."""
    STRATEGY_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = _strategy_cache_path(strategy_id)
    rec = {
        "content_hash": content_hash,
        "nodes": [_node_to_dict(n) for n in nodes],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def load_llm_chunk_cache(strategy_id: str) -> Dict[str, List[TextNode]]:
    """
    Load cached chunks for an LLM strategy, keyed by document content_hash.

    This is intentionally simple JSONL storage; we may later migrate to a more
    structured or vector-store-native cache.
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


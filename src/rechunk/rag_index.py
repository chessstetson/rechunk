"""
Build a pooled :class:`VectorStoreIndex` from per-strategy chunk caches.

Chunking is performed elsewhere (Temporal worker); this module only reads
``storage/strategies`` (or ``RECHUNK_STRATEGY_CACHE_DIR``) and merges nodes.
"""

from __future__ import annotations

from collections.abc import Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, MetadataMode, TextNode

from rechunk.cache import load_chunk_cache
from rechunk.corpus import ContentRef
from rechunk.strategies import Strategy


def split_long_nodes_for_embedding(
    nodes: list,
    *,
    # OpenAI embedding endpoint hard-limit is 8192 tokens for some models; use a safety margin.
    max_embed_tokens: int = 7500,
    overlap_tokens: int = 200,
    max_chars_fallback: int = 12000,
) -> list[BaseNode]:
    """
    Ensure no node content exceeds embedding limits by splitting large nodes.

    Token-aware when ``tiktoken`` is available; otherwise char-based fallback.
    """
    enc = None
    try:
        import tiktoken  # type: ignore

        try:
            enc = tiktoken.encoding_for_model("text-embedding-3-small")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

    result: list[BaseNode] = []
    for node in nodes:
        if hasattr(node, "get_content"):
            text = node.get_content(metadata_mode=MetadataMode.NONE)
        else:
            text = getattr(node, "text", "")
        if not isinstance(text, str):
            text = str(text)

        token_count = None
        if enc is not None:
            try:
                token_count = len(enc.encode(text))
            except Exception:
                token_count = None
        if (token_count is not None and token_count <= max_embed_tokens) or (
            token_count is None and len(text) <= max_chars_fallback
        ):
            result.append(node)
            continue

        meta = getattr(node, "metadata", None) or {}
        ref_doc_id = getattr(node, "ref_doc_id", None) or meta.get("source_doc", "")
        base_id = getattr(node, "id_", None) or ref_doc_id or "node"
        part_idx = 1
        if enc is not None:
            try:
                toks = enc.encode(text)
                step = max(1, max_embed_tokens - max(0, overlap_tokens))
                start = 0
                while start < len(toks):
                    window = toks[start : start + max_embed_tokens]
                    chunk_text = enc.decode(window)
                    part_id = f"{base_id}_part{part_idx}"
                    result.append(
                        TextNode(id_=part_id, text=chunk_text, metadata=dict(meta), ref_doc_id=ref_doc_id)
                    )
                    part_idx += 1
                    start += step
                continue
            except Exception:
                pass

        start = 0
        while start < len(text):
            chunk_text = text[start : start + max_chars_fallback]
            part_id = f"{base_id}_part{part_idx}"
            result.append(TextNode(id_=part_id, text=chunk_text, metadata=dict(meta), ref_doc_id=ref_doc_id))
            start += max_chars_fallback
            part_idx += 1
    return result


def collect_pooled_nodes_from_strategy_caches(
    strategies: list[Strategy],
    content_refs: Sequence[ContentRef],
    *,
    quiet: bool = False,
) -> list[BaseNode]:
    """
    Load cached chunks for each strategy and concatenate into one node list
    (pooling). Does not split for embedding limits or build an index.
    """
    all_nodes: list[BaseNode] = []
    n_docs = len(content_refs)
    for idx, s in enumerate(strategies, 1):
        if not quiet:
            print(f"\n  Strategy {idx}/{len(strategies)}: {s.id} ({s.kind}) — {n_docs} content object(s)")
        cache = load_chunk_cache(s.id)
        strat_nodes: list[BaseNode] = []
        for j, ref in enumerate(content_refs, 1):
            content_hash = ref.content_hash
            label = ref.source_hint or content_hash[:12] + "..."
            if content_hash in cache:
                strat_nodes.extend(cache[content_hash])
                if not quiet:
                    print(f"    [{j}/{n_docs}] cache hit for hash={content_hash[:12]}... ({label})")
            else:
                if not quiet:
                    print(
                        f"    [{j}/{n_docs}] no cached chunks for {label!r} (strategy {s.id}); "
                        "run worker + start_strategy_chunking to backfill."
                    )
        all_nodes.extend(strat_nodes)
        if not quiet:
            print(f"    → {len(strat_nodes)} chunks from cache")
    if not quiet:
        print(f"\n  Total before max-length enforcement: {len(all_nodes)} chunks from {len(strategies)} strategy(ies)")
    return all_nodes


def build_vector_index_from_strategies(
    strategies: list[Strategy],
    content_refs: Sequence[ContentRef],
    *,
    quiet: bool = False,
) -> tuple[VectorStoreIndex, list[BaseNode]]:
    """
    Build index from cache only. All chunking is done by the Temporal worker;
    callers never run chunking here.
    """
    all_nodes = collect_pooled_nodes_from_strategy_caches(strategies, content_refs, quiet=quiet)
    all_nodes = split_long_nodes_for_embedding(all_nodes)
    if not quiet:
        print(f"  Total after max-length enforcement: {len(all_nodes)} chunks")

    try:
        index = VectorStoreIndex(all_nodes)
    except Exception as e:
        raise RuntimeError(
            "Failed to build VectorStoreIndex (embedding step). "
            "This usually means one chunk exceeded the embed token limit or the embedding API failed. "
            "Try rerunning, or reduce chunk sizes."
        ) from e
    return index, all_nodes

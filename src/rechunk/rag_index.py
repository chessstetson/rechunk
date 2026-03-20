"""
Build a pooled :class:`VectorStoreIndex` from per-strategy chunk caches.

Chunking is performed elsewhere (Temporal worker); this module only reads
``storage/strategies`` (or ``RECHUNK_STRATEGY_CACHE_DIR``) and merges nodes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
    embed_model: Any | None = None,
) -> tuple[VectorStoreIndex, list[BaseNode]]:
    """
    Build index from cache only. All chunking is done by the Temporal worker;
    callers never run chunking here.

    If ``embed_model`` is ``None``, LlamaIndex uses :attr:`Settings.embed_model`.
    """
    all_nodes = collect_pooled_nodes_from_strategy_caches(strategies, content_refs, quiet=quiet)
    all_nodes = split_long_nodes_for_embedding(all_nodes)
    if not quiet:
        print(f"  Total after max-length enforcement: {len(all_nodes)} chunks")

    try:
        index = (
            VectorStoreIndex(all_nodes, embed_model=embed_model)
            if embed_model is not None
            else VectorStoreIndex(all_nodes)
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to build VectorStoreIndex (embedding step). "
            "This usually means one chunk exceeded the embed token limit or the embedding API failed. "
            "Try rerunning, or reduce chunk sizes."
        ) from e
    return index, all_nodes


def load_or_build_vector_index_from_strategies(
    strategies: list[Strategy],
    content_refs: Sequence[ContentRef],
    *,
    embed_model: Any,
    quiet: bool = False,
    use_disk_cache: bool = True,
) -> tuple[VectorStoreIndex, list[BaseNode]]:
    """
    Like :func:`build_vector_index_from_strategies`, but reuses a persisted
    :class:`VectorStoreIndex` when strategy ids, content hashes, chunk-cache mtimes,
    and embedding model fingerprint match (see ``rechunk.vector_index_cache``).
    """
    from rechunk.cache import get_strategy_cache_mtimes
    from rechunk.vector_index_cache import (
        compute_vector_index_cache_key,
        disk_cache_disabled,
        embed_model_fingerprint,
        persist_dir_for_cache_key,
        persist_vector_index,
        try_load_vector_index_from_disk,
    )

    strategy_ids = [s.id for s in strategies]
    mtimes = get_strategy_cache_mtimes(strategy_ids)
    hashes = [ref.content_hash for ref in content_refs]
    use_disk = use_disk_cache and not disk_cache_disabled()
    emb_fp = embed_model_fingerprint(embed_model)

    if use_disk:
        key = compute_vector_index_cache_key(
            strategy_ids=strategy_ids,
            content_hashes=hashes,
            strategy_cache_mtimes=mtimes,
            embed_model_fp=emb_fp,
        )
        persist_dir = persist_dir_for_cache_key(key)
        loaded = try_load_vector_index_from_disk(persist_dir, embed_model)
        if loaded is not None:
            nodes = list(loaded.docstore.docs.values())
            if not quiet:
                print(
                    f"  Vector index loaded from disk cache ({key[:12]}…, {len(nodes)} chunks). "
                    f"Use --no-vector-index-cache to force re-embed."
                )
            return loaded, nodes

    index, all_nodes = build_vector_index_from_strategies(
        strategies, content_refs, quiet=quiet, embed_model=embed_model
    )
    if use_disk:
        key = compute_vector_index_cache_key(
            strategy_ids=strategy_ids,
            content_hashes=hashes,
            strategy_cache_mtimes=mtimes,
            embed_model_fp=emb_fp,
        )
        persist_dir = persist_dir_for_cache_key(key)
        try:
            persist_vector_index(index, persist_dir)
            if not quiet:
                print(f"  Saved vector index to disk cache ({key[:12]}…).")
        except Exception as e:
            if not quiet:
                print(f"  [WARN] Could not persist vector index cache: {e}", flush=True)

    return index, all_nodes

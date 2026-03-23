"""
Build a pooled :class:`VectorStoreIndex` from chunk caches or VectorStore rows.

**Default (ECS path):** reads pre-embedded rows from :class:`~rechunk.vector_store.filesystem.FilesystemVectorStore`
(keyed by strategy fingerprint + embedding fingerprint + schema). Chunking/embeddings are produced by the
Temporal worker.

**Legacy:** reads ``storage/strategies/*_chunks.jsonl`` via :func:`collect_pooled_nodes_from_strategy_caches`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, MetadataMode, TextNode

from rechunk.cache import load_chunk_cache
from rechunk.corpus import ContentRef
from rechunk.fingerprints import compute_strategy_fingerprint
from rechunk.strategies import Strategy, strategy_to_dict
from rechunk.vector_store.freshness import get_vector_store_strategy_mtimes
from rechunk.vector_store.protocol import VectorStore
from rechunk.vectorization_config import VECTOR_SCHEMA_VERSION


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

        meta = dict(getattr(node, "metadata", None) or {})
        # Split parts are arbitrary slices of ``text``; drop LLM span hints (single or multi).
        meta.pop("span_ranges", None)
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
                    child = TextNode(id_=part_id, text=chunk_text, metadata=dict(meta), ref_doc_id=ref_doc_id)
                    result.append(child)
                    part_idx += 1
                    start += step
                continue
            except Exception:
                pass

        start = 0
        while start < len(text):
            chunk_text = text[start : start + max_chars_fallback]
            part_id = f"{base_id}_part{part_idx}"
            child = TextNode(id_=part_id, text=chunk_text, metadata=dict(meta), ref_doc_id=ref_doc_id)
            result.append(child)
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


def _text_node_from_vector_row(
    row: dict,
    *,
    content_hash: str,
    strategy_id: str,
    row_index: int,
) -> TextNode:
    """Build a :class:`TextNode` with ``embedding`` set from a VectorStore row dict."""
    text = row.get("chunk_text")
    if not isinstance(text, str):
        text = str(text or "")
    emb = row.get("embedding")
    meta = dict(row.get("metadata") or {})
    span_s = row.get("span_start")
    span_e = row.get("span_end")
    if span_s is not None:
        meta.setdefault("span_start", span_s)
    if span_e is not None:
        meta.setdefault("span_end", span_e)
    sr = row.get("span_ranges")
    if isinstance(sr, list) and sr:
        meta.setdefault("span_ranges", sr)
    ref_doc = meta.get("source_doc") or content_hash[:16]
    node_id = f"{content_hash[:16]}_{strategy_id}_{span_s}_{span_e}_{row_index}"
    embedding_list = list(emb) if isinstance(emb, list) else None
    return TextNode(
        id_=node_id,
        text=text,
        metadata=meta,
        ref_doc_id=ref_doc,
        embedding=embedding_list,
    )


def collect_pooled_nodes_from_vector_store(
    vector_store: VectorStore,
    strategies: list[Strategy],
    content_refs: Sequence[ContentRef],
    *,
    embedding_fingerprint: str,
    vector_schema_version: str = VECTOR_SCHEMA_VERSION,
    quiet: bool = False,
) -> list[BaseNode]:
    """
    Load row bundles per (strategy fingerprint, content hash) and merge into one node list.

    Expects the same ``embedding_fingerprint`` / schema the worker used when calling ``upsert_rows``.
    """
    all_nodes: list[BaseNode] = []
    n_docs = len(content_refs)
    for idx, s in enumerate(strategies, 1):
        sfp = compute_strategy_fingerprint(strategy_to_dict(s))
        if not quiet:
            print(
                f"\n  Strategy {idx}/{len(strategies)}: {s.id} ({s.kind}) — "
                f"{n_docs} content object(s) [VectorStore rows]"
            )
        strat_nodes: list[BaseNode] = []
        for j, ref in enumerate(content_refs, 1):
            content_hash = ref.content_hash
            label = ref.source_hint or content_hash[:12] + "..."
            rows = vector_store.read_rows_for_hash(
                content_hash=content_hash,
                strategy_fingerprint=sfp,
                embedding_fingerprint=embedding_fingerprint,
                vector_schema_version=vector_schema_version,
            )
            if rows:
                for ri, row in enumerate(rows):
                    strat_nodes.append(
                        _text_node_from_vector_row(
                            row,
                            content_hash=content_hash,
                            strategy_id=s.id,
                            row_index=ri,
                        )
                    )
                if not quiet:
                    print(f"    [{j}/{n_docs}] {len(rows)} row(s) for hash={content_hash[:12]}… ({label})")
            elif not quiet:
                print(
                    f"    [{j}/{n_docs}] no vector rows for {label!r} (strategy {s.id}); "
                    "run worker + start_strategy_chunking.py."
                )
        all_nodes.extend(strat_nodes)
        if not quiet:
            print(f"    → {len(strat_nodes)} chunks from VectorStore")
    if not quiet:
        print(f"\n  Total from VectorStore: {len(all_nodes)} chunks from {len(strategies)} strategy(ies)")
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


def build_vector_index_from_vector_store(
    vector_store: VectorStore,
    strategies: list[Strategy],
    content_refs: Sequence[ContentRef],
    *,
    embedding_fingerprint: str,
    vector_schema_version: str = VECTOR_SCHEMA_VERSION,
    quiet: bool = False,
    embed_model: Any | None = None,
) -> tuple[VectorStoreIndex, list[BaseNode]]:
    """
    Assemble a pooled index from persisted vector rows (embeddings already computed by the worker).

    Does not re-chunk; optional split pass is skipped because rows are already sized for embedding.
    """
    all_nodes = collect_pooled_nodes_from_vector_store(
        vector_store,
        strategies,
        content_refs,
        embedding_fingerprint=embedding_fingerprint,
        vector_schema_version=vector_schema_version,
        quiet=quiet,
    )
    if not quiet:
        print(f"  Building VectorStoreIndex from {len(all_nodes)} pre-embedded nodes")

    try:
        index = (
            VectorStoreIndex(all_nodes, embed_model=embed_model)
            if embed_model is not None
            else VectorStoreIndex(all_nodes)
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to build VectorStoreIndex from vector rows. "
            "Check that rows include chunk_text + embedding and match Settings.embed_model."
        ) from e
    return index, all_nodes


def load_or_build_vector_index_from_vector_store(
    vector_store: VectorStore,
    strategies: list[Strategy],
    content_refs: Sequence[ContentRef],
    *,
    embed_model: Any,
    embedding_fingerprint: str,
    vector_schema_version: str = VECTOR_SCHEMA_VERSION,
    quiet: bool = False,
    use_disk_cache: bool = True,
) -> tuple[VectorStoreIndex, list[BaseNode]]:
    """
    Like :func:`load_or_build_vector_index_from_strategies`, but sources nodes from VectorStore
    rows and keys the disk cache by per-strategy row-bundle mtimes (``cache_source=vector_rows``).
    """
    from rechunk.vector_index_cache import (
        compute_vector_index_cache_key,
        disk_cache_disabled,
        embed_model_fingerprint,
        persist_dir_for_cache_key,
        persist_vector_index,
        try_load_vector_index_from_disk,
    )

    strategy_ids = [s.id for s in strategies]
    hashes = [ref.content_hash for ref in content_refs]
    use_disk = use_disk_cache and not disk_cache_disabled()
    emb_fp = embed_model_fingerprint(embed_model)
    mtimes = get_vector_store_strategy_mtimes(
        vector_store,
        strategies,
        content_refs,
        embedding_fingerprint=embedding_fingerprint,
        vector_schema_version=vector_schema_version,
    )

    if use_disk:
        key = compute_vector_index_cache_key(
            strategy_ids=strategy_ids,
            content_hashes=hashes,
            strategy_cache_mtimes=mtimes,
            embed_model_fp=emb_fp,
            cache_source="vector_rows",
        )
        persist_dir = persist_dir_for_cache_key(key)
        loaded = try_load_vector_index_from_disk(persist_dir, embed_model)
        if loaded is not None:
            nodes = list(loaded.docstore.docs.values())
            if len(nodes) > 0:
                if not quiet:
                    print(
                        f"  Vector index loaded from disk cache ({key[:12]}…, {len(nodes)} chunks). "
                        f"Use --no-vector-index-cache to force rebuild."
                    )
                return loaded, nodes
            if not quiet:
                print(
                    f"  Ignoring empty disk vector cache ({key[:12]}…); rebuilding from VectorStore rows.",
                    flush=True,
                )

    index, all_nodes = build_vector_index_from_vector_store(
        vector_store,
        strategies,
        content_refs,
        embedding_fingerprint=embedding_fingerprint,
        vector_schema_version=vector_schema_version,
        quiet=quiet,
        embed_model=embed_model,
    )
    if use_disk:
        key = compute_vector_index_cache_key(
            strategy_ids=strategy_ids,
            content_hashes=hashes,
            strategy_cache_mtimes=mtimes,
            embed_model_fp=emb_fp,
            cache_source="vector_rows",
        )
        persist_dir = persist_dir_for_cache_key(key)
        if len(all_nodes) > 0:
            try:
                persist_vector_index(index, persist_dir)
                if not quiet:
                    print(f"  Saved vector index to disk cache ({key[:12]}…).")
            except Exception as e:
                if not quiet:
                    print(f"  [WARN] Could not persist vector index cache: {e}", flush=True)
        elif not quiet:
            print(
                "  Skipped saving empty vector index to disk cache (vectorize corpus first).",
                flush=True,
            )

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
            # Do not reuse an empty persisted index (e.g. saved before chunk backfill).
            if len(nodes) > 0:
                if not quiet:
                    print(
                        f"  Vector index loaded from disk cache ({key[:12]}…, {len(nodes)} chunks). "
                        f"Use --no-vector-index-cache to force re-embed."
                    )
                return loaded, nodes
            if not quiet:
                print(
                    f"  Ignoring empty disk vector cache ({key[:12]}…); rebuilding from chunk cache.",
                    flush=True,
                )

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
        if len(all_nodes) > 0:
            try:
                persist_vector_index(index, persist_dir)
                if not quiet:
                    print(f"  Saved vector index to disk cache ({key[:12]}…).")
            except Exception as e:
                if not quiet:
                    print(f"  [WARN] Could not persist vector index cache: {e}", flush=True)
        elif not quiet:
            print(
                "  Skipped saving empty vector index to disk cache (run worker + chunking first).",
                flush=True,
            )

    return index, all_nodes

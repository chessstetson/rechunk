"""Disk cache for VectorStoreIndex (embedding reuse across CLI launches)."""

from __future__ import annotations

from llama_index.core.embeddings import MockEmbedding

from rechunk.cache import append_chunk_cache, compute_content_hash
from rechunk.corpus import ContentRef
from rechunk.rag_index import load_or_build_vector_index_from_strategies
from rechunk.strategies import Strategy
from rechunk.vector_index_cache import (
    compute_vector_index_cache_key,
    embed_model_fingerprint,
)


def test_cache_key_stable_for_hash_order() -> None:
    fp = embed_model_fingerprint(MockEmbedding(embed_dim=8))
    k1 = compute_vector_index_cache_key(
        strategy_ids=["s1"],
        content_hashes=["bbb", "aaa"],
        strategy_cache_mtimes={"s1": 1.0},
        embed_model_fp=fp,
    )
    k2 = compute_vector_index_cache_key(
        strategy_ids=["s1"],
        content_hashes=["aaa", "bbb"],
        strategy_cache_mtimes={"s1": 1.0},
        embed_model_fp=fp,
    )
    assert k1 == k2


def test_load_or_build_second_call_uses_disk_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RECHUNK_STRATEGY_CACHE_DIR", str(tmp_path / "strat_cache"))
    cache_root = tmp_path / "vec_cache"
    monkeypatch.setenv("RECHUNK_VECTOR_INDEX_CACHE_DIR", str(cache_root))

    text = "hello cache disk"
    h = compute_content_hash(text)
    ref = ContentRef(content_hash=h, source_hint="x.txt")
    n = __import__("llama_index.core.schema", fromlist=["TextNode"]).TextNode(
        id_="n1", text="chunk one", metadata={"strategy": "s_disk", "source_doc": "x.txt"}
    )
    append_chunk_cache("s_disk", h, [n])
    strategies = [Strategy(id="s_disk", kind="builtin_splitter", instruction="x")]
    emb = MockEmbedding(embed_dim=8)

    idx1, nodes1 = load_or_build_vector_index_from_strategies(
        strategies, [ref], embed_model=emb, quiet=True, use_disk_cache=True
    )
    assert len(nodes1) >= 1

    idx2, nodes2 = load_or_build_vector_index_from_strategies(
        strategies, [ref], embed_model=emb, quiet=True, use_disk_cache=True
    )
    assert len(nodes2) == len(nodes1)
    assert idx1 is not idx2  # new object, but loaded from disk
    assert len(list(cache_root.iterdir())) >= 1

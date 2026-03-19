"""Tests for pooled cache → nodes / split helpers (no full interactive CLI)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode

from rechunk.cache import append_chunk_cache, compute_content_hash
from rechunk.rag_index import (
    collect_pooled_nodes_from_strategy_caches,
    split_long_nodes_for_embedding,
)
from rechunk.strategies import Strategy


@pytest.fixture
def isolated_cache_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        monkeypatch.setenv("RECHUNK_STRATEGY_CACHE_DIR", str(root))
        yield root


def test_collect_pooled_nodes_merges_two_strategies(isolated_cache_dir: Path) -> None:
    text = "hello pooled world"
    h = compute_content_hash(text)
    doc = Document(text=text, id_="a.txt", metadata={"content_hash": h})

    n1 = TextNode(id_="n1", text="chunk A", metadata={"strategy": "s_a"})
    n2 = TextNode(id_="n2", text="chunk B", metadata={"strategy": "s_b"})

    append_chunk_cache("s_a", h, [n1])
    append_chunk_cache("s_b", h, [n2])

    strategies = [
        Strategy(id="s_a", kind="builtin_splitter", instruction="a"),
        Strategy(id="s_b", kind="builtin_splitter", instruction="b"),
    ]
    pooled = collect_pooled_nodes_from_strategy_caches(strategies, [doc], quiet=True)
    assert len(pooled) == 2
    texts = {getattr(n, "text", "") for n in pooled}
    assert texts == {"chunk A", "chunk B"}


def test_split_long_nodes_char_fallback_splits_oversized() -> None:
    # Force char path: tiny limit so any non-trivial text splits.
    long_text = "x" * 5000
    node = TextNode(id_="big", text=long_text, metadata={"strategy": "s"})
    out = split_long_nodes_for_embedding(
        [node],
        max_embed_tokens=1,
        max_chars_fallback=1000,
    )
    assert len(out) >= 5
    assert sum(len(getattr(n, "text", "")) for n in out) >= len(long_text)


def test_collect_pooled_empty_when_cache_missing() -> None:
    doc = Document(text="z", id_="z.txt", metadata={"content_hash": compute_content_hash("z")})
    strategies = [Strategy(id="missing_strategy", kind="builtin_splitter", instruction="x")]
    pooled = collect_pooled_nodes_from_strategy_caches(strategies, [doc], quiet=True)
    assert pooled == []

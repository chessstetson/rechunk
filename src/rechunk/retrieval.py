"""
Thin retrieval + synthesis helpers over a LlamaIndex :class:`VectorStoreIndex`.

Use these in tests or scripts without going through the interactive CLI.
"""

from __future__ import annotations

from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import QueryBundle


def retrieve_top_k(index: VectorStoreIndex, query: str, top_k: int = 5) -> Any:
    """
    Return ``NodeWithScore`` list from cosine-similarity retrieval (embeddings only).

    LlamaIndex type is typically ``list[NodeWithScore]``; kept as ``Any`` to avoid
    tight coupling to internal schema names across versions.
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(QueryBundle(query_str=query))


def synthesize_with_retrieved_nodes(
    index: VectorStoreIndex,
    query: str,
    nodes_with_scores: Any,
) -> Any:
    """Run the query engine's synthesize step on pre-retrieved nodes."""
    engine = index.as_query_engine()
    query_bundle = QueryBundle(query_str=query)
    return engine.synthesize(query_bundle, nodes_with_scores)

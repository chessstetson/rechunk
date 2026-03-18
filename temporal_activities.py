"""
Temporal activities for ReChunk: chunk documents with a strategy and write to cache.

Activities run in workers; they do I/O and LLM calls. Workflows only orchestrate.
Two paths: LLM-based (chunk_doc_with_strategy) and LlamaIndex built-in (chunk_doc_with_builtin_splitter).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from temporalio import activity

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

from rechunk import LLMNodeParser
from rechunk.cache import append_chunk_cache, compute_content_hash


@dataclass
class ChunkDocInput:
    """Input for chunk_doc_with_strategy activity.

    Doc text is read from disk (claim-check); not passed in the payload.
    """

    strategy_id: str
    strategy_instruction: str
    model: Optional[str]
    docs_root: str
    doc_id: str
    content_hash: str


@activity.defn
async def chunk_doc_with_strategy(input: ChunkDocInput) -> None:
    """Chunk one document with an LLM strategy and append results to the shared cache."""
    doc_path = Path(input.docs_root) / input.doc_id
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return

    content_hash = input.content_hash or compute_content_hash(text)
    doc = Document(text=text, id_=input.doc_id)
    parser = LLMNodeParser(
        strategy_id=input.strategy_id,
        strategy_instruction=input.strategy_instruction,
    )
    # Uses Settings.llm when llm is None (set by CLI or worker)
    nodes = parser.get_nodes_from_documents([doc])
    append_chunk_cache(input.strategy_id, content_hash, nodes)


@dataclass
class BuiltinChunkInput:
    """Input for chunk_doc_with_builtin_splitter activity. Doc text read from disk."""

    strategy_id: str
    splitter: str  # "sentence" or "token"
    docs_root: str
    doc_id: str
    content_hash: str


@activity.defn
async def chunk_doc_with_builtin_splitter(input: BuiltinChunkInput) -> None:
    """Chunk one document with a LlamaIndex built-in splitter and append to cache."""
    doc_path = Path(input.docs_root) / input.doc_id
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return

    chunk_size, chunk_overlap = 1024, 20
    if input.splitter == "token":
        parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    doc = Document(text=text, id_=input.doc_id)
    nodes = parser.get_nodes_from_documents([doc])
    for n in nodes:
        n.metadata = getattr(n, "metadata", None) or {}
        n.metadata["strategy"] = input.strategy_id
        n.metadata.setdefault("source_doc", getattr(n, "ref_doc_id", "") or input.doc_id)
    append_chunk_cache(input.strategy_id, input.content_hash, nodes)

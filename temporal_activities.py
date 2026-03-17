from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from temporalio import activity

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

from rechunk import LLMNodeParser
from rechunk.cache import append_llm_chunk_cache, compute_content_hash


@dataclass
class ChunkDocInput:
    """Input for chunk_doc_with_strategy Activity.

    NOTE: This is a first, pragmatic version. We may later adopt a more
    structured manifest or an external doc store.
    """

    strategy_id: str
    strategy_instruction: str
    model: Optional[str]
    docs_root: str          # root directory for documents
    doc_id: str             # path relative to docs_root


@dataclass
class BuiltinChunkInput:
    """Input for chunk_doc_with_builtin_splitter Activity."""

    strategy_id: str
    splitter: str  # "sentence" or "token"
    docs_root: str
    doc_id: str


@activity.defn
async def chunk_doc_with_strategy(input: ChunkDocInput) -> None:
    """LLM chunking Activity for a single document + strategy.

    Reads the document from disk, runs LLMNodeParser, and appends chunks to the
    per-strategy cache. Temporal handles retries for this Activity, so it is
    safe to re-run on failure.
    """
    doc_path = Path(input.docs_root) / input.doc_id
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return

    content_hash = compute_content_hash(text)

    parser = LLMNodeParser(
        strategy_id=input.strategy_id,
        strategy_instruction=input.strategy_instruction,
    )
    # For v1 we ignore input.model and rely on Settings.llm; we may later
    # switch to injecting a specific LLM instance here.
    nodes = parser.get_nodes_from_documents([Document(text=text, id_=input.doc_id)])

    append_llm_chunk_cache(input.strategy_id, content_hash, nodes)


@activity.defn
async def chunk_doc_with_builtin_splitter(input: BuiltinChunkInput) -> None:
    """Built-in splitter Activity for a single document + strategy.

    Uses SentenceSplitter or TokenTextSplitter to compute chunks and appends
    them to the per-strategy cache. This mirrors the behaviour of the old
    synchronous path, but runs in a Temporal Activity instead.
    """
    doc_path = Path(input.docs_root) / input.doc_id
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return

    if input.splitter == "token":
        parser = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
    else:
        parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    nodes = parser.get_nodes_from_documents([Document(text=text, id_=input.doc_id)])
    for n in nodes:
        n.metadata = (n.metadata or {}) | {
            "strategy": input.strategy_id,
            "source_doc": input.doc_id,
        }

    content_hash = compute_content_hash(text)
    append_llm_chunk_cache(input.strategy_id, content_hash, nodes)



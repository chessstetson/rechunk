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
from rechunk.cache import (
    append_chunk_cache,
    compute_content_hash,
    get_cached_hashes_for_strategy,
)
from rechunk.doc_loader import extract_file_content


@dataclass
class LoadDocManifestInput:
    """Input for load_doc_manifest activity. Returns list of {doc_id, content_hash}."""

    docs_root: str
    doc_ids: list


@activity.defn
async def load_doc_manifest(input: LoadDocManifestInput) -> list:
    """
    Read each document from disk and compute content hash. Returns list of
    {"doc_id": str, "content_hash": str}. Skips docs that cannot be read.
    """
    import sys

    docs_root = Path(input.docs_root)
    manifest = []
    for doc_id in input.doc_ids:
        path = docs_root / doc_id
        if not path.exists():
            continue
        text = extract_file_content(path)
        if not text or not text.strip():
            continue
        content_hash = compute_content_hash(text)
        manifest.append({"doc_id": doc_id, "content_hash": content_hash})
    print(f"      [manifest] {len(manifest)} docs (of {len(input.doc_ids)} requested)", file=sys.stderr, flush=True)
    return manifest


@dataclass
class GetCachedHashesInput:
    """Input for get_cached_hashes activity. Returns list of content_hash in cache."""

    strategy_id: str


@activity.defn
async def get_cached_hashes(input: GetCachedHashesInput) -> list:
    """Return all content_hash values already in the strategy's chunk cache."""
    import sys

    hashes = get_cached_hashes_for_strategy(input.strategy_id)
    print(f"      [cache] {input.strategy_id!r}: {len(hashes)} doc(s) already chunked", file=sys.stderr, flush=True)
    return hashes


@dataclass
class LogWorkflowSummaryInput:
    """Input for log_workflow_summary. Prints one line to stderr so the worker shows feedback."""

    strategy_id: str
    total: int
    skipped: int
    processed: int


@activity.defn
async def log_workflow_summary(input: LogWorkflowSummaryInput) -> None:
    """Print workflow completion summary to stderr so the worker terminal shows feedback."""
    import sys

    msg = (
        f"ReChunk workflow rechunk-{input.strategy_id} completed: "
        f"{input.total} total, {input.skipped} already cached, {input.processed} chunked."
    )
    print(msg, file=sys.stderr, flush=True)


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
    text = extract_file_content(doc_path)
    if not text or not text.strip():
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
    text = extract_file_content(doc_path)
    if not text or not text.strip():
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

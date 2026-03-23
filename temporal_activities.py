"""
Temporal activities for ReChunk: chunk documents with a strategy and write to cache.

Activities run in workers; they do I/O and LLM calls. Workflows only orchestrate.
Two paths: LLM-based (chunk_doc_with_strategy) and LlamaIndex built-in (chunk_doc_with_builtin_splitter).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from temporalio import activity

from temporal_vectorization_inputs import DocumentVectorizationInput

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import MetadataMode

from rechunk import LLMNodeParser
from rechunk.strategies import strategy_definition_uses_llm
from rechunk.cache import (
    append_chunk_cache,
    compute_content_hash,
    get_cached_hashes_for_strategy,
)
from rechunk.doc_loader import extract_file_content
from rechunk.rag_index import split_long_nodes_for_embedding


@dataclass
class LoadDocManifestInput:
    """Input for load_doc_manifest activity. Returns list of {doc_id, content_hash}."""

    docs_root: str
    doc_ids: list


@dataclass
class LoadIngestSnapshotInput:
    """Path to ingest snapshot JSON (written before workflow start; worker reads same filesystem)."""

    snapshot_path: str


@activity.defn(name="ingest_filesystem_corpus_from_snapshot")
async def ingest_filesystem_corpus_from_snapshot(input: LoadIngestSnapshotInput) -> dict:
    """
    Read an ingest snapshot (path on shared disk), extract + hash-verify, write into ECS,
    prune logical docs not in this inventory, replace ``corpus_content_hashes.json`` from ECS.

    Owns **ingest only** — does not start vectorization (separate workflow / queue).
    """
    import sys
    from pathlib import Path

    from rechunk.extracted_content import FilesystemExtractedContentService
    from rechunk.index_service import IndexService
    from rechunk.ingest_snapshot import read_ingest_snapshot
    from rechunk.vector_store import FilesystemVectorStore

    try:
        docs_root, manifest = read_ingest_snapshot(Path(input.snapshot_path))
    except (FileNotFoundError, ValueError, OSError) as e:
        from temporalio.exceptions import ApplicationError

        raise ApplicationError(str(e), non_retryable=True) from e

    doc_ids = [m["doc_id"] for m in manifest]
    ecs = FilesystemExtractedContentService()
    vs = FilesystemVectorStore()
    idx = IndexService(ecs=ecs, vector_store=vs, strategies_path=None)
    idx.ingest_filesystem_docs(docs_root, doc_ids)
    ecs.apply_source_inventory("filesystem", doc_ids)
    hashes_written = idx.sync_active_manifest_file()

    print(
        f"      [ingest] snapshot={input.snapshot_path!r} logical={len(doc_ids)} "
        f"active_hashes={len(hashes_written)}",
        file=sys.stderr,
        flush=True,
    )
    return {
        "docs_root": str(docs_root.resolve()),
        "ingested_logical_docs": len(doc_ids),
        "active_unique_hashes": len(hashes_written),
        "snapshot_path": input.snapshot_path,
    }


@activity.defn
async def load_manifest_from_ingest_snapshot(input: LoadIngestSnapshotInput) -> dict:
    """
    Read ingest snapshot, verify each file's hash matches disk, return ``docs_root`` + ``manifest``.

    Workflow history stays small (path only); doc list and hashes live in the snapshot file.
    """
    import sys

    from temporalio.exceptions import ApplicationError

    from rechunk.ingest_snapshot import read_ingest_snapshot

    try:
        docs_root, manifest = read_ingest_snapshot(Path(input.snapshot_path))
    except (FileNotFoundError, ValueError, OSError) as e:
        raise ApplicationError(str(e), non_retryable=True) from e
    print(
        f"      [ingest_snapshot] {len(manifest)} docs from {input.snapshot_path}",
        file=sys.stderr,
        flush=True,
    )
    return {"docs_root": str(docs_root), "manifest": manifest}


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
class MergeActiveCorpusManifestInput:
    """Content hashes from this workflow's doc manifest (SHA-256 hex strings)."""

    content_hashes: list


@activity.defn
async def merge_active_corpus_manifest(input: MergeActiveCorpusManifestInput) -> dict:
    """
    Merge workflow manifest hashes into the shared active corpus file (hash-only JSON).

    Ingest owns this artifact; Q&A can ``--manifest`` the same path. See
    ``rechunk.active_corpus_manifest``.
    """
    from rechunk.active_corpus_manifest import (
        active_corpus_manifest_path,
        merge_content_hashes_into_active_manifest,
    )

    path = active_corpus_manifest_path()
    if not input.content_hashes:
        return {"total_unique_hashes": 0, "path": str(path)}
    import sys

    merged = merge_content_hashes_into_active_manifest(input.content_hashes)
    print(
        f"      [active_corpus] merged {len(input.content_hashes)} hash(es) from workflow → "
        f"{len(merged)} unique in {path}",
        file=sys.stderr,
        flush=True,
    )
    return {"total_unique_hashes": len(merged), "path": str(path)}


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


@activity.defn(name="vectorize_content_for_strategy")
async def vectorize_content_for_strategy(input: DocumentVectorizationInput) -> dict[str, Any]:
    """
    Load extracted text from ECS, chunk (LLM or builtin), dual-write JSONL cache + VectorStore rows.

    Skips if vector rows already exist for (hash, strategy_fp, embed_fp, schema).
    """
    import sys

    from rechunk.node_span_utils import char_spans_for_nodes
    from rechunk.worker_runtime import get_worker_ecs, get_worker_vector_store

    ecs = get_worker_ecs()
    vs = get_worker_vector_store()

    done = set(
        vs.list_vectorized_hashes(
            strategy_fingerprint=input.strategy_fingerprint,
            embedding_fingerprint=input.embedding_fingerprint,
            vector_schema_version=input.vector_schema_version,
        )
    )
    if input.content_hash.lower() in {x.lower() for x in done}:
        print(
            f"      [vectorize] skip {input.content_hash[:12]}… (already vectorized for strategy fp)",
            file=sys.stderr,
            flush=True,
        )
        return {"status": "skipped", "rows": 0}

    try:
        ec = ecs.get_content(input.content_hash)
    except KeyError as e:
        from temporalio.exceptions import ApplicationError

        raise ApplicationError(str(e), non_retryable=True) from e

    text = (ec.extracted_text or "").strip()
    if not text:
        return {"status": "skipped", "rows": 0}

    doc_id = ec.logical_doc_id or input.content_hash[:16]
    sd = input.strategy_definition
    strategy_id = sd.get("id", input.strategy_id)

    if not strategy_definition_uses_llm(sd):
        splitter = sd.get("splitter", "sentence")
        chunk_size, chunk_overlap = 1024, 20
        if splitter == "token":
            parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc = Document(text=text, id_=doc_id)
        nodes = parser.get_nodes_from_documents([doc])
        for n in nodes:
            n.metadata = getattr(n, "metadata", None) or {}
            n.metadata["strategy"] = strategy_id
            n.metadata.setdefault("source_doc", getattr(n, "ref_doc_id", "") or doc_id)
    else:
        doc = Document(text=text, id_=doc_id)
        parser = LLMNodeParser(
            strategy_id=strategy_id,
            strategy_instruction=sd.get("instruction", ""),
        )
        nodes = parser.get_nodes_from_documents([doc])

    if not nodes:
        return {"status": "skipped", "rows": 0}

    # OpenAI embeddings reject inputs >8192 tokens; oversized chunks need splitting.
    n_before = len(nodes)
    nodes = list(split_long_nodes_for_embedding(nodes))
    if len(nodes) != n_before:
        print(
            f"      [vectorize] split {n_before} → {len(nodes)} chunk(s) for embedding token limits",
            file=sys.stderr,
            flush=True,
        )

    append_chunk_cache(strategy_id, input.content_hash, nodes)

    embed_model = Settings.embed_model
    spans = char_spans_for_nodes(text, nodes)
    chunk_texts: list[str] = []
    for node in nodes:
        if hasattr(node, "get_content"):
            chunk_texts.append(node.get_content(metadata_mode=MetadataMode.NONE))
        else:
            chunk_texts.append(getattr(node, "text", "") or "")

    if hasattr(embed_model, "get_text_embedding_batch"):
        embeddings = embed_model.get_text_embedding_batch(chunk_texts)
    else:
        embeddings = [embed_model.get_text_embedding(t) for t in chunk_texts]

    rows: list[dict[str, Any]] = []
    for node, (span_start, span_end), emb, ct in zip(nodes, spans, embeddings, chunk_texts, strict=True):
        meta = dict(getattr(node, "metadata", None) or {})
        row: dict[str, Any] = {
            "content_hash": input.content_hash,
            "span_start": int(span_start),
            "span_end": int(span_end),
            "embedding": list(emb),
            "metadata": meta,
            "chunk_text": ct,
        }
        sr = meta.get("span_ranges")
        if isinstance(sr, list) and sr:
            row["span_ranges"] = sr
        rows.append(row)

    vs.upsert_rows(
        strategy_fingerprint=input.strategy_fingerprint,
        embedding_fingerprint=input.embedding_fingerprint,
        vector_schema_version=input.vector_schema_version,
        rows=rows,
    )
    print(
        f"      [vectorize] {input.content_hash[:12]}… strategy={strategy_id!r} rows={len(rows)}",
        file=sys.stderr,
        flush=True,
    )
    return {"status": "processed", "rows": len(rows)}

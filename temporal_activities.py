"""
Temporal activities for ReChunk: ingest, vectorize (chunk + embed + VectorStore), logging.

Activities run in workers; they do I/O and LLM calls. Workflows only orchestrate.
Vectorization supports built-in splitter, LLM verbatim chunks, and **derived** (synthetic text + ``source_spans``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from temporalio import activity

from temporal_vectorization_inputs import DocumentVectorizationInput

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import MetadataMode

from rechunk import DerivedNodeParser, LLMNodeParser
from rechunk.strategies import strategy_definition_uses_derived, strategy_definition_uses_llm
from rechunk.cache import append_chunk_cache
from rechunk.node_span_utils import ensure_metadata_source_spans_for_nodes
from rechunk.rag_index import split_long_nodes_for_embedding


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


@activity.defn(name="vectorize_content_for_strategy")
async def vectorize_content_for_strategy(input: DocumentVectorizationInput) -> dict[str, Any]:
    """
    Load extracted text from ECS, chunk (LLM or builtin), dual-write JSONL cache + VectorStore rows.

    Skips if vector rows already exist for (hash, strategy_fp, embed_fp, schema).
    """
    import sys

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

    if strategy_definition_uses_derived(sd):
        doc = Document(text=text, id_=doc_id)
        parser = DerivedNodeParser(
            strategy_id=strategy_id,
            strategy_instruction=sd.get("instruction", ""),
        )
        nodes = parser.get_nodes_from_documents([doc])
    elif strategy_definition_uses_llm(sd):
        doc = Document(text=text, id_=doc_id)
        parser = LLMNodeParser(
            strategy_id=strategy_id,
            strategy_instruction=sd.get("instruction", ""),
        )
        nodes = parser.get_nodes_from_documents([doc])
    else:
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

    ensure_metadata_source_spans_for_nodes(text, nodes)

    append_chunk_cache(strategy_id, input.content_hash, nodes)

    embed_model = Settings.embed_model
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
    for node, emb, ct in zip(nodes, embeddings, chunk_texts, strict=True):
        meta = dict(getattr(node, "metadata", None) or {})
        rows.append(
            {
                "content_hash": input.content_hash,
                "embedding": list(emb),
                "metadata": meta,
                "chunk_text": ct,
            }
        )

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

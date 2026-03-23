#!/usr/bin/env python3
"""
Run ReChunk on a set of documents: build a pooled vector index, then query.

**Default (ECS + VectorStore):** ingests the tree into ``storage/ecs``, takes the active corpus from ECS,
and builds the index from **VectorStore row bundles** (``storage/vector_store_dev``) — same artifacts
the Temporal worker writes. Ingest ECS, then ``scripts/start_strategy_chunking.py <strategy_id>`` to vectorize.

**Legacy:** ``--legacy-jsonl`` reads ``storage/strategies/*_chunks.jsonl`` only (no ECS ingest in the index path).

Usage:
  python scripts/run_with_docs.py --ecs --interactive          # ECS active set only (no docs path; recommended after Temporal ingest)
  python scripts/run_with_docs.py <path> --interactive        # optional: re-sync tree into ECS from disk, then index
  python scripts/run_with_docs.py <path> --legacy-jsonl ...   # index from JSONL chunk caches only
  python scripts/run_with_docs.py --manifest hashes.json ...  # hashes only; VectorStore rows must exist
  python scripts/run_with_ecs.py --interactive                # same as ``run_with_docs.py --ecs ...``

  Provide exactly one corpus source: ``--ecs``, ``--manifest FILE``, or ``<path>``.

Generate a manifest: python scripts/write_corpus_manifest.py <path> out.json

``RECHUNK_OPENAI_EMBEDDING_MODEL`` must match the worker (default ``text-embedding-3-small``).

Vector index disk cache: ``storage/vector_index_cache/`` (separate keys for VectorStore vs JSONL).
Override with RECHUNK_VECTOR_INDEX_CACHE_DIR; disable with --no-vector-index-cache.

Requires OPENAI_API_KEY for LLM synthesis at query time (and for embedding if rows are missing).

Temporal (optional): use ``python temporal_workers.py`` to run **both** ingest and vectorization
queues locally, or ``temporal_workers.py ingest`` / ``vectorization`` for split processes.
Ingest workflows go to ``rechunk-ingest``; vectorization to ``rechunk-strategy-chunking``.
CLI: ``scripts/start_corpus_ingest.py <docs>`` then ``scripts/start_strategy_chunking.py <strategy_id>`` (strategy step uses ECS only).
"""

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Project root and src on path (rechunk + temporal_workflows)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from rechunk.cache import cache_updated_since, get_strategy_cache_mtimes
from rechunk.corpus import ContentRef
from rechunk.corpus_manager import (
    EcsActiveCorpusManager,
    FilesystemCorpusManager,
    HashManifestCorpusManager,
)
from rechunk.fingerprints import compute_strategy_fingerprint
from rechunk.rag_index import (
    load_or_build_vector_index_from_strategies,
    load_or_build_vector_index_from_vector_store,
)
from rechunk.retrieval import retrieve_top_k, synthesize_with_retrieved_nodes
from rechunk.vector_store.freshness import (
    get_vector_store_strategy_mtimes,
    vector_store_cache_updated_since,
)
from rechunk.vectorization_config import OPENAI_EMBEDDING_MODEL, VECTOR_SCHEMA_VERSION
from rechunk.strategies import (
    DEFAULT_BASELINE_STRATEGY,
    Strategy,
    load_strategies,
    save_strategies,
    strategy_to_dict,
)
from rechunk.temporal_client import (
    trigger_pending_vectorization_sync,
    trigger_strategy_chunking_sync,
)

# Strategies metadata file: project root (next to pyproject.toml)
STRATEGIES_FILE = Path(__file__).resolve().parent.parent / "rechunk_strategies.json"

# Max characters of chunk text to show in retrieval feedback
CHUNK_PREVIEW_LEN = 280


def _print_vector_store_fingerprint_mismatch_hint(
    vector_store: Any,
    strategies: list[Strategy],
    *,
    strategies_file: Path,
) -> None:
    """
    If row bundles exist under a different strategy fingerprint than this run expects, explain why.
    """
    list_sfp = getattr(vector_store, "list_row_strategy_fingerprints", None)
    if not callable(list_sfp):
        return
    on_disk = list_sfp()
    if not on_disk:
        return
    expected = [compute_strategy_fingerprint(strategy_to_dict(s)) for s in strategies]
    if any(e in on_disk for e in expected):
        return

    def _short_fp(h: str) -> str:
        return f"{h[:16]}…"

    print("\n" + "=" * 70, flush=True)
    print(
        "  DIAGNOSTIC: VectorStore has row data under a different strategy fingerprint than this CLI.",
        flush=True,
    )
    vs_root = getattr(vector_store, "root", "<vector_store>")
    print(f"    Row bundle dirs under {vs_root}/rows/: ", end="", flush=True)
    print(", ".join(_short_fp(h) for h in on_disk[:8]) + (" …" if len(on_disk) > 8 else ""), flush=True)
    print(f"    This run expects (from {strategies_file.name}):", flush=True)
    for s, e in zip(strategies, expected, strict=True):
        print(f"      {s.id!r} → {_short_fp(e)}", flush=True)
    print(
        "  Strategy fingerprints hash the full strategy JSON (including the builtin `instruction` string).",
        flush=True,
    )
    print(
        "  If you chunked while `rechunk_strategies.json` was missing, older releases used a different",
        flush=True,
    )
    print(
        "  fallback instruction than `run_with_docs` / `run_with_ecs` when they write defaults — paths diverge.",
        flush=True,
    )
    print("  Options:", flush=True)
    print(
        "    • Quick read of rows written by older CLIs: set `instruction` to the literal "
        "`builtin splitter` for `s_default` in rechunk_strategies.json (matches old fallback), or",
        flush=True,
    )
    print(
        "    • Upgrade and re-run `start_strategy_chunking.py` (fallback now matches defaults), or",
        flush=True,
    )
    print(
        "    • Point `rechunk_strategies.json` at the same definition the workflow used (see Temporal history), or",
        flush=True,
    )
    print(
        "    • Remove `storage/vector_store_dev/rows/` and re-vectorize (if you can re-embed).",
        flush=True,
    )
    print("=" * 70 + "\n", flush=True)


def _wait_until_non_empty_index_or_quit(
    strategies: list[Strategy],
    content_refs: Sequence[ContentRef],
    index: VectorStoreIndex,
    nodes: list,
    last_cache_mtimes: dict[str, float],
    *,
    use_disk_cache: bool,
    use_vector_store_rows: bool,
    vector_store: Any | None,
    embedding_fingerprint: str | None,
) -> tuple[VectorStoreIndex, list, dict[str, float]] | None:
    """
    While the pooled index has zero nodes, do not offer the normal question prompt.

    Lets the user reload from cache (or picks up worker writes via mtime) until chunks
    exist, or they quit. Returns None if the user exits early.
    """
    strategy_ids = [s.id for s in strategies]
    print(
        "\nNo embedding chunks are indexed yet — retrieval cannot run.\n"
        "Start the Temporal worker and wait for vectorization "
        + ("(VectorStore rows)" if use_vector_store_rows else "(strategy JSONL cache)")
        + ", then press Enter here to reload (or type r). "
        "The cache is also checked automatically when it changes.\n"
        "Quit with q when done."
    )
    while len(nodes) == 0:
        cache_dirty = False
        if use_vector_store_rows and vector_store is not None and embedding_fingerprint is not None:
            cache_dirty = vector_store_cache_updated_since(
                vector_store,
                strategies,
                content_refs,
                last_cache_mtimes,
                embedding_fingerprint=embedding_fingerprint,
                vector_schema_version=VECTOR_SCHEMA_VERSION,
            )
        else:
            cache_dirty = cache_updated_since(strategy_ids, last_cache_mtimes)

        if cache_dirty:
            try:
                if use_vector_store_rows and vector_store is not None and embedding_fingerprint is not None:
                    index, nodes = load_or_build_vector_index_from_vector_store(
                        vector_store,
                        strategies,
                        content_refs,
                        embed_model=Settings.embed_model,
                        embedding_fingerprint=embedding_fingerprint,
                        vector_schema_version=VECTOR_SCHEMA_VERSION,
                        quiet=True,
                        use_disk_cache=use_disk_cache,
                    )
                    last_cache_mtimes = get_vector_store_strategy_mtimes(
                        vector_store,
                        strategies,
                        content_refs,
                        embedding_fingerprint=embedding_fingerprint,
                        vector_schema_version=VECTOR_SCHEMA_VERSION,
                    )
                else:
                    index, nodes = load_or_build_vector_index_from_strategies(
                        strategies,
                        content_refs,
                        embed_model=Settings.embed_model,
                        quiet=True,
                        use_disk_cache=use_disk_cache,
                    )
                    last_cache_mtimes = get_strategy_cache_mtimes(strategy_ids)
                if len(nodes) > 0:
                    print(f"\n  [Cache updated; index now has {len(nodes)} chunks.]")
                    break
            except Exception as e:
                print(f"\n  [WARN] Cache updated but index rebuild failed: {e}", file=sys.stderr)
                if use_vector_store_rows and vector_store is not None and embedding_fingerprint is not None:
                    last_cache_mtimes = get_vector_store_strategy_mtimes(
                        vector_store,
                        strategies,
                        content_refs,
                        embedding_fingerprint=embedding_fingerprint,
                        vector_schema_version=VECTOR_SCHEMA_VERSION,
                    )
                else:
                    last_cache_mtimes = get_strategy_cache_mtimes(strategy_ids)

        if len(nodes) > 0:
            break

        try:
            cmd = input("\n[Enter] or [r] reload from cache  |  [q] quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return None
        if cmd in ("q", "quit", "exit"):
            print("Bye.")
            return None
        # Enter, "r", "reload" → try rebuild
        try:
            if use_vector_store_rows and vector_store is not None and embedding_fingerprint is not None:
                index, nodes = load_or_build_vector_index_from_vector_store(
                    vector_store,
                    strategies,
                    content_refs,
                    embed_model=Settings.embed_model,
                    embedding_fingerprint=embedding_fingerprint,
                    vector_schema_version=VECTOR_SCHEMA_VERSION,
                    quiet=True,
                    use_disk_cache=use_disk_cache,
                )
                last_cache_mtimes = get_vector_store_strategy_mtimes(
                    vector_store,
                    strategies,
                    content_refs,
                    embedding_fingerprint=embedding_fingerprint,
                    vector_schema_version=VECTOR_SCHEMA_VERSION,
                )
            else:
                index, nodes = load_or_build_vector_index_from_strategies(
                    strategies,
                    content_refs,
                    embed_model=Settings.embed_model,
                    quiet=True,
                    use_disk_cache=use_disk_cache,
                )
                last_cache_mtimes = get_strategy_cache_mtimes(strategy_ids)
            if len(nodes) == 0:
                print(
                    "  Still 0 chunks. Keep the worker running or verify "
                    + (
                        "storage/vector_store_dev row files."
                        if use_vector_store_rows
                        else "storage/strategies cache files."
                    )
                )
        except Exception as e:
            print(f"  [WARN] Index rebuild failed: {e}", file=sys.stderr)

    return index, nodes, last_cache_mtimes


def _enqueue_vectorization_after_new_strategy(
    new_strategy: Strategy,
    *,
    strategies_path: Path,
    docs_root: Path | None,
    doc_ids: list[str] | None,
    enqueue_ecs_vectorization: bool,
    temporal_address: str | None,
) -> None:
    if docs_root is not None and doc_ids:
        trigger_strategy_chunking_sync(
            docs_root, doc_ids, new_strategy, temporal_address=temporal_address
        )
        return
    if not enqueue_ecs_vectorization:
        return
    wid = trigger_pending_vectorization_sync(
        new_strategy,
        temporal_address=temporal_address,
        strategies_path=strategies_path,
    )
    sid = new_strategy.id
    if wid:
        print(
            f"\n  Vectorization queued for {sid!r} (workflow {wid}). "
            "Keep ``python temporal_workers.py`` running; press [r] in the Q&A loop to reload the index.\n"
        )
    else:
        print(
            "\n  No Temporal workflow started (nothing pending, Temporal unreachable, or error). "
            f"You can run: python scripts/start_strategy_chunking.py {sid!r}\n"
        )


def manage_strategies_interactively(
    strategies: list[Strategy],
    strategies_path: Path,
    docs_root: Path | None = None,
    doc_ids: list[str] | None = None,
    *,
    enqueue_ecs_vectorization: bool = False,
    temporal_address: str | None = None,
) -> None:
    """Simple CLI to add/remove strategies (built-in splitters or LLM). Saves after each add/delete.

    If ``docs_root`` and ``doc_ids`` are set (filesystem corpus mode), adding a strategy also calls
    :func:`trigger_strategy_chunking_sync`.

    If ``enqueue_ecs_vectorization`` is True (ECS + VectorStore / ``--ecs`` / manifest), adding a
    strategy enqueues :class:`BatchDocumentVectorizationWorkflow` for pending hashes via
    :func:`trigger_pending_vectorization_sync` — same as ``scripts/start_strategy_chunking.py``.
    """
    while True:
        print("\nCurrent strategies:")
        for i, s in enumerate(strategies, 1):
            extra = f"  splitter={s.splitter}" if s.kind == "builtin_splitter" else ""
            if s.kind == "llm":
                extra = f"  model={getattr(s, 'model', None) or 'gpt-4o-mini'}"
            print(f"  [{i}] {s.id}  (type={s.kind}{extra})  instruction={s.instruction!r}")
        print(
            "\nStrategy menu:\n"
            "  [a] Add new LLM strategy\n"
            "  [b] Add built-in splitter (Sentence or Token)\n"
            "  [d] Delete a strategy\n"
            "  [x] Back (no more changes)\n"
        )
        choice = input("Choice [a/b/d/x]: ").strip().lower()
        if choice in ("x", "", "q", "exit"):
            break
        if choice == "a":
            sid = input("New strategy id (e.g. s_procedures): ").strip()
            if not sid:
                print("No id entered; skipping.")
                continue
            instr = input("Strategy instruction (what semantic unit to chunk around): ").strip()
            if not instr:
                print("No instruction entered; skipping.")
                continue
            model_input = input("Model (default gpt-4o-mini, or e.g. gpt-4o): ").strip()
            model = model_input if model_input else None
            strategies.append(Strategy(id=sid, kind="llm", instruction=instr, model=model))
            save_strategies(strategies_path, strategies)
            print(f"Added LLM strategy {sid!r} (model={model or 'gpt-4o-mini'}). Saved to {strategies_path.name}")
            _enqueue_vectorization_after_new_strategy(
                strategies[-1],
                strategies_path=strategies_path,
                docs_root=docs_root,
                doc_ids=doc_ids,
                enqueue_ecs_vectorization=enqueue_ecs_vectorization,
                temporal_address=temporal_address,
            )
        elif choice == "b":
            which = input("Built-in splitter: [1] SentenceSplitter  [2] TokenTextSplitter  (1 or 2): ").strip()
            if which == "2":
                sid = "s_token"
                splitter = "token"
                instr = "Token-based splitting (LlamaIndex default chunk_size=1024, overlap=20)"
            else:
                sid = "s_sentence"
                splitter = "sentence"
                instr = "Sentence-based splitting (LlamaIndex default chunk_size=1024, overlap=20)"
            strategies.append(
                Strategy(id=sid, kind="builtin_splitter", instruction=instr, splitter=splitter),
            )
            save_strategies(strategies_path, strategies)
            print(f"Added built-in splitter {sid!r} ({splitter}). Saved to {strategies_path.name}")
            _enqueue_vectorization_after_new_strategy(
                strategies[-1],
                strategies_path=strategies_path,
                docs_root=docs_root,
                doc_ids=doc_ids,
                enqueue_ecs_vectorization=enqueue_ecs_vectorization,
                temporal_address=temporal_address,
            )
        elif choice == "d":
            idx_str = input("Enter number of strategy to delete (or blank to cancel): ").strip()
            if not idx_str:
                continue
            try:
                idx = int(idx_str)
            except ValueError:
                print("Invalid number.")
                continue
            if not (1 <= idx <= len(strategies)):
                print("Out of range.")
                continue
            removed = strategies.pop(idx - 1)
            save_strategies(strategies_path, strategies)
            print(f"Removed strategy {removed.id!r}. Saved to {strategies_path.name}")
        else:
            print("Unknown choice; please enter a/b/d/x.")


def run_query_with_feedback(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 5,
    total_chunks: int | None = None,
    strategy_ids: list[str] | None = None,
) -> None:
    """
    Run retrieval (embedding comparison), show chunks + scores, then LLM synthesis.
    Prints timing and explains cosine-similarity retrieval vs LLM "dress-up".
    When total_chunks/strategy_ids are provided, shows that retrieval is over all strategies.
    """
    # ---- Retrieval (embedding comparison) ----
    print("\n" + "=" * 60)
    print("RETRIEVAL (embedding comparison)")
    print("=" * 60)
    print(
        "Your question is embedded, then compared to each chunk's embedding via"
        " cosine similarity (higher score = more similar). No LLM call yet."
    )
    if total_chunks is not None and strategy_ids:
        print(f"  Pool: {total_chunks} chunks from all strategies combined: {', '.join(strategy_ids)}")
    t0 = time.perf_counter()
    nodes_with_scores = retrieve_top_k(index, query, top_k=top_k)
    t1 = time.perf_counter()
    print(f"  Time: {(t1 - t0) * 1000:.0f} ms")
    print(f"  Top {len(nodes_with_scores)} chunks:")
    for i, nws in enumerate(nodes_with_scores, 1):
        node = nws.node
        score = getattr(nws, "score", None)
        score_str = f"  score={score:.4f}" if score is not None else "  (no score)"
        if hasattr(node, "get_content"):
            text = node.get_content(metadata_mode=MetadataMode.NONE)
        else:
            text = getattr(node, "text", str(node))
        preview = (text[:CHUNK_PREVIEW_LEN] + "…") if len(text) > CHUNK_PREVIEW_LEN else text
        meta = getattr(node, "metadata", None) or {}
        source = getattr(node, "ref_doc_id", None) or meta.get("source_doc", "?")
        strategy = meta.get("strategy", "?")
        start = getattr(node, "start_char_idx", None)
        end = getattr(node, "end_char_idx", None)
        if start is not None and end is not None:
            loc = f"  source={source!r}  strategy={strategy}  chars {start}–{end}"
        else:
            loc = f"  source={source!r}  strategy={strategy}  (position in doc not stored)"
        print(f"    {i}.{score_str}{loc}")
        print(f"       {preview!r}")
    print()

    # ---- LLM synthesis ----
    print("=" * 60)
    print("LLM RESPONSE (synthesis from retrieved chunks)")
    print("=" * 60)
    print("The LLM sees your question + the chunks above and produces an answer.")
    t2 = time.perf_counter()
    response = synthesize_with_retrieved_nodes(index, query, nodes_with_scores)
    t3 = time.perf_counter()
    print(f"  Time: {(t3 - t2) * 1000:.0f} ms")
    answer = getattr(response, "response", str(response))
    print(f"  Answer: {answer}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ReChunk on documents: chunk with LLM strategy, build index, optional query."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Filesystem corpus root or file (omit with --ecs or --manifest)",
    )
    parser.add_argument(
        "--ecs",
        action="store_true",
        help="Use ECS active corpus only (no docs path; no CLI re-ingest from disk).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON file: array of SHA-256 hex strings only, or {\"content_hashes\": [...]}",
    )
    parser.add_argument(
        "--strategy-id",
        default="s_sections",
        help="Strategy id for optional LLM strategy (used only with --strategy)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="If set, add an LLM chunking strategy (instruction text). Default is no LLM; uses LlamaIndex SentenceSplitter only.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="If set, run this query against the index and print the response",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="After chunking/indexing once, prompt for questions in a loop (fast retrieval feel)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for chunking (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-vector-index-cache",
        action="store_true",
        help="Always recompute embeddings; do not load/save disk cache (see RECHUNK_VECTOR_INDEX_CACHE_DIR)",
    )
    parser.add_argument(
        "--legacy-jsonl",
        action="store_true",
        help="Build the index from storage/strategies JSONL only (skip ECS ingest + VectorStore row assembly).",
    )
    args = parser.parse_args()

    n_sources = sum(
        [
            args.path is not None,
            args.manifest is not None,
            bool(args.ecs),
        ]
    )
    if n_sources != 1:
        parser.error(
            "Provide exactly one corpus source: <path>, --manifest PATH, or --ecs (not multiple, not none)."
        )

    manifest_mode = args.manifest is not None
    ecs_only_mode = bool(args.ecs)

    # Use OpenAI for LLM and embeddings (reads OPENAI_API_KEY from env)
    Settings.llm = OpenAI(model=args.model, temperature=0.1)
    embed_model_name = os.environ.get("RECHUNK_OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=embed_model_name)

    use_disk_cache = not args.no_vector_index_cache
    use_vector_store_rows = not args.legacy_jsonl

    if ecs_only_mode:
        corpus_manager: (
            EcsActiveCorpusManager | FilesystemCorpusManager | HashManifestCorpusManager
        ) = EcsActiveCorpusManager()
    elif manifest_mode:
        corpus_manager = HashManifestCorpusManager(args.manifest)
    else:
        corpus_manager = FilesystemCorpusManager(Path(args.path))

    content_refs = corpus_manager.list_active_content_refs()
    hints = corpus_manager.temporal_ingest_hints()
    if hints is not None:
        docs_root, doc_ids = hints.docs_root, hints.doc_ids
    else:
        docs_root, doc_ids = None, []
    print(corpus_manager.summary_message(len(content_refs)))

    vector_store: Any | None = None
    embedding_fp: str | None = None
    if use_vector_store_rows:
        from rechunk.extracted_content import FilesystemExtractedContentService
        from rechunk.index_service import IndexService
        from rechunk.vector_store import FilesystemVectorStore

        vector_store = FilesystemVectorStore()
        ecs = FilesystemExtractedContentService()
        index_svc = IndexService(ecs=ecs, vector_store=vector_store, strategies_path=STRATEGIES_FILE)
        embedding_fp = index_svc.embedding_fingerprint()
        if not manifest_mode and not ecs_only_mode and hints is not None:
            index_svc.ingest_filesystem_docs(hints.docs_root, hints.doc_ids)
            ecs.apply_source_inventory("filesystem", hints.doc_ids)
            index_svc.sync_active_manifest_file()
            content_refs = ecs.list_active_content_refs()
        elif ecs_only_mode:
            content_refs = ecs.list_active_content_refs()
        print(
            f"ECS + VectorStore index path: {len(content_refs)} active content hash(es); "
            f"embedding fingerprint {embedding_fp[:12]}… (must match worker).",
            flush=True,
        )
    else:
        print("Legacy index path: JSONL chunk caches under storage/strategies/.", flush=True)

    # Load saved strategies if present; otherwise use default baseline and cue worker.
    strategies = load_strategies(STRATEGIES_FILE)
    if strategies is None:
        print()
        print("=" * 70)
        print("  WARNING: No strategy file found (or file empty/invalid).")
        print("  Using the default BASELINE strategy only (built-in splitter, NOT LLM).")
        if manifest_mode:
            print("  Manifest mode: this tool will not start a Temporal workflow (no path list on wire).")
            print("  Run scripts/start_corpus_ingest.py then scripts/start_strategy_chunking.py <strategy_id>.")
        else:
            print("  The worker will be cued to backfill chunks for this strategy.")
        print("  Ensure the Temporal workers are running (python temporal_workers.py).")
        print("=" * 70)
        print()
        strategies = [DEFAULT_BASELINE_STRATEGY]
        # Write a strategy file so the active strategy set is explicit/persistent.
        try:
            save_strategies(STRATEGIES_FILE, strategies)
            print(f"Wrote default strategies to {STRATEGIES_FILE.resolve()}")
            if not STRATEGIES_FILE.exists():
                print(
                    f"[WARN] Tried to write {STRATEGIES_FILE.resolve()} but it still does not exist.",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"[WARN] Could not write {STRATEGIES_FILE.resolve()}: {e}", file=sys.stderr)
        if args.strategy:
            strategies.append(
                Strategy(id=args.strategy_id, kind="llm", instruction=args.strategy),
            )
    else:
        print(f"Loaded {len(strategies)} strategy(ies) from {STRATEGIES_FILE.name}")
        if args.strategy:
            strategies.append(
                Strategy(id=args.strategy_id, kind="llm", instruction=args.strategy),
            )

    # Filesystem corpus + ECS/VectorStore path: queue batch vectorization for every strategy
    # (pending hashes only). Previously only happened when no strategy file existed.
    if (
        use_vector_store_rows
        and not manifest_mode
        and not ecs_only_mode
        and docs_root is not None
    ):
        temporal_addr = os.environ.get("TEMPORAL_ADDRESS")
        for s in strategies:
            trigger_pending_vectorization_sync(
                s,
                temporal_address=temporal_addr,
                strategies_path=STRATEGIES_FILE,
            )

    from rechunk.vector_store.freshness import get_vector_store_strategy_mtimes

    def rebuild_index(*, quiet: bool) -> tuple[VectorStoreIndex, list]:
        if use_vector_store_rows and vector_store is not None and embedding_fp is not None:
            return load_or_build_vector_index_from_vector_store(
                vector_store,
                strategies,
                content_refs,
                embed_model=Settings.embed_model,
                embedding_fingerprint=embedding_fp,
                vector_schema_version=VECTOR_SCHEMA_VERSION,
                quiet=quiet,
                use_disk_cache=use_disk_cache,
            )
        return load_or_build_vector_index_from_strategies(
            strategies,
            content_refs,
            embed_model=Settings.embed_model,
            quiet=quiet,
            use_disk_cache=use_disk_cache,
        )

    def snapshot_cache_mtimes() -> dict[str, float]:
        if use_vector_store_rows and vector_store is not None and embedding_fp is not None:
            return get_vector_store_strategy_mtimes(
                vector_store,
                strategies,
                content_refs,
                embedding_fingerprint=embedding_fp,
                vector_schema_version=VECTOR_SCHEMA_VERSION,
            )
        return get_strategy_cache_mtimes([s.id for s in strategies])

    index, nodes = rebuild_index(quiet=False)
    print(f"ReChunk produced {len(nodes)} nodes across {len(strategies)} strategy(ies)")
    print("Index ready (embeddings from cache or API).")

    if use_vector_store_rows and vector_store is not None and len(nodes) == 0:
        _print_vector_store_fingerprint_mismatch_hint(
            vector_store,
            strategies,
            strategies_file=STRATEGIES_FILE,
        )

    if args.interactive:
        last_cache_mtimes = snapshot_cache_mtimes()
        if len(nodes) == 0:
            waited = _wait_until_non_empty_index_or_quit(
                strategies,
                content_refs,
                index,
                nodes,
                last_cache_mtimes,
                use_disk_cache=use_disk_cache,
                use_vector_store_rows=use_vector_store_rows,
                vector_store=vector_store,
                embedding_fingerprint=embedding_fp,
            )
            if waited is None:
                return
            index, nodes, last_cache_mtimes = waited
        print(
            f"\n{len(nodes)} chunks formed from {len(strategies)} strategy(ies). "
            "Enter a question (or 'quit'/'q' to exit). Index is rebuilt automatically when the worker updates the cache."
        )
        while True:
            cache_dirty = False
            if use_vector_store_rows and vector_store is not None and embedding_fp is not None:
                cache_dirty = vector_store_cache_updated_since(
                    vector_store,
                    strategies,
                    content_refs,
                    last_cache_mtimes,
                    embedding_fingerprint=embedding_fp,
                    vector_schema_version=VECTOR_SCHEMA_VERSION,
                )
            else:
                cache_dirty = cache_updated_since([s.id for s in strategies], last_cache_mtimes)

            if cache_dirty:
                try:
                    index, nodes = rebuild_index(quiet=True)
                    last_cache_mtimes = snapshot_cache_mtimes()
                    print(f"\n  [Cache updated by worker; index rebuilt with {len(nodes)} chunks.]")
                    if len(nodes) == 0:
                        print(
                            "  (Index is still empty — use [r] reload or wait for more cache writes before asking.)"
                        )
                except Exception as e:
                    # Don't crash the session if embeddings fail; keep the previous index.
                    print(f"\n  [WARN] Cache updated but index rebuild failed: {e}", file=sys.stderr)
            try:
                q = input("\nYour question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not q:
                # Empty input should just reprompt (don't exit).
                continue
            if q.lower() in ("quit", "q", "exit"):
                print("Bye.")
                break
            if q.lower() in ("reload", "r"):
                index, nodes = rebuild_index(quiet=True)
                last_cache_mtimes = snapshot_cache_mtimes()
                print(f"Index rebuilt: {len(nodes)} chunks.")
                if len(nodes) == 0:
                    print("  Index is still empty; questions are disabled until chunks exist.")
                    waited = _wait_until_non_empty_index_or_quit(
                        strategies,
                        content_refs,
                        index,
                        nodes,
                        last_cache_mtimes,
                        use_disk_cache=use_disk_cache,
                        use_vector_store_rows=use_vector_store_rows,
                        vector_store=vector_store,
                        embedding_fingerprint=embedding_fp,
                    )
                    if waited is None:
                        break
                    index, nodes, last_cache_mtimes = waited
                continue
            if len(nodes) == 0:
                print(
                    "  No chunks in the index; cannot answer. Use [r] reload or wait for the worker, then try again."
                )
                continue
            run_query_with_feedback(
                index, q, top_k=args.top_k,
                total_chunks=len(nodes), strategy_ids=[s.id for s in strategies],
            )
            # Ask for feedback and optionally adjust strategies
            fb = input(
                "Was this answer correct? [y]es / [n]o / [i]ncomplete / [s]kip: "
            ).strip().lower()
            if fb in ("n", "i"):
                manage_strategies_interactively(
                    strategies,
                    STRATEGIES_FILE,
                    docs_root=docs_root,
                    doc_ids=doc_ids,
                    enqueue_ecs_vectorization=use_vector_store_rows,
                    temporal_address=os.environ.get("TEMPORAL_ADDRESS"),
                )
                # Rebuild index after any strategy changes
                index, nodes = rebuild_index(quiet=False)
                print(
                    f"\nRebuilt index: {len(nodes)} chunks from {len(strategies)} strategy(ies)."
                )
    elif args.query:
        if len(nodes) == 0:
            print(
                "Cannot run --query: the index has no chunks yet (vector rows or JSONL cache missing).",
                file=sys.stderr,
            )
            print(
                "Start the Temporal worker, ingest ECS, run scripts/start_strategy_chunking.py <strategy_id>, then retry.",
                file=sys.stderr,
            )
            sys.exit(1)
        run_query_with_feedback(
            index, args.query, top_k=args.top_k,
            total_chunks=len(nodes), strategy_ids=[s.id for s in strategies],
        )
    else:
        print("\nNo --query given. Use --query \"your question\" or --interactive to ask questions.")


if __name__ == "__main__":
    main()

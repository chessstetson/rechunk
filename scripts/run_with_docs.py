#!/usr/bin/env python3
"""
Run ReChunk on a set of documents: build index from cache, then query.

Chunking is done by the Temporal worker only. Run the worker and scripts/start_strategy_chunking.py
to backfill chunks for each strategy; this script only reads from storage/strategies/ and never runs
chunking (LLM or built-in) itself.

Usage:
  python scripts/run_with_docs.py <path>                     # filesystem corpus → index from cache
  python scripts/run_with_docs.py --manifest hashes.json ...   # hash-only manifest (no paths on wire)
  python scripts/run_with_docs.py <path> --query "?"          # one query with retrieval + LLM
  python scripts/run_with_docs.py <path> --interactive       # prompt for questions

Generate a manifest: python scripts/write_corpus_manifest.py <path> out.json

After a successful StrategyChunkingWorkflow, the worker merges manifest hashes into
storage/corpus_content_hashes.json (override with RECHUNK_ACTIVE_CORPUS_MANIFEST).
Then: python scripts/run_with_docs.py --manifest storage/corpus_content_hashes.json --interactive

Requires OPENAI_API_KEY for LLM synthesis at query time.
"""

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

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
from rechunk.corpus_manager import FilesystemCorpusManager, HashManifestCorpusManager
from rechunk.rag_index import build_vector_index_from_strategies
from rechunk.retrieval import retrieve_top_k, synthesize_with_retrieved_nodes
from rechunk.strategies import (
    DEFAULT_BASELINE_STRATEGY,
    Strategy,
    load_strategies,
    save_strategies,
)
from rechunk.temporal_client import trigger_strategy_chunking_sync

# Strategies metadata file: project root (next to pyproject.toml)
STRATEGIES_FILE = Path(__file__).resolve().parent.parent / "rechunk_strategies.json"

# Max characters of chunk text to show in retrieval feedback
CHUNK_PREVIEW_LEN = 280


def _wait_until_non_empty_index_or_quit(
    strategies: list[Strategy],
    content_refs: Sequence[ContentRef],
    index: VectorStoreIndex,
    nodes: list,
    last_cache_mtimes: dict[str, float],
) -> tuple[VectorStoreIndex, list, dict[str, float]] | None:
    """
    While the pooled index has zero nodes, do not offer the normal question prompt.

    Lets the user reload from cache (or picks up worker writes via mtime) until chunks
    exist, or they quit. Returns None if the user exits early.
    """
    strategy_ids = [s.id for s in strategies]
    print(
        "\nNo embedding chunks are indexed yet — retrieval cannot run.\n"
        "Start the Temporal worker and wait for the strategy cache to backfill, then press "
        "Enter here to reload from cache (or type r). The cache is also checked automatically "
        "when it changes.\n"
        "Quit with q when done."
    )
    while len(nodes) == 0:
        if cache_updated_since(strategy_ids, last_cache_mtimes):
            try:
                index, nodes = build_vector_index_from_strategies(strategies, content_refs, quiet=True)
                last_cache_mtimes = get_strategy_cache_mtimes(strategy_ids)
                if len(nodes) > 0:
                    print(f"\n  [Cache updated; index now has {len(nodes)} chunks.]")
                    break
            except Exception as e:
                print(f"\n  [WARN] Cache updated but index rebuild failed: {e}", file=sys.stderr)
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
            index, nodes = build_vector_index_from_strategies(strategies, content_refs, quiet=True)
            last_cache_mtimes = get_strategy_cache_mtimes(strategy_ids)
            if len(nodes) == 0:
                print("  Still 0 chunks. Keep the worker running or verify storage/strategies cache files.")
        except Exception as e:
            print(f"  [WARN] Index rebuild failed: {e}", file=sys.stderr)

    return index, nodes, last_cache_mtimes


def manage_strategies_interactively(
    strategies: list[Strategy],
    strategies_path: Path,
    docs_root: Path | None = None,
    doc_ids: list[str] | None = None,
) -> None:
    """Simple CLI to add/remove strategies (built-in splitters or LLM). Saves after each add/delete.
    If docs_root and doc_ids are provided, starting a new strategy also triggers the Temporal workflow."""
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
            if docs_root is not None and doc_ids:
                trigger_strategy_chunking_sync(docs_root, doc_ids, strategies[-1])
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
            if docs_root is not None and doc_ids:
                trigger_strategy_chunking_sync(docs_root, doc_ids, strategies[-1])
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
        help="Filesystem corpus root or file (omit if using --manifest)",
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
    args = parser.parse_args()

    if (args.path is None) == (args.manifest is None):
        parser.error("Provide exactly one of: corpus path OR --manifest PATH (not both, not neither).")

    manifest_mode = args.manifest is not None

    # Use OpenAI for LLM and embeddings (reads OPENAI_API_KEY from env)
    Settings.llm = OpenAI(model=args.model, temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    if manifest_mode:
        corpus_manager: FilesystemCorpusManager | HashManifestCorpusManager = HashManifestCorpusManager(
            args.manifest
        )
    else:
        corpus_manager = FilesystemCorpusManager(Path(args.path))

    content_refs = corpus_manager.list_active_content_refs()
    hints = corpus_manager.temporal_ingest_hints()
    if hints is not None:
        docs_root, doc_ids = hints.docs_root, hints.doc_ids
    else:
        docs_root, doc_ids = None, []
    print(corpus_manager.summary_message(len(content_refs)))

    # Load saved strategies if present; otherwise use default baseline and cue worker.
    strategies = load_strategies(STRATEGIES_FILE)
    if strategies is None:
        print()
        print("=" * 70)
        print("  WARNING: No strategy file found (or file empty/invalid).")
        print("  Using the default BASELINE strategy only (built-in splitter, NOT LLM).")
        if manifest_mode:
            print("  Manifest mode: this tool will not start a Temporal workflow (no path list on wire).")
            print("  Run scripts/start_strategy_chunking.py with your corpus root, or use a filesystem path once.")
        else:
            print("  The worker will be cued to backfill chunks for this strategy.")
        print("  Ensure the Temporal worker is running (python temporal_worker.py).")
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
        if not manifest_mode and docs_root is not None:
            trigger_strategy_chunking_sync(docs_root, doc_ids, DEFAULT_BASELINE_STRATEGY)
    else:
        print(f"Loaded {len(strategies)} strategy(ies) from {STRATEGIES_FILE.name}")
        if args.strategy:
            strategies.append(
                Strategy(id=args.strategy_id, kind="llm", instruction=args.strategy),
            )

    index, nodes = build_vector_index_from_strategies(strategies, content_refs)
    print(f"ReChunk produced {len(nodes)} nodes across {len(strategies)} strategy(ies)")
    print("Index built (embeddings computed).")

    if args.interactive:
        last_cache_mtimes = get_strategy_cache_mtimes([s.id for s in strategies])
        if len(nodes) == 0:
            waited = _wait_until_non_empty_index_or_quit(
                strategies, content_refs, index, nodes, last_cache_mtimes
            )
            if waited is None:
                return
            index, nodes, last_cache_mtimes = waited
        print(
            f"\n{len(nodes)} chunks formed from {len(strategies)} strategy(ies). "
            "Enter a question (or 'quit'/'q' to exit). Index is rebuilt automatically when the worker updates the cache."
        )
        while True:
            # Before each prompt: if worker wrote new chunks, rebuild index from cache (no user action needed).
            if cache_updated_since([s.id for s in strategies], last_cache_mtimes):
                try:
                    index, nodes = build_vector_index_from_strategies(strategies, content_refs, quiet=True)
                    last_cache_mtimes = get_strategy_cache_mtimes([s.id for s in strategies])
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
                index, nodes = build_vector_index_from_strategies(strategies, content_refs, quiet=True)
                last_cache_mtimes = get_strategy_cache_mtimes([s.id for s in strategies])
                print(f"Index rebuilt: {len(nodes)} chunks.")
                if len(nodes) == 0:
                    print("  Index is still empty; questions are disabled until chunks exist.")
                    waited = _wait_until_non_empty_index_or_quit(
                        strategies, content_refs, index, nodes, last_cache_mtimes
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
                    strategies, STRATEGIES_FILE,
                    docs_root=docs_root,
                    doc_ids=doc_ids,
                )
                # Rebuild index after any strategy changes
                index, nodes = build_vector_index_from_strategies(strategies, content_refs)
                print(
                    f"\nRebuilt index: {len(nodes)} chunks from {len(strategies)} strategy(ies)."
                )
    elif args.query:
        if len(nodes) == 0:
            print(
                "Cannot run --query: the index has no chunks yet (strategy cache is empty or not backfilled).",
                file=sys.stderr,
            )
            print(
                "Start the Temporal worker, wait for storage/strategies to fill, then retry.",
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

#!/usr/bin/env python3
"""
Run ReChunk on a set of documents: load docs, chunk with an LLM strategy, build index, query.

Usage:
  python scripts/run_with_docs.py <path>                    # chunk + index, then remind to use --query
  python scripts/run_with_docs.py <path> --query "?"        # one query with retrieval + LLM feedback
  python scripts/run_with_docs.py <path> --interactive       # chunk + index once, then prompt for questions

Requires OPENAI_API_KEY in the environment.
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Project root on path so "rechunk" resolves when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.schema import MetadataMode, QueryBundle
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from rechunk import LLMNodeParser

# Max characters of chunk text to show in retrieval feedback
CHUNK_PREVIEW_LEN = 280


@dataclass
class Strategy:
    """In-memory representation of a chunking strategy."""

    id: str
    kind: str  # "builtin_splitter" (no LLM) or "llm"
    instruction: str
    # For kind="builtin_splitter": which LlamaIndex parser. "sentence" | "token"
    splitter: str = "sentence"
    # For kind="llm": OpenAI model name (default gpt-4o-mini when None)
    model: str | None = None


# Strategies file: project root (next to pyproject.toml)
STRATEGIES_FILE = Path(__file__).resolve().parent.parent / "rechunk_strategies.json"


def _strategy_to_dict(s: Strategy) -> dict:
    return {
        "id": s.id,
        "kind": s.kind,
        "instruction": s.instruction,
        "splitter": getattr(s, "splitter", "sentence"),
        "model": getattr(s, "model", None),
    }


def _dict_to_strategy(d: dict) -> Strategy:
    return Strategy(
        id=d["id"],
        kind=d["kind"],
        instruction=d["instruction"],
        splitter=d.get("splitter", "sentence"),
        model=d.get("model"),
    )


def load_strategies(path: Path) -> list[Strategy] | None:
    """Load strategy set from JSON file. Returns None if file missing or invalid."""
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list) or len(data) == 0:
            return None
        return [_dict_to_strategy(item) for item in data]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def save_strategies(path: Path, strategies: list[Strategy]) -> None:
    """Serialize strategy set to JSON."""
    data = [_strategy_to_dict(s) for s in strategies]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _extract_file_content(path: Path) -> tuple[str | None, str]:
    """Extract text content from supported file types.

    Returns (content, description). content is None if the file type isn't supported
    or required libraries are missing.
    """
    import os

    ext = path.suffix.lower()
    file_size = path.stat().st_size

    try:
        if ext == ".pdf":
            try:
                import PyPDF2  # type: ignore[import]

                with path.open("rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text_parts: list[str] = []
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        text_parts.append(page_text)
                content = "\n".join(text_parts)
                return content, f"PDF document ({file_size:,} bytes)"
            except ImportError:
                print(f"[WARN] PyPDF2 not installed; skipping PDF {path}", file=sys.stderr)
                return None, "PDF document (PyPDF2 not installed)"

        if ext == ".docx":
            try:
                import docx  # type: ignore[import]

                doc = docx.Document(str(path))
                paragraphs = [p.text for p in doc.paragraphs]
                content = "\n".join(paragraphs)
                return content, f"Word document ({file_size:,} bytes)"
            except ImportError:
                print(f"[WARN] python-docx not installed; skipping DOCX {path}", file=sys.stderr)
                return None, "Word document (python-docx not installed)"

        # Plain text-like files
        if ext in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text, f"Text file ({ext})"

        # Fallback: try as text
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                return text, f"Unknown text file ({ext})"
        except Exception:
            pass

        return None, f"Unsupported or binary file ({ext})"
    except Exception as e:
        return None, f"Error reading {path}: {e}"


def _content_hash(content: str) -> str:
    """SHA-256 hash of document content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_documents(path: Path) -> list[Document]:
    """Load supported files from a file or directory into LlamaIndex Documents.

    For a directory, scans the entire directory tree recursively (all subdirectories).
    Supports .txt, .md, .pdf, .docx. Other text-like files are best-effort.
    Documents with identical content (same hash) are loaded only once.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    docs: list[Document] = []
    seen_hashes: set[str] = set()

    if path.is_file():
        content, desc = _extract_file_content(path)
        if content:
            h = _content_hash(content)
            docs.append(
                Document(
                    text=content,
                    id_=path.name,
                    metadata={"content_hash": h},
                )
            )
        else:
            raise FileNotFoundError(f"Unable to read file {path}: {desc}")
    else:
        # Recursively scan whole tree (all subdirectories)
        candidates = sorted(path.rglob("*"))
        for f in (p for p in candidates if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}):
            content, desc = _extract_file_content(f)
            if content:
                h = _content_hash(content)
                if h in seen_hashes:
                    print(f"[SKIP] Duplicate content (same hash): {f.relative_to(path)}", file=sys.stderr)
                    continue
                seen_hashes.add(h)
                doc_id = str(f.relative_to(path))
                docs.append(
                    Document(
                        text=content,
                        id_=doc_id,
                        metadata={"content_hash": h},
                    )
                )
            else:
                print(f"[WARN] Skipping {f}: {desc}", file=sys.stderr)
        if not docs:
            raise FileNotFoundError(f"No supported files under {path}")

    return docs


def build_index_for_strategies(
    strategies: list[Strategy],
    docs: list[Document],
) -> tuple[VectorStoreIndex, list]:
    """Run all active strategies over docs and build a combined index."""
    from llama_index.core.schema import BaseNode

    all_nodes: list[BaseNode] = []
    n_docs = len(docs)
    for idx, s in enumerate(strategies, 1):
        print(f"\n  Strategy {idx}/{len(strategies)}: {s.id} ({s.kind}) — {n_docs} documents")
        if s.kind == "builtin_splitter":
            # LlamaIndex defaults: chunk_size=1024, chunk_overlap=20 (tokens)
            chunk_size, chunk_overlap = 1024, 20
            if getattr(s, "splitter", "sentence") == "token":
                parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            else:
                parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            nodes = parser.get_nodes_from_documents(docs, show_progress=True)
            for n in nodes:
                n.metadata = getattr(n, "metadata", None) or {}
                n.metadata["strategy"] = s.id
                n.metadata.setdefault("source_doc", getattr(n, "ref_doc_id", ""))
            all_nodes.extend(nodes)
            print(f"    → {len(nodes)} chunks")
        elif s.kind == "llm":
            model = getattr(s, "model", None) or "gpt-4o-mini"
            llm = OpenAI(model=model, temperature=0.1)
            parser = LLMNodeParser(
                strategy_id=s.id,
                strategy_instruction=s.instruction,
                llm=llm,
            )
            nodes = parser.get_nodes_from_documents(docs, show_progress=True)
            all_nodes.extend(nodes)
            print(f"    → {len(nodes)} chunks")
    print(f"\n  Total: {len(all_nodes)} chunks from {len(strategies)} strategy(ies)")
    index = VectorStoreIndex(all_nodes)
    return index, all_nodes


def manage_strategies_interactively(
    strategies: list[Strategy],
    strategies_path: Path,
) -> None:
    """Simple CLI to add/remove strategies (built-in splitters or LLM). Saves after each add/delete."""
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
    retriever = index.as_retriever(similarity_top_k=top_k)
    query_bundle = QueryBundle(query_str=query)

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
    nodes_with_scores = retriever.retrieve(query)
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
    engine = index.as_query_engine()
    t2 = time.perf_counter()
    response = engine.synthesize(query_bundle, nodes_with_scores)
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
        type=Path,
        help="Path to a .txt file or directory containing .txt files",
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

    # Use OpenAI for LLM and embeddings (reads OPENAI_API_KEY from env)
    Settings.llm = OpenAI(model=args.model, temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    docs = load_documents(args.path)
    print(f"Loaded {len(docs)} document(s) from {args.path}")

    # Load saved strategies if present; otherwise default to one built-in splitter.
    strategies = load_strategies(STRATEGIES_FILE)
    if strategies is None:
        strategies = [
            Strategy(
                id="s_default",
                kind="builtin_splitter",
                instruction="Sentence-based splitting (LlamaIndex default, chunk_size=1024)",
                splitter="sentence",
            ),
        ]
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

    index, nodes = build_index_for_strategies(strategies, docs)
    print(f"ReChunk produced {len(nodes)} nodes across {len(strategies)} strategy(ies)")
    print("Index built (embeddings computed).")

    if args.interactive:
        print(
            f"\n{len(nodes)} chunks formed from {len(strategies)} strategy(ies). "
            "Enter a question (or 'quit' / 'q' to exit). You'll see retrieval then LLM response each time."
        )
        while True:
            try:
                q = input("\nYour question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not q or q.lower() in ("quit", "q", "exit"):
                print("Bye.")
                break
            run_query_with_feedback(
                index, q, top_k=args.top_k,
                total_chunks=len(nodes), strategy_ids=[s.id for s in strategies],
            )
            # Ask for feedback and optionally adjust strategies
            fb = input(
                "Was this answer correct? [y]es / [n]o / [i]ncomplete / [s]kip: "
            ).strip().lower()
            if fb in ("n", "i"):
                manage_strategies_interactively(strategies, STRATEGIES_FILE)
                # Rebuild index after any strategy changes
                index, nodes = build_index_for_strategies(strategies, docs)
                print(
                    f"\nRebuilt index: {len(nodes)} chunks from {len(strategies)} strategy(ies)."
                )
    elif args.query:
        run_query_with_feedback(
            index, args.query, top_k=args.top_k,
            total_chunks=len(nodes), strategy_ids=[s.id for s in strategies],
        )
    else:
        print("\nNo --query given. Use --query \"your question\" or --interactive to ask questions.")


if __name__ == "__main__":
    main()

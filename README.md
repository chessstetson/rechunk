# ReChunk

**Adaptive, feedback-driven RAG chunking** — an extension for [LlamaIndex](https://www.llamaindex.ai/) that treats chunking as a living, strategy-driven process instead of a one-time split.

- **Index = f(corpus, S)** — the index is a pure function of your documents and a set of chunking strategies.
- **Strategy layers** — each strategy is a natural-language instruction; an LLM does the chunking. Chunks are tagged by strategy for multi-layer retrieval (v0.2+).
- **Feedback loop (roadmap)** — poor answers trigger diagnosis and, when the answer exists in the corpus, proposal of new strategies.

See **[rechunk_strategy.md](rechunk_strategy.md)** for the full design, lifecycle, and roadmap.

## Install

```bash
pip install -e .
```

Requires Python 3.10+. Optional: set `OPENAI_API_KEY` (or configure another LLM via LlamaIndex `Settings.llm`).

## Run with your own docs

From the project root (with `OPENAI_API_KEY` set and the package installed, e.g. in a venv):

```bash
# Chunk a directory of .txt files (or a single .txt file)
python scripts/run_with_docs.py path/to/your/docs

# Chunk and run a query (with retrieval + LLM feedback: chunks, scores, then answer)
python scripts/run_with_docs.py path/to/your/docs --query "What is the main idea?"

# Interactive: chunk once, then ask questions in a loop (feel how fast embedding retrieval is)
python scripts/run_with_docs.py path/to/your/docs --interactive
```

Use `docs` for the included sample. With `--query` or `--interactive`, the script shows **retrieval** (embedding cosine similarity, which chunks were picked, timing) and then the **LLM response** (synthesis from those chunks, timing). Options: `--strategy-id`, `--strategy`, `--model`, `--query`, `--interactive`, `--top-k`.

### Strategy layers and union retrieval

- Each **strategy** (built-in splitter or LLM-based) produces its own **layer of chunks**:
  - Built-in: Sentence/Token splitters (no LLM) → chunks tagged with `metadata["strategy"] = "s_default"` / `"s_token"`, etc.
  - LLM: custom natural-language strategies → chunks tagged with their `strategy_id`.
- The index is built over the **union of all layers** (all chunks from all strategies).
- At query time, retrieval runs over this union, and the retrieval log shows, for each top‑k hit:
  - The **source document** and the **strategy id** (`strategy=<id>`) that produced that chunk.

## Quick start (v0.1 — single strategy, from code)

```python
from llama_index.core import Document, VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from rechunk import LLMNodeParser

# Optional: set global LLM (otherwise uses LlamaIndex default)
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

parser = LLMNodeParser(
    strategy_id="s_sections",
    strategy_instruction="Split the document by its structural divisions: chapters, headings, and subheadings. Each chunk is one section.",
)

docs = [Document(text="Your long document here...")]
nodes = parser.get_nodes_from_documents(docs)

index = VectorStoreIndex(nodes)
# Query as usual
```

## Roadmap

| Phase   | Milestone |
|---------|-----------|
| **v0.1** | `LLMNodeParser` — LLM-based chunking with a single strategy ✅ |
| v0.2 | Multi-strategy index with layer tagging and union retrieval |
| v0.3 | Feedback signal ingestion and Branch A/B diagnosis |
| v0.4 | Automatic strategy proposal via LLM |
| v0.5 | Strategy hit tracking and pruning |
| v1.0 | Full feedback loop, LlamaIndex plugin packaging, docs |

## License

MIT

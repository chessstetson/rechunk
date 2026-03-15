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

## Quick start (v0.1 — single strategy)

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

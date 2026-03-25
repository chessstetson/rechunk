# ReChunk

**Adaptive, feedback-driven RAG chunking** — an extension for [LlamaIndex](https://www.llamaindex.ai/) that treats chunking as a living, strategy-driven process instead of a one-time split. Reliable re-indexing asynchronously via Temporal.

## About

Search over private data often qualitatively underperforms public. You know that feeling — you search for an email you know is there, and it eludes you. Each private corpus has idiosyncrasies never seen in public data, and little or no ground truth. A chunking strategy that works beautifully on one corpus fails silently on another.

The usual technique for embedding-based-search is to pick a chunking strategy at setup time and leave it. ReChunk takes a different view: the search index over a private corpus should be adaptive and respond to user feedback.

When retrieval produces a bad answer, ReChunk lets you change how your documents are chunked without a full reindex. New strategies run in parallel with existing ones. The system adapts to failure rather than silently compounding it. ReChunk can use tried-and-tested procedural chunking from LlamaIndex, which it extends — but it can also derive custom chunks using the LLM itself, tuned to the specific structure of ideas in your corpus. Durable execution via Temporal ensures re-indexing is reliable and resumable at scale.

Built on [LlamaIndex](https://www.llamaindex.ai/) and [Temporal](https://temporal.io/).

### At a glance

- **Index = f(corpus, S)** — the index is a pure function of your documents and a set of chunking strategies.
- **Strategy layers** — each strategy is a natural-language instruction; an LLM does the chunking. Chunks are tagged by strategy for multi-layer retrieval (v0.2+).
- **Feedback loop (roadmap)** — poor answers trigger diagnosis and, when the answer exists in the corpus, proposal of new strategies.

Optional **local** docs (not tracked in git): `rechunk_strategy.md` (design / roadmap), `REPOSITORY_DESCRIPTION.md` (GitHub *About* blurb), `TEMPORAL_IMPLEMENTATION_STEPS.md` (implementation checklist).

**Tracked design note — derived chunks (planned):** **[DERIVED_CHUNKS.md](DERIVED_CHUNKS.md)** describes the upcoming `derived` strategy kind (synthetic embeddable text + `source_spans`), dedup keys, persistence, and a **prominent “Future revisions”** section for how we might evolve the design.

## Install

```bash
pip install -e .
```

Requires Python 3.10+. Optional: set `OPENAI_API_KEY` (or configure another LLM via LlamaIndex `Settings.llm`).

## Run with your own docs

From the project root (with `OPENAI_API_KEY` set and the package installed, e.g. in a venv):

```bash
# Interactive helper: prompts for a path if you omit it; ingests into ECS, queues embeddings (Temporal), then Q&A
python scripts/run_interactive.py
python scripts/run_interactive.py path/to/your/docs

# Chunk a directory of .txt files (or a single .txt file)
python scripts/run_with_docs.py path/to/your/docs

# Chunk and run a query (with retrieval + LLM feedback: chunks, scores, then answer)
python scripts/run_with_docs.py path/to/your/docs --query "What is the main idea?"

# Interactive: chunk once, then ask questions in a loop (feel how fast embedding retrieval is)
python scripts/run_with_docs.py path/to/your/docs --interactive
```

Use `docs` for the included sample. With `--query` or `--interactive`, the script shows **retrieval** (embedding cosine similarity, which chunks were picked, timing) and then the **LLM response** (synthesis from those chunks, timing). Options: `--strategy-id`, `--strategy`, `--model`, `--query`, `--interactive`, `--top-k`.

### Benchmark corpora (Wikipedia, CUAD, PG-19)

To pull **small subsets** from Hugging Face into a plain `.txt` tree (same shape as a normal `docs/` upload):

```bash
pip install -e ".[benchmark-corpora]"   # pins datasets<4 (required for script-based hubs like pg19)
python scripts/prepare_hf_benchmark_corpus.py wikipedia --n 200
python scripts/prepare_hf_benchmark_corpus.py cuad --n 40
python scripts/prepare_hf_benchmark_corpus.py pg19 --n 15 --split validation  # streams until n books; add --full-split to download whole split
```

Defaults write under `docs/benchmark_corpora/<preset>/`. See **`scripts/BENCHMARK_CORPORA.md`** for flags and ingest commands.

### Temporal (ingest vs vectorization)

Chunking/embeddings run in workers on **two task queues** (see `src/rechunk/temporal_queues.py`):

1. **`rechunk-ingest`** — `FilesystemCorpusIngestWorkflow`: corpus snapshot → ECS + hash manifest (no OpenAI embed required on this worker).
2. **`rechunk-strategy-chunking`** — `BatchDocumentVectorizationWorkflow` (one workflow, many activities per hash): read ECS, chunk, write VectorStore rows.

Run **`python temporal_workers.py`** to poll **both** queues in one process (local dev), or **`ingest`** / **`vectorization`** for split processes. Then:

```bash
python scripts/start_corpus_ingest.py path/to/docs --wait
python scripts/start_strategy_chunking.py s_default
```

### Strategy layers and union retrieval

- Each **strategy** (built-in splitter or LLM-based) produces its own **layer of chunks**:
  - Built-in: Sentence/Token splitters (no LLM) → chunks tagged with `metadata["strategy"] = "s_default"` / `"s_token"`, etc.
  - LLM: custom natural-language strategies → chunks tagged with their `strategy_id`.
- The index is built over the **union of all layers** (all chunks from all strategies).
- At query time, retrieval runs over this union, and the retrieval log shows, for each top‑k hit:
  - The **source document** and the **strategy id** (`strategy=<id>`) that produced that chunk.

### Quick demo
![getting_started_rechunk](https://github.com/user-attachments/assets/1af15911-7926-4d39-b90e-c8a72003b54e)


### System diagrams

```mermaid
flowchart TD
    CLI["CLI / scripts"]:::client

    subgraph svc ["IndexService layer"]
        IS["IndexService\nindex_service.py"]
        CH["Chunker\ndiff + plan work items"]
        CSI["corpus_snapshot_id\nderived from active hashes"]
    end

    subgraph ecs ["ExtractedContentService"]
        ECS["FilesystemExtractedContentService\nsource of truth for document content"]
        ECSS["storage/ecs/\ncontent/ · state/active_logical.json"]
    end

    subgraph vs ["VectorStore"]
        VS["FilesystemVectorStore\nrows + corpus collections"]
        VSS["storage/vector_store_dev/\nrows/ · collections/"]
    end

    subgraph fp ["Key derivation"]
        FP["fingerprints.py\nstrategy_fp · embedding_fp"]
    end

    CLI --> IS
    IS --> CH
    IS --> CSI
    CH -->|list_active_hashes| ECS
    CH -->|list_vectorized_hashes| VS
    IS -->|get_content| ECS
    IS -->|get_collection / put_collection| VS
    CSI -.->|used by| VS
    FP -.->|used by| CH
    FP -.->|used by| VS
    ECS --- ECSS
    VS --- VSS

    classDef client fill:#EEEDFE,stroke:#7F77DD,color:#3C3489
    classDef store fill:#E1F5EE,stroke:#1D9E75,color:#085041
    classDef derived fill:#F1EFE8,stroke:#888780,color:#444441
```

```mermaid
flowchart TD
    SRC["Source files\n(local filesystem)"]

    subgraph ingest ["① Ingest — rechunk-ingest queue"]
        SNAP["build_and_write_ingest_snapshot\n(path claim-check)"]
        WFI["FilesystemCorpusIngestWorkflow"]
        ACT_I["ingest_filesystem_corpus_from_snapshot\n(activity)"]
        ECS["ExtractedContentService\nensure_content · apply_source_inventory"]
        MAN["corpus_content_hashes.json\n(active manifest)"]
    end

    subgraph vec ["② Vectorize — rechunk-strategy-chunking queue"]
        DIFF["Chunker.list_pending\ndiff ECS hashes vs VectorStore rows"]
        WFV["BatchDocumentVectorizationWorkflow"]
        ACT_V["vectorize_content_for_strategy\n(activity)"]
        CHUNK["chunk (LLM or builtin)\n+ embed (OpenAI)"]
        VS_ROWS["VectorStore.upsert_rows\nspan_start · span_end · embedding"]
    end

    subgraph query ["③ Query"]
        CSI["compute_corpus_snapshot_id\n(hash of active hashes)"]
        COL["VectorStore.get_collection\n(cached LlamaIndex index)"]
        RAG["retrieve_top_k\n+ synthesize"]
    end

    SRC --> SNAP --> WFI --> ACT_I --> ECS --> MAN
    ECS -->|list_active_hashes| DIFF
    DIFF -->|pending work items| WFV --> ACT_V
    ACT_V -->|get_content| ECS
    ACT_V --> CHUNK --> VS_ROWS

    ECS -->|list_active_hashes| CSI
    CSI --> COL
    VS_ROWS -->|rows assembled into collection| COL
    COL --> RAG
```

```mermaid
flowchart TD
    TC["temporal_client.py\ntrigger_filesystem_ingest_sync\ntrigger_pending_vectorization_sync"]

    subgraph qi ["Task queue: rechunk-ingest"]
        WI["FilesystemCorpusIngestWorkflow"]
        AI["ingest_filesystem_corpus\n_from_snapshot"]
    end

    subgraph qv ["Task queue: rechunk-strategy-chunking"]
        WBV["BatchDocumentVectorizationWorkflow\n★ new v12 path"]
        WDV["DocumentVectorizationWorkflow\n(single-hash variant)"]
        WSC["StrategyChunkingWorkflow\n(legacy — file-path based)"]

        AV["vectorize_content_for_strategy\n★ reads ECS · writes VectorStore rows"]
        ALG["log_workflow_summary"]
        ALC["chunk_doc_with_strategy\nchunk_doc_with_builtin_splitter\n(legacy — writes JSONL cache)"]
    end

    subgraph rt ["worker_runtime.py"]
        WR["configure_worker_runtime\nECS + VectorStore injected at startup"]
    end

    TC --> WI
    TC --> WBV
    WI --> AI
    WBV --> AV
    WBV --> ALG
    WDV --> AV
    WDV --> ALG
    WSC --> ALC
    WSC --> ALG
    AV --> WR
    WR -->|get_worker_ecs| AV
    WR -->|get_worker_vector_store| AV

    classDef new fill:#E1F5EE,stroke:#1D9E75,color:#085041
    classDef legacy fill:#FAEEDA,stroke:#BA7517,color:#633806
    class WBV,WDV,AV new
    class WSC,ALC legacy
```

## Roadmap

### Product

| Milestone | Target |
|-----------|--------|
| **Prototype** | Done |
| **Benchmarking** | Mar 2026 |
| **Strategy balancing** | Apr 2026 |
| **Gradient descent in embedding space** | Apr 2026 |


## License

MIT

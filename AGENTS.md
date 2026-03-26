# Agent / maintainer guide

Short orientation for humans and coding agents working on **ReChunk**.

## What this repo is

- **RAG chunking** on top of **LlamaIndex**, with **Temporal** for durable ingest + vectorization.
- **Corpus text** lives in **ECS** (`FilesystemExtractedContentService` → `storage/ecs/`).
- **Embeddings + chunk rows** live in a **VectorStore** (dev default: `FilesystemVectorStore` → `storage/vector_store_dev/`).
- **Strategies** (how to chunk) live in **`rechunk_strategies.json`** at the repo root (see `rechunk_strategies.json.example`).

## Layout

| Area | Role |
|------|------|
| `src/rechunk/` | Library: strategies, ECS, vector store, index service, parsers, RAG helpers |
| `scripts/` | CLI entry points (`run_with_docs.py`, `run_interactive.py`, ingest, HF export, **doctor**) |
| `temporal_*.py` (repo root) | Temporal workflows, activities, workers (import `rechunk` + `temporal_vectorization_inputs`) |
| `temporal_vectorization_inputs.py` | Workflow/activity payloads (stdlib-only; safe for workflow sandbox imports) |

## Task queues (do not rename casually)

Defined in `src/rechunk/temporal_queues.py`:

- **`rechunk-ingest`** — filesystem → ECS + manifest (`FilesystemCorpusIngestWorkflow`).
- **`rechunk-strategy-chunking`** — chunk + embed + `VectorStore.upsert_rows` (`BatchDocumentVectorizationWorkflow`, etc.).

Workers: `python temporal_workers.py` (both queues) or `ingest` / `vectorization` subcommands.

## Strategy kinds (`rechunk.strategies`)

- **`builtin_splitter`** — LlamaIndex sentence/token windows; `splitter`: `"sentence"` \| `"token"`.
- **`llm`** — LLM emits verbatim (or multi-span) chunks; provenance in **`metadata["source_spans"]`** (and related fields).
- **`derived`** — LLM emits **synthetic** `content` + required **`source_spans`**; see `DERIVED_CHUNKS.md`.

Vector rows and JSONL caches use **`metadata["source_spans"]`** as the canonical provenance shape (sorted span keys for merge identity). See `src/rechunk/derived_metadata.py` and `src/rechunk/vector_store/filesystem.py`.

## Integration surface (embedding ReChunk in an app)

Prefer importing from **`rechunk`** (installed package) rather than calling scripts:

- `IndexService`, `Chunker`, `FilesystemExtractedContentService`, `FilesystemVectorStore`
- `Strategy`, `load_strategies`, `save_strategies`, `strategy_to_dict`
- `compute_strategy_fingerprint`, `compute_embedding_fingerprint`, `VECTOR_SCHEMA_VERSION`

Temporal orchestration from Python: `rechunk.temporal_client` (and `temporalio` directly for advanced use).

**Less stable / script-oriented:** argument parsers under `scripts/`, workflow IDs, stderr log wording.

## Environment variables (common)

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Embeddings (worker) + LLM chunking + CLI synthesis |
| `TEMPORAL_ADDRESS` | Default `localhost:7233` |
| `RECHUNK_OPENAI_EMBEDDING_MODEL` | Must match worker and CLI index build |
| `RECHUNK_VECTOR_STORE_DEV_ROOT` | Override vector store root |
| `RECHUNK_STRATEGY_CACHE_DIR` | Override JSONL strategy cache (tests / legacy) |
| `RECHUNK_MAX_CONCURRENT_ACTIVITIES` | Vectorization worker parallelism (default `8`) |
| `RECHUNK_BATCH_VECTORIZATION_FANOUT` | Workflow wave size (default `32`) |
| `RECHUNK_BATCH_WORKFLOW_TASK_TIMEOUT_SECONDS` | Workflow task timeout for batch vectorization |

## Verification

From repo root with venv active:

```bash
pip install -e ".[dev]"
python scripts/rechunk_doctor.py
pytest tests/ -q
```

After touching chunking, vector rows, or workflows, run at least **`tests/test_phase_b_filesystem_stores.py`**, **`tests/test_phase_c_vectorization.py`**, and parser/metadata tests if relevant.

## Docs for humans

- **README.md** — quick start + architecture overview.
- **`scripts/BENCHMARK_CORPORA.md`** — Hugging Face export presets.
- **`DERIVED_CHUNKS.md`** — derived strategy design + evolution notes.

## Pitfalls

- **Worker must be running** for ECS + VectorStore path; otherwise index stays empty or stale.
- **Embedding fingerprint** must match between the process that wrote rows and the process that builds the index (`OPENAI_EMBEDDING_MODEL` / `RECHUNK_OPENAI_EMBEDDING_MODEL`).
- **`datasets` 4.x** breaks some HF benchmark presets; use `pip install -e ".[benchmark-corpora]"` (pins `<4`).
- Legacy **`--legacy-jsonl`** index path skips VectorStore; default path is ECS + rows.

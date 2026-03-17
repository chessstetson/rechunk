## Temporal Integration Design for ReChunk (Branch: `temporal`)

This document describes how we will use Temporal to run **LLM-based chunking strategies in parallel** with the real‑time RAG index.

The goal: **RAG stays responsive using existing chunks and cache**, while Temporal runs background workflows to compute and update new chunking strategies.

---

### 1. Roles: RAG App vs Temporal

- **RAG app (existing code)**
  - Loads docs and computes non‑LLM chunks (SentenceSplitter / TokenTextSplitter) synchronously.
  - Reads LLM‑based chunks from the on‑disk cache under `storage/strategies/` (per strategy, per `content_hash`).
  - Builds the `VectorStoreIndex` from whatever chunks currently exist.
  - Does **not** wait for Temporal — it just uses the current cache snapshot.

- **Temporal (new on `temporal` branch)**
  - Owns **long‑running background workflows** that:
    - Iterate over documents.
    - For each strategy + document, decide whether LLM chunking is needed.
    - Delegate LLM chunking work to **Activities**.
    - Persist results into the same cache (`storage/strategies/...`).
  - One or more **worker processes** host these workflows and activities.

---

### 2. Core Temporal Concepts We’ll Use

- **Workflow** – deterministic orchestration:
  - Captures “which documents were processed for this strategy”, “what hashes are in cache”, etc.
  - Survives crashes and restarts (Temporal replays the event history).

- **Activities** – non‑deterministic work:
  - Make LLM calls, read/write cache files, etc.
  - Have their own timeouts and retry policies.

- **Task Queues / Workers**
  - A `temporal_worker.py` process will:
    - Connect to Temporal.
    - Register workflows and activities.
    - Poll a task queue (e.g. `rechunk-strategy-chunking`) and execute tasks.

---

### 3. Workflows and Activities

#### 3.1 Workflow: `StrategyChunkingWorkflow`

- **Purpose**
  - Orchestrate chunking for a **single strategy** across a corpus of documents.

- **Inputs**
  - `strategy_id`: string (e.g. `"s_procedures"`).
  - `strategy_instruction`: string (LLM prompt instruction).
  - `model`: optional LLM model name (default `gpt-4o-mini`).
  - Optionally: a corpus identifier or explicit doc list (for now we can reuse the local loader).

- **Responsibilities**
  - Load the list of documents and their `content_hash` (reusing our existing loader and hashing).
  - For each document:
    - If `(strategy_id, content_hash)` **already has cached nodes** in `storage/strategies/{strategy_id}_chunks.jsonl`, skip.
    - Else:
      - Schedule **`chunk_doc_with_strategy` activity** for that document.
  - Optionally, support:
    - **Signals** (pause/stop) and **Queries** (progress inspection) later.

#### 3.2 Activity: `chunk_doc_with_strategy`

- **Purpose**
  - Do the actual LLM chunking for a **single document + strategy**, and persist the result.

- **Inputs**
  - `strategy_id`, `strategy_instruction`, `model`.
  - `doc_id`, `text`, `content_hash`.

- **Steps**
  1. Run an LLM chunking routine equivalent to `LLMNodeParser` for that document.
  2. Produce `TextNode`‑like structures (chunks) with:
     - `id_`, `text`, `metadata` (including `strategy`, `source_doc`), `ref_doc_id`, and optional `start_char_idx` / `end_char_idx`.
  3. Append those nodes as JSONL records to:
     - `storage/strategies/{strategy_id}_chunks.jsonl` keyed by `content_hash`.
  4. Let Temporal’s retry policy handle:
     - Timeouts, transient OpenAI errors, etc.

- **Error handling**
  - For persistent failures, we can:
    - Record a “failed for this (strategy_id, content_hash)” entry.
    - Optionally have a separate workflow / activity to revisit failures.

---

### 4. Interaction with Existing Cache and Index

We already have:

- Per‑doc `content_hash` in `Document.metadata`.
- A per‑strategy cache format under `storage/strategies/{strategy_id}_chunks.jsonl`:
  - Each record: `{ "content_hash": "...", "nodes": [...] }`.
- Logic to:
  - Load cached nodes for `(strategy_id, content_hash)`.
  - Run LLM chunking only when there is no cache entry.

#### 4.1 Changes for Workflow Integration

1. **Move the “LLM per‑doc loop” into the Temporal workflow:**
   - Instead of the CLI iterating over docs and calling `LLMNodeParser` itself, the workflow will:
     - Iterate docs and schedule `chunk_doc_with_strategy` activities.

2. **Make the existing CLI “cache‑only” aware:**
   - When building the index, the CLI should:
     - Use built‑in splitters as before.
     - For LLM strategies:
       - Only read **cached** chunks from `storage/strategies/...`; do not re‑LLM chunk synchronously.
   - This ensures RAG uses **whatever the workers have already produced**, and does not block on Temporal.

3. **Incremental updates**
   - As Temporal finishes more docs for a strategy:
     - The cache file grows.
     - The next RAG index rebuild (or refresh) sees more chunks for that strategy.

---

### 5. Concrete Files to Add (Python, Temporal SDK)

Assuming the Python Temporal SDK (`temporalio`) from the official docs:

- `temporal_workflows.py`
  - Define `StrategyChunkingWorkflow` with `@workflow.defn` and `@workflow.run`.
  - Use Activities via Temporal’s `workflow.execute_activity(...)`.

- `temporal_activities.py`
  - Define `chunk_doc_with_strategy` with `@activity.defn`.
  - Reuse / wrap existing LLM chunking logic:
    - Use our `LLMNodeParser` under the hood or equivalent logic.
    - Call cache helpers: `_append_llm_chunk_cache(...)`.

- `temporal_worker.py`
  - Connect to Temporal:
    - e.g. `client = await Client.connect("localhost:7233", ...)`
  - Start a `Worker` that:
    - Listens on e.g. `task_queue="rechunk-strategy-chunking"`.
    - Registers `StrategyChunkingWorkflow` and `chunk_doc_with_strategy`.

- `start_strategy_chunking.py` (or CLI integration)
  - Connects as a Temporal client.
  - Starts `StrategyChunkingWorkflow` for a given strategy:
    - e.g. `await client.execute_workflow(StrategyChunkingWorkflow.run, args, id="rechunk-strategy-s_procedures", task_queue="rechunk-strategy-chunking")`.

---

### 6. How “Parallel to RAG” Works in Practice

1. You use the existing scripts (or a new CLI) to **add a new LLM strategy** (`s_procedures`).
2. The app:
   - Saves the strategy to `rechunk_strategies.json`.
   - **Starts a Temporal workflow** for that strategy in the background.
3. While the workflow runs:
   - Workers chunk docs with LLM and write into `storage/strategies/s_procedures_chunks.jsonl`.
   - Your RAG app continues to:
     - Build an index from:
       - Existing non‑LLM chunks.
       - Any cached LLM chunks already written.
     - Answer queries in real time.
4. As time goes on:
   - Background chunking completes more docs.
   - Rebuilding or refreshing the index naturally picks up more LLM‑based chunks, improving recall and answer quality **without ever blocking queries on LLM chunking**.

---

### 7. Future Enhancements (Out of Scope for v1)

- **Embedding caching**
  - Persist per‑node embeddings so we don’t re‑embed nodes every run.
  - Use a persistent vector store (e.g. SQLite, Chroma, or a Temporal‑integrated store).

- **Signals / Queries**
  - Add Signals to pause/resume chunking workflows or to mark strategies as deprecated.
  - Add Queries to inspect progress per strategy from the CLI.

- **Multi‑strategy workflows**
  - One workflow orchestrating multiple strategies (e.g. a “StrategySetWorkflow” that runs child workflows per strategy).

- **OpenAI Agents SDK integration**
  - Wrap chunking Activities as tools via `activity_as_tool`, allowing an LLM “orchestrator agent” to decide when to trigger chunking or re‑chunking.

For now, the v1 objective is: **use Temporal to make LLM chunking durable, retryable, and parallel to RAG**, while keeping the data model and cache formats aligned with the existing ReChunk code.


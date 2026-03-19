# ReChunk × Temporal — Step-by-Step Implementation Strategy

This document breaks down the **ReChunk × Temporal development plan** (e.g. `rechunk-temporal-plan_revised.md`) into ordered, testable steps on branch **temporal_02**. Current baseline: **main** (built-in + LLM strategies, both use `storage/strategies/{strategy_id}_chunks.jsonl`; LLM path still does synchronous chunking on cache miss).

---

## Step 0 — Verify activity logic with tests (ASAP)

**Goal:** Confirm as soon as possible that the code that will live inside Temporal activities works and doesn’t break when run there. Add tests first so we can refactor with confidence.

### Step 0.1 — Tests for activity logic (right after 1.1)

- [ ] **0.1.1** Add `tests/test_activity_logic.py` (or `tests/test_temporal_activity.py`) that:
  - **Cache round-trip:** For a test strategy id and a small list of `TextNode`s, call `append_chunk_cache(strategy_id, content_hash, nodes)` then `load_chunk_cache(strategy_id)` and assert the loaded nodes match (same text, metadata, ref_doc_id). Use a temporary directory for `STRATEGY_STORAGE_DIR` so tests don’t touch real cache (override via env or module-level test fixture).
  - **Activity logic in isolation (no Temporal):** Implement a small helper that does exactly what `chunk_doc_with_strategy` will do: (1) read doc text from a path, (2) run `LLMNodeParser(…).get_nodes_from_documents([Document(...)])` with a **mocked LLM** so no API call, (3) call `append_chunk_cache(strategy_id, content_hash, nodes)`, (4) load from cache and assert. Use a temp file with fixed content and a mock that returns a known JSON chunk list. This proves the pipeline (read → parse → cache → load) works before it’s inside an Activity.
- [ ] **0.1.2** Ensure `rechunk.cache` is written so it can be overridden in tests (e.g. `STRATEGY_STORAGE_DIR` from env or a parameter) for hermetic tests.

**Check:** `pytest tests/test_activity_logic.py` (or the chosen name) passes; no Temporal server or real LLM required.

### Step 0.2 — Test that the Activity runs inside Temporal (right after 1.2)

- [ ] **0.2.1** Add a test that runs the real `chunk_doc_with_strategy` activity in Temporal’s test environment:
  - Use `temporalio.testing.WorkflowEnvironment` (or run a real worker in-process) to start a workflow that executes `chunk_doc_with_strategy` once with a temp doc path and mocked or minimal LLM.
  - Assert the cache file (or `load_chunk_cache`) contains the expected entry after the workflow completes.
- [ ] **0.2.2** This verifies serialization of activity input/output, async execution, and that the activity code path doesn’t break when executed by the Temporal runtime.

**Check:** Test passes; we have confidence that “required functionality inside Temporal activities” works and the code won’t break.

---

## Phase 1 — Wrap LLM chunking in an Activity

**Goal:** One document’s LLM chunking is durable and retryable via Temporal. No change yet to how the RAG app builds the index.

### Step 1.1 — Add Temporal dependency and shared cache module

- [ ] **1.1.1** Add `temporalio` to `pyproject.toml` dependencies.
- [ ] **1.1.2** Create `src/rechunk/cache.py` that:
  - Defines `STRATEGY_STORAGE_DIR` (same path as `run_with_docs`: `storage/strategies/` under project root).
  - Exposes `compute_content_hash(text: str) -> str` (SHA-256, same as CLI).
  - Exposes `append_chunk_cache(strategy_id, content_hash, nodes)` writing one JSONL record (same format as `_append_strategy_chunk_cache` in the CLI).
  - Exposes `load_chunk_cache(strategy_id) -> dict[content_hash, list[nodes]]` (same format as `_load_strategy_chunk_cache`).
  - Uses the same `_node_to_dict` / `_dict_to_node` and JSONL layout so worker and CLI share one cache.
- [ ] **1.1.3** (Optional) Refactor `run_with_docs.py` to call into `rechunk.cache` for append/load so there’s a single implementation. If time-constrained, duplicate the logic in the Activity first and refactor later.

**Check:** `src/rechunk/cache.py` can be imported and `append_chunk_cache` + `load_chunk_cache` round-trip nodes for a test strategy id.

---

### Step 1.2 — Implement `chunk_doc_with_strategy` Activity

- [ ] **1.2.1** Create `temporal_activities.py` in the **project root** (so worker and CLI can import without sys.path hacks).
- [ ] **1.2.2** Define activity input (e.g. dataclass) with: `strategy_id`, `strategy_instruction`, `model` (optional), `doc_id` (path relative to a docs root), `content_hash`, and `docs_root` (path to document directory).
- [ ] **1.2.3** Implement `@activity.defn async def chunk_doc_with_strategy(input)`:
  - Resolve `doc_path = docs_root / doc_id`, read text from disk (claim-check; no doc text in workflow input).
  - Build `Document(text=..., id_=doc_id)` and run existing `LLMNodeParser(strategy_id=..., strategy_instruction=..., ...).get_nodes_from_documents([doc])`.
  - Call `rechunk.cache.append_chunk_cache(strategy_id, input.content_hash, nodes)`.
- [ ] **1.2.4** Use `Settings.llm` or inject LLM from env so the Activity uses the same OpenAI model as the CLI (e.g. gpt-4o-mini). No workflow/activity code in workflow file (sandbox-safe).

**Check:** Run the Activity in isolation (or via a one-line workflow that runs one Activity) for one document; confirm `storage/strategies/{strategy_id}_chunks.jsonl` has one new line and the CLI’s index build sees that chunk.

---

### Step 1.3 — Minimal sequential `StrategyChunkingWorkflow`

- [ ] **1.3.1** Create `temporal_workflows.py` in the **project root** (sandbox-safe: only `temporalio`, `asyncio`, `dataclasses`, `typing`).
- [ ] **1.3.2** Define workflow input dataclass: `strategy_id`, `strategy_instruction`, `model`, `docs_root`, `doc_ids: list[str]` (and optionally a list of `content_hash` per doc if we have it; else Activity can compute it).
- [ ] **1.3.3** Implement `StrategyChunkingWorkflow.run(input)`:
  - For each `doc_id` in `input.doc_ids`, call `workflow.execute_activity("chunk_doc_with_strategy", ChunkDocInput(...), start_to_close_timeout=timedelta(minutes=5))`.
  - Execute activities **sequentially** (for loop with await) for Phase 1.
- [ ] **1.3.4** Workflow ID: `rechunk-{strategy_id}` (per plan). Use `WorkflowIDConflictPolicy.USE_EXISTING` or `ALLOW_DUPLICATE` as appropriate when starting.

**Check:** Start the workflow with a small corpus (e.g. 2–3 docs); worker processes all; cache file grows; CLI index build shows those chunks.

---

### Step 1.4 — Worker process

- [ ] **1.4.1** Create `temporal_worker.py` in the project root.
- [ ] **1.4.2** Connect to Temporal (e.g. `localhost:7233`), create a `Worker` with task queue e.g. `rechunk-strategy-chunking`, register `StrategyChunkingWorkflow` and `chunk_doc_with_strategy`.
- [ ] **1.4.3** Document in README or script docstring: run Temporal server (e.g. `temporal server start-dev`), set `OPENAI_API_KEY`, then `python temporal_worker.py`.

**Check:** Worker starts without errors; starting a workflow from the Python client results in activities running and cache written.

---

### Step 1.5 — Verify retries

- [x] **1.5.1** Simulate an OpenAI rate-limit or timeout (e.g. mock or temporary bad key) for one document; confirm only that Activity retries and eventually succeeds or fails, and that other documents still complete. **Done:** `test_activity_retry_one_doc_fails_then_succeeds` in `tests/test_activity_logic.py` runs the full workflow with 2 docs, a wrapper activity that fails on first attempt for the second doc, and asserts both docs end up in cache after Temporal retries. Requires Temporal test server (skips if unavailable).
- [ ] **1.5.2** Optionally add a small script or instruction to trigger one workflow via `temporalio.client.Client.start_workflow` for manual testing.

**Phase 1 done when:** Running the workflow produces the same `_chunks.jsonl` output as the existing synchronous path, and a transient failure retries only the affected activity.

---

## Phase 2 — Parallel fan-out and skip-if-cached

**Goal:** Chunk all documents in parallel; skip docs that are already in the cache.

### Step 2.1 — Skip-if-cached in the workflow

- [x] **2.1.1** Add an Activity `load_doc_manifest(docs_root, doc_ids)` (or equivalent) that returns a list of `{doc_id, content_hash}` by reading each doc from disk and computing hash. Workflow must not do I/O; Activity does. **Done:** `load_doc_manifest` in `temporal_activities.py`; `get_cached_hashes_for_strategy` in `rechunk.cache`.
- [x] **2.1.2** Add a way for the workflow to know which `(strategy_id, content_hash)` pairs are already cached. **Done:** Activity `get_cached_hashes(strategy_id)` reads the strategy JSONL and returns list of `content_hash`.
- [x] **2.1.3** In `StrategyChunkingWorkflow.run`, after loading manifest and cached set, build the list of tasks only for docs where `content_hash not in cached_hashes`. **Done.**

**Check:** Re-running the same workflow for the same strategy and corpus schedules no activities (all cached); changing one doc schedules only one activity.

---

### Step 2.2 — Parallel execution

- [x] **2.2.1** Replace the sequential `for doc in ...: await execute_activity(...)` with a list of activity tasks and `await asyncio.gather(*tasks)`. **Done.**
- [x] **2.2.2** Keep `start_to_close_timeout=timedelta(minutes=5)` per activity. Ensure worker has enough concurrency (default is often 100). **Done.**

**Check:** A 20-document corpus completes noticeably faster than sequential; one doc failure retries without blocking others.

---

## Phase 3 — RAG app cache-read-only; worker creates all strategy chunks

**Goal:** The CLI only reads from cache (no chunking). The Temporal worker is the only writer for all strategies (LLM and built-in).

### Step 3.1 — CLI reads cache only (all strategies)

- [x] **3.1.1** In `build_index_for_strategies` in `run_with_docs.py`, for `s.kind == "llm"`:
  - Remove the block that creates `LLMNodeParser` and calls `parser.get_nodes_from_documents([doc])` on cache miss.
  - Keep: load `cache = _load_strategy_chunk_cache(s.id)`; for each doc, if `content_hash in cache` then add those nodes; **else skip that doc for this strategy** (no synchronous LLM call).
  - Optionally log or print “no cached chunks for doc X, strategy Y; skip” so the user knows to run the worker or wait.
- [x] **3.1.2** Built-in strategies no longer compute-on-miss in CLI; worker writes all chunks. **Done.**

**Check:** With no cache, index build finishes quickly; after running worker + start_strategy_chunking, index build picks up chunks.

---

### Step 3.2 — Workflow supports both kinds; start script and trigger

- [x] **3.2.1** `StrategyChunkingInput` has `kind` and `splitter`; workflow calls `chunk_doc_with_strategy` or `chunk_doc_with_builtin_splitter`. **Done.**
- [x] **3.2.2** `start_strategy_chunking.py` accepts `--kind llm|builtin` and `--splitter sentence|token`. **Done.**
- [x] **3.2.3** Adding a strategy in the interactive menu triggers the workflow (`_trigger_strategy_chunking_sync`). **Done.**

**Phase 3 done when:** CLI only reads cache; all chunking is done by the worker. Rebuild index after worker backfills to see new chunks.

---

## File summary

| File | Phase | Action |
|------|--------|--------|
| `pyproject.toml` | 1.1 | Add `temporalio` |
| `src/rechunk/cache.py` | 1.1 | New: shared cache read/write and content_hash |
| `tests/test_activity_logic.py` (or `test_temporal_activity.py`) | 0.1, 0.2 | New: cache round-trip; activity logic in isolation; activity run in Temporal test env |
| `temporal_activities.py` | 1.2, 2.1 | New: `chunk_doc_with_strategy`; later `load_doc_manifest`, `get_cached_hashes` |
| `temporal_workflows.py` | 1.3, 2.1, 2.2 | New: `StrategyChunkingWorkflow` (sequential → parallel, skip-if-cached) |
| `temporal_worker.py` | 1.4 | New: register workflow + activities, run worker |
| `scripts/run_with_docs.py` | 3.1 | Change: all strategies read-only from cache; no sync chunking; trigger workflow on add |
| `scripts/start_strategy_chunking.py` | 3.2 | New: CLI to start `StrategyChunkingWorkflow` |

---

## Order of work (recommended)

1. **1.1** (cache module + temporalio) → **0.1** (tests: cache round-trip + activity logic in isolation) → **1.2** (implement activity) → **0.2** (test activity runs inside Temporal) → **1.3** → **1.4** → **1.5**.
2. **2.1** (skip-if-cached) → **2.2** (parallel) (Phase 2).
3. **3.1** (RAG cache-read-only) → **3.2** (trigger script) (Phase 3).

Tests come **as soon as** the cache and activity logic exist so we verify that required functionality works inside Temporal activities and the code won't break before relying on the full worker/workflow stack. Each phase ends with a verifiable “done when” so you can merge or iterate.

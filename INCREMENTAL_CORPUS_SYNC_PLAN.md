# Plan: incremental corpus sync and automatic vectorization

This document proposes how ReChunk could **detect new or changed corpus data**, **ingest it into ECS**, and **incrementally refresh search indexes** (vector layers per strategy) in a way that scales from **local folders** to **remote object stores and future backends**.  
**No implementation yet** — design choices and a phased roadmap only.

---

## 1. Goals

- **Detect** additions, updates, and (optionally) deletions relative to what the system already considers “active” for a given corpus source.
- **Ingest** new/changed blobs into **ECS** (content-addressed by hash), using **durable** execution (**Temporal**) where appropriate.
- **Vectorize** only what is **new or stale** for each configured strategy (existing **Chunker diff** + **`BatchDocumentVectorizationWorkflow`** pattern).
- Support **more than local disk**: same logical pipeline should extend to **S3, GCS, Azure Blob, HTTP crawl surfaces, vendor DMS exports**, etc., without rewriting chunk/embed semantics.
- Keep operations **observable** (what changed, what was skipped, per-strategy backlog) and **safe to retry** (idempotent by `content_hash` + fingerprints).

## 2. Non-goals (for this plan)

- Training or fine-tuning **embedding models** or **LLMs** (“the model” here means **retrieval index / strategy layers**, not PyTorch checkpoints).
- Replacing Temporal or ECS with a specific cloud vendor stack in v1.
- Perfect **real-time** guarantees (sub-second) across all backends — **near-real-time** or **scheduled** may be the practical default for remote sources.

## 3. Current baseline (terminology)

- **ECS** = extracted content store (local `FilesystemExtractedContentService` today); authoritative **text + hash** per logical document revision.
- **Ingest** = `FilesystemCorpusIngestWorkflow` + snapshot claim-check; updates ECS and corpus hash manifest.
- **Vectorization** = `BatchDocumentVectorizationWorkflow` + `vectorize_content_for_strategy`; writes **VectorStore** rows; skips already-vectorized `(hash, strategy_fp, embed_fp, schema)`.
- **Index** = union of chunks/embeddings for all strategies; **corpus snapshot id** ties a query-time collection to the set of active content hashes.

Incremental work today is largely **hash-level**: new files → new hashes → ingest → diff → vectorize pending hashes.

---

## 4. What “new data” means

| Event | ECS | VectorStore |
|--------|-----|-------------|
| New file / new object | New `content_hash` after ingest | Pending for each strategy until vectorized |
| Edited file (content changed) | **New** `content_hash`; old hash may become unreferenced | New rows for new hash; old rows may remain until **GC / tombstone** policy |
| Deleted file | Remove from **active manifest** / logical inventory | Optional: delete rows for orphaned hashes or retain for audit |

The plan should treat **content hash** as the primary identity (already aligned with ReChunk). **Logical doc id** (path, object key, external id) maps to **current** hash via ingest inventory.

---

## 5. Detection mechanisms (pick per deployment)

Different sources favor different triggers. The design should allow **pluggable “corpus watchers”** that all produce the same downstream shape: **`CorpusDelta` → ingest snapshot (or incremental ingest API) → vectorization enqueue**.

### 5.1 Local filesystem

- **Polling**: periodic scan (mtime/size or full hash); simple, portable; latency = poll interval.
- **OS notifications** (`FSEvents` / `inotify`): lower latency on one machine; **not** sufficient for remote sources; usually **one root** per watcher process.
- **Explicit path**: user or automation **points at a folder** (or list of paths); run diff against last-known inventory — good for CI and “upload complete” events.

### 5.2 Object storage (S3 / GCS / Azure)

- **Periodic list + ETag/MD5**: scalable; tune prefix, pagination, delimiter.
- **Event notifications** (S3 EventBridge, GCS Pub/Sub, etc.): near-real-time; requires infra; handler enqueues **small** Temporal workflows or signals.
- **Manifest sidecar**: upstream publishes `inventory.jsonl` (path, etag, size); ReChunk consumes it — great for **large** off-machine corpora.

### 5.3 “Always checking” vs on-demand

- **Always on**: long-running **poller** or **event subscriber** service (separate from interactive CLI); forwards work to Temporal.
- **On-demand**: CLI/API **“sync now”** with optional `--since` / cursor; same code path, easier to operate early.

**Recommendation:** implement **on-demand sync + poll interval** first; add **push/events** when a deployment has the infra.

---

## 6. Proposed architecture

### 6.1 Corpus source abstraction

Introduce a narrow interface (conceptual):

- **`list_candidates(since_cursor?) →` iterator of** `{ logical_id, locator, etag?, size?, last_modified? }`
- **`fetch_content(locator) →` bytes or stream** (for ingest activity claim-check)
- **`stable_cursor()`** for resumable polls (opaque per source)

**LocalDirSource**, **S3PrefixSource**, **ManifestFileSource** become implementations.  
ECS and vectorization **do not** depend on which source produced the blob.

### 6.2 Delta → ingest

Two patterns:

1. **Snapshot (current style)**  
   Build a manifest of logical ids + **precomputed hashes** (or hash in ingest activity).  
   Already works for filesystem; extend snapshot builder to **pull from object metadata** where full hash isn’t precomputed.

2. **Incremental ingest API (future)**  
   `ApplySourceDelta(added[], removed[])` workflow with activities that **stream** new objects into ECS — better for huge remote buckets when full snapshots are expensive.

Start with **(1)** unless snapshot build time becomes the bottleneck.

### 6.3 Automatic vectorization

After ingest completes (or continuously in the background):

- For **each active strategy** in `rechunk_strategies.json` (or an allowlist):  
  **`Chunker.list_pending`** (ECS hashes − VectorStore rows for that fingerprint) → start **`BatchDocumentVectorizationWorkflow`** when pending non-empty.

Options:

- **Coupled**: ingest workflow’s last step **starts** vectorization child workflows (or signals a “fan-out” workflow).  
- **Decoupled (recommended)**: ingest emits “corpus version bumped”; a small **`ReconcileVectorizationWorkflow`** or external scheduler runs **every N minutes** and enqueues batch jobs per strategy.  
  Decoupling avoids one giant workflow history and matches **multi-strategy** and **partial failures**.

### 6.4 “All models”

Interpret as **all strategy layers**:

- Either **one reconcile loop** enqueues **one batch per strategy** when any pending hashes exist, or  
- **Priority queue** (e.g. baseline splitter first, expensive LLM strategies later).

**Embedding model changes** remain a **full re-embed** problem (`embedding_fingerprint` mismatch); out of scope for incremental *file* sync except to **surface** “re-index required” in UI/CLI.

---

## 7. Scalability and remote resources

- **Claim-check pattern**: workflows carry **paths / URIs / snapshot ids**, not file contents, in history.
- **Bounded concurrency**: existing **`RECHUNK_BATCH_VECTORIZATION_FANOUT`** and **`RECHUNK_MAX_CONCURRENT_ACTIVITIES`**; ingest may need similar caps for **parallel downloads**.
- **Auth**: source plugins hold **credentials** via env / workload identity; never in workflow args.
- **Cost**: optional **budget** hooks (max new objects per run, max GB per day).

---

## 8. Operational concerns

- **Dedup**: content-addressed ingest already dedupes identical bytes across paths.
- **Deletes**: define policy — **logical delete** (remove from active set) vs **physical delete** from ECS/VectorStore.
- **Ordering**: eventual consistency for object stores; ingest idempotent retries handle duplicates.
- **Observability**: metrics — `last_successful_sync`, `pending_hashes_by_strategy`, `ingest_lag_seconds`.

---

## 9. Phased roadmap

| Phase | Scope | Outcome |
|-------|--------|---------|
| **A** | **Local poll + explicit path** CLI (`rechunk sync /path` or `watch --interval`) building snapshots → existing ingest + **auto-start** vectorization for all strategies | Validated E2E without new cloud code |
| **B** | **S3/GCS prefix** source + cursor; same ingest/vector path | Off-machine corpus; batch-friendly |
| **C** | **Event-driven** ingest (webhook / queue consumer) + optional **incremental ingest** without full manifest | Near-real-time, lower list costs |
| **D** | **Row GC** for removed hashes; optional **schedule** CRON / Temporal Schedule API | Long-running stable indexes |

---

## 10. Open decisions

- Should **every strategy** auto-run on every sync, or only a configured subset (cost control)?
- **Single global reconcile** vs **one Temporal schedule per corpus source**?
- **Multi-tenant** isolation (per-tenant ECS prefix + queue names) if ReChunk is hosted as a service?
- **Public API** (`POST /sources/.../sync`) vs CLI-only v1?

---

## 11. Related files (today)

- `temporal_workflows.py` — ingest + batch vectorization  
- `temporal_activities.py` — `ingest_filesystem_corpus_from_snapshot`, `vectorize_content_for_strategy`  
- `src/rechunk/temporal_client.py` — enqueue patterns  
- `src/rechunk/index_service.py` / `chunker.py` — pending diff logic  
- `src/rechunk/ingest_snapshot.py` — snapshot format  

---

*Document version: draft for discussion. Update when an implementation milestone is chosen.*

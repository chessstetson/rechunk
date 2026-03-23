# Derived chunks — planned strategy

**Status:** Design / roadmap. The `derived` strategy kind is **not** fully implemented in application code yet; this document keeps the approach coherent while implementation evolves (e.g. on branch `derivedchunks`).

---

## What “derived” means

ReChunk today supports:

| Kind | Role |
|------|------|
| **`builtin_splitter`** | Mechanical sentence/token windows |
| **`llm`** | Verbatim (or multi-span) excerpts from the source; `content` is built from document offsets |

**`derived`** (planned) adds a third path: the LLM **writes synthetic text** optimized for embedding and retrieval (summaries, inventories, obligation profiles, timelines, etc.), while **`source_spans`** (and optional short `quote` anchors) tie that text back to **real regions of the source document**. The embedder still sees **plain string `content`** / `chunk_text`; nothing requires a second LLM pass to re-materialize that string — it is persisted like other chunks (JSONL + vector rows).

---

## Identity, deduplication, and collisions

For a **given derived strategy** and **given document** (`content_hash`):

- **Canonical span key:** take all `(start_char, end_char)` pairs from `source_spans`, **normalize** (e.g. sort by `start`, then `end`).  
  **Same multiset ⇒ same chunk identity:** `[(10,20),(5,15)]` and `[(5,15),(10,20)]` are **one** key.
- **Collisions** (two LLM-emitted nodes with the same canonical key): **keep one row** (last or first — pick a policy), and **log a warning** so operators know the model duplicated a slot.
- **Grounding rule:** derived text must **always** be justified by **valid** `source_spans` within `doc_len` — no “free floating” nodes without provenance.

---

## Clause presence / absence pattern (intentional exception shape)

For strategies that classify **many legal clause types** (venue, withdrawal, no-shop, …):

- **Present:** one row per hit, with spans pointing at the actual clause text (and `content` that is easy to embed — possibly still verbatim-heavy or lightly polished).
- **Several absent in one go:** it can be **acceptable** to emit **one** derived row whose `content` lists multiple absences (“no withdrawal, no no-shop, …”) and whose citations use a **whole-document span** `(0, doc_len)` — meaning *we grounded this conclusion on a full-document read*. That yields **one** canonical key for that aggregate row (not N colliding empty-span rows). Retrieval is coarser for “which single absence?” queries; that is an accepted tradeoff unless you later split absences per type (see **Future revisions**).

---

## Persistence and audit

The **exact string sent to the embedding model** is stored as:

- `text` on each node in `storage/strategies/{strategy_id}_chunks.jsonl` (via the existing chunk cache), and  
- `chunk_text` on each vector-store row,

with **`source_spans` (and related flags) in `metadata`**. Humans can audit without re-running the derivation LLM.

---

## Implementation touchpoints (checklist)

When implementing, expect to touch (at least):

- `Strategy` / `normalize_strategy_kind` — add `"derived"`
- New **`DerivedNodeParser`** (prompt: `content` + `source_spans` + optional `quote`)
- **`vectorize_content_for_strategy`** — branch for derived; **do not** locate chunks with `find(content)` in the source; build spans / merge keys from `source_spans`
- **Vector row merge keys** — include canonical sorted spans (and handle derived vs verbatim row shapes so unrelated nodes never overwrite each other)

---

# Future revisions — how we might change this

> **This section is deliberately prominent.** The plan above is meant to stay stable for v1 of `derived`, but product and retrieval needs may force adjustments. If you change behavior, update **this doc** and any merge-key / normalization code together.

Possible revisions:

1. **Finer absence retrieval**  
   If one aggregate “all missing clauses” row is too blunt, split into **per–clause-type** rows, each with either a **narrow** span (e.g. “definitions + misc” only) or a **structured key** (e.g. `clause_type` + canonical spans) so merge keys stay unique without lying about provenance.

2. **Merge key beyond sorted spans**  
   If envelope-only or whole-doc keys collide across **different** intended nodes (same strategy, same doc), extend the row key with a **stable node id** from the LLM or a hash of `content` — only if we accept that the “span set alone” is no longer the sole identity.

3. **`split_long_nodes_for_embedding`**  
   Today oversized nodes are split mechanically; for derived nodes that may **break** “one node = one answer.” Options: skip splitting for `derived`, compress in one LLM pass, or split with explicit metadata that provenance is partial.

4. **Long-document derivation**  
   Windowed derivation (per chunk of doc) + merge strategy is underspecified; we may add a **second-pass** “merge windows” node or hierarchical IDs.

5. **Stricter verification**  
   Optional validation that `quote` is a substring of `source[start:end]`; stricter ordering rules for overlapping `source_spans`.

6. **Schema / fingerprint versioning**  
   If canonical span normalization changes, bump a **vector schema** or strategy fingerprint component so old and new rows don’t silently mix.

7. **UX**  
   How provenance is shown (multi-span highlights, whole-doc badge for aggregate absence rows) may feed back into prompt shape and required fields.

---

## Related reading

- Live LLM verbatim chunking prompt: `src/rechunk/node_parser.py` (`CHUNKING_PROMPT`, multi-span `spans`)
- Chunk cache format: `src/rechunk/cache.py`
- Vector rows: `temporal_activities.py` (`chunk_text`, `metadata`, `span_start` / `span_end`)

# ReChunk: Strategy & Implementation Document

> **Status:** Early design / pre-implementation  
> **Baseline framework:** LlamaIndex  
> **Core concept:** Adaptive, feedback-driven RAG chunking

---

## 1. Problem Statement

Existing RAG frameworks (LangChain, LlamaIndex, Haystack, etc.) apply chunking strategies
statically at ingest time. Chunk boundaries are decided once and never revisited, regardless
of how well those chunks serve real user queries. When retrieval fails, the root cause is
often the chunking strategy itself — but no current open source system detects or corrects
this automatically.

ReChunk solves this by treating chunking as a living, feedback-driven process.

---

## 2. Core Design Principle

The index is a pure function of two inputs:

```
index = f(corpus, S)
```

Where:
- **corpus** — the set of raw source documents
- **S** — the active set of chunking strategies `{s1, s2, ..., sn}`
- **index** — the resulting vector store of embedded chunks

The index is a **derived artifact**. It has no independent state. Given the same corpus
and strategy set, you always get the same index. This means:

- The index can always be fully rebuilt from scratch
- There is no such thing as a "partial update" to the index — only additions and removals of strategy layers
- History of how you arrived at a strategy set is irrelevant; only the current set matters

---

## 3. Strategy Layers

Each chunking strategy `si` is an independent, atomic unit:

```
index = f(corpus, s1) ∪ f(corpus, s2) ∪ ... ∪ f(corpus, sn)
```

The final index is the **union** of all strategy layers. Key properties:

- Strategies are **additive** — adding a new strategy never invalidates or changes existing chunks
- Strategies are **independent** — each strategy layer can be computed and stored separately
- Strategies are **deletable** — a strategy layer can be dropped when it is no longer useful,
  removing all its chunks from the index
- Strategy layers are tagged on every chunk they produce, so attribution is always clear

### What is a Strategy?

A strategy is an LLM prompt instruction that defines a chunking intent. Examples:

| Strategy ID | Instruction |
|-------------|-------------|
| `s_entities` | "Identify all named entities (people, places, organizations) and create one chunk per entity, including all text that directly describes or relates to that entity." |
| `s_sections` | "Split the document by its structural divisions: chapters, headings, and subheadings. Each chunk is one section." |
| `s_procedures` | "Identify all procedures — sequences of actions taken with specific objects or tools — and create one chunk per procedure." |
| `s_qa` | "Identify all questions explicitly or implicitly posed in the document, and create one chunk per question-answer pair." |

Strategies are **parameterized natural language**. The LLM is the chunker.

---

## 4. LLM-Based Chunking

Instead of algorithmic splitters (token windows, sentence boundaries), ReChunk uses an LLM
to perform chunking according to a strategy instruction.

### Prompt Template

```
You are a document chunking engine. Given the document below, apply the following
chunking strategy and return a JSON array of chunks.

Strategy: {strategy_instruction}

Rules:
- Each chunk must be self-contained enough to answer a question without requiring adjacent context
- Include enough surrounding context in each chunk to make it meaningful in isolation
- Return ONLY valid JSON. No preamble, no explanation.

Response format:
[
  {
    "chunk_id": "unique string",
    "content": "the chunk text",
    "metadata": { "strategy": "{strategy_id}", "source_doc": "{doc_id}" }
  }
]

Document:
{document_text}
```

### Handling Large Documents

For documents exceeding the LLM context window:

1. **Pre-segment** using a lightweight structural pass (headings, page breaks) to get sections
2. **Apply LLM chunking per section** with overlap context from adjacent sections
3. **Reconcile boundaries** — merge or deduplicate chunks that span section boundaries

---

## 5. The Feedback Loop

### 5.1 Feedback Signal

A feedback signal is generated when a user indicates a query was poorly answered. This can be:

- **Explicit** — thumbs down, "this didn't answer my question", rating < threshold
- **Implicit** — user immediately rephrases and resubmits, session abandonment after retrieval

Each signal carries:
- The original query
- The retrieved chunks that were returned
- The generated answer (if any)
- A signal type: `poor_answer` or `no_answer`

### 5.2 Two-Branch Diagnosis

When a feedback signal arrives, ReChunk runs a diagnostic to determine the cause:

```
                    Feedback signal received
                            │
                            ▼
                  Run exhaustive search
                  across full corpus
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Answer found                Answer not found
        in corpus                    anywhere
              │                           │
              ▼                           ▼
    BRANCH A: Wrong strategy       BRANCH B: Unanswerable
    (information exists,           (information genuinely
    chunking failed to             not in corpus)
    surface it)
              │                           │
              ▼                           ▼
    Queue new strategy             Log as unanswerable.
    for next rebuild               Surface to corpus
                                   maintainer.
```

**Branch A** is the interesting path. The exhaustive search confirms the answer exists
somewhere in the raw corpus — it was just never accessible because no strategy produced
a chunk that captured it cleanly.

**Branch B** is important for honesty. Without this branch, the system would keep trying
to rechunk its way to an answer that doesn't exist.

### 5.3 Greedy / One-Shot Strategy Updates

ReChunk is greedy: the first feedback signal of a new class triggers an immediate strategy
proposal. There is no waiting to accumulate evidence. Reasoning:

- A single missed query class is sufficient evidence that a strategy gap exists
- Waiting introduces a window where the gap continues to hurt users
- False positives (adding an unnecessary strategy) are low-cost — unused strategies are pruned

### 5.4 Strategy Proposal

When Branch A fires, an LLM is used to propose a new strategy instruction:

```
A user asked the following question and did not get a good answer:

Query: {query}

An exhaustive search confirmed this information exists in the corpus but was not
surfaced by the current chunking strategies:

Current strategies: {strategy_list}

Propose a new chunking strategy instruction that would produce chunks capable of
answering this class of question. Be specific about what semantic unit to chunk around.

Return JSON: { "strategy_id": "...", "instruction": "..." }
```

The proposed strategy is reviewed (automatically or by a human operator depending on
configuration) and added to the active strategy set `S`.

---

## 6. Index Lifecycle

```
┌─────────────────────────────────────────────────────┐
│                    CORPUS (raw docs)                │
└────────────────────────┬────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   Strategy Set S    │
              │  {s1, s2, ..., sn}  │
              └──────────┬──────────┘
                         │  f(corpus, S)
              ┌──────────▼──────────┐
              │        INDEX        │
              │  Layer 1 (s1 chunks)│
              │  Layer 2 (s2 chunks)│
              │  Layer n (sn chunks)│
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │      RETRIEVAL      │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   FEEDBACK SIGNAL   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  DIAGNOSIS + UPDATE │
              │  Add sn+1 to S  OR  │
              │  Log as unanswerable│
              └─────────────────────┘
```

### Adding a New Document

New documents inherit the current strategy set. Run `f(doc_new, S)` and union into the index.
No rebuild of existing chunks required.

### Adding a New Strategy

Run `f(corpus, s_new)` and union the resulting chunks into the index.
No existing chunks are touched.

### Removing a Strategy

Drop all chunks tagged with that strategy's ID from the index.
Corpus and remaining strategy layers are unaffected.

### Full Rebuild

Re-run `f(corpus, S)` for all strategies. Used when the corpus changes substantially
or when the LLM used for chunking is upgraded.

---

## 7. Strategy Pruning

To prevent unbounded index growth, strategies are monitored for utility:

- Each chunk tracks a **hit count** — how many times it was retrieved and included in a
  final answer
- Strategies whose chunks accumulate zero or near-zero hits over a rolling window are
  flagged as candidates for pruning
- Pruning removes all chunks for that strategy and retires it from `S`
- Pruning decisions can be automatic (below threshold) or require human approval

This closes the loop: strategies are added greedily when gaps are detected, and removed
conservatively when they prove unnecessary.

---

## 8. LlamaIndex Integration

ReChunk is implemented as a **LlamaIndex extension**, not a fork. Integration points:

| LlamaIndex Primitive | ReChunk Usage |
|---------------------|---------------|
| `NodeParser` | Subclassed as `LLMNodeParser` — calls LLM with strategy prompt instead of algorithmic splitting |
| `VectorStoreIndex` | Used as the underlying index; each strategy layer is a tagged namespace within it |
| `StorageContext` | Manages persistence of corpus, strategy set, and index state |
| `RetrieverEvaluator` | Repurposed as the feedback signal collector |
| `BaseRetriever` | Subclassed for cross-layer retrieval |

ReChunk exposes a clean interface on top of these primitives so consumers never need to
interact with LlamaIndex directly if they don't want to.

---

## 9. Open Questions

The following design decisions are unresolved and need further discussion:

1. **Exhaustive search implementation** — what does "exhaustive search" look like in Branch B?
   Full BM25 over raw corpus? LLM-based scan? Hybrid? Cost vs. recall tradeoff.

2. **Cross-layer retrieval** — at query time, do we retrieve across all strategy layers
   simultaneously (and rerank), or do we route queries to the most relevant strategy layer first?

3. **Strategy versioning** — if a strategy instruction is edited (refined), do we treat it as
   a new strategy (recompute from scratch) or a mutation of the existing one?

4. **Human-in-the-loop** — should strategy additions always require human approval, or can
   they be applied automatically in low-risk scenarios?

5. **Chunk deduplication** — multiple strategies may produce overlapping chunks. Do we
   deduplicate at index time, at retrieval time, or not at all?

6. **Evaluation harness** — how do we measure whether a new strategy actually improved
   retrieval quality before committing it to production?

---

## 10. Roadmap (Proposed)

| Phase | Milestone |
|-------|-----------|
| v0.1 | `LLMNodeParser` — LLM-based chunking with a single strategy |
| v0.2 | Multi-strategy index with layer tagging and union retrieval |
| v0.3 | Feedback signal ingestion and Branch A/B diagnosis |
| v0.4 | Automatic strategy proposal via LLM |
| v0.5 | Strategy hit tracking and pruning |
| v1.0 | Full feedback loop end-to-end, LlamaIndex plugin packaging, docs |

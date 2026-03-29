"""
Temporal task queue names.

* **Ingest** — filesystem → ECS + active manifest (no chunking / embeddings).
* **Vectorization** — Phase C batch / per-hash vectorization (needs OpenAI embed + LLM for non-builtin strategies).
"""

TASK_QUEUE_INGEST = "rechunk-ingest"
TASK_QUEUE_VECTORIZATION = "rechunk-strategy-chunking"

"""
Temporal task queue names.

* **Ingest** — filesystem → ECS + active manifest (no chunking / embeddings).
* **Vectorization** — Phase C per-hash workflows + legacy batch chunking (needs OpenAI embed + LLM).
"""

TASK_QUEUE_INGEST = "rechunk-ingest"
TASK_QUEUE_VECTORIZATION = "rechunk-strategy-chunking"

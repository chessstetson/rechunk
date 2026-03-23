"""
ReChunk: Adaptive, feedback-driven RAG chunking.

LlamaIndex extension for strategy-based LLM chunking and (future) feedback-driven
strategy updates.

Design notes may live in ``rechunk_strategy.md`` locally (gitignored).
"""

from rechunk.node_parser import LLMNodeParser

__all__ = ["LLMNodeParser"]
__version__ = "0.1.0"

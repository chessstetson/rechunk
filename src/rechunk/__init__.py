"""
ReChunk: Adaptive, feedback-driven RAG chunking.

LlamaIndex extension for strategy-based LLM chunking and (future) feedback-driven
strategy updates.

See rechunk_strategy.md for design and roadmap.
"""

from rechunk.node_parser import LLMNodeParser

__all__ = ["LLMNodeParser"]
__version__ = "0.1.0"

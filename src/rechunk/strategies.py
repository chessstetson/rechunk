"""
Chunking strategy metadata: in-memory model and JSON persistence.

CLI and tests load/save the active strategy set via a caller-provided path
(e.g. project root ``rechunk_strategies.json``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Strategy:
    """In-memory representation of a chunking strategy."""

    id: str
    kind: str  # "builtin_splitter" (no LLM) or "llm"
    instruction: str
    # For kind="builtin_splitter": which LlamaIndex parser. "sentence" | "token"
    splitter: str = "sentence"
    # For kind="llm": OpenAI model name (default gpt-4o-mini when None)
    model: str | None = None


# Default when no strategy file exists: baseline (built-in) only, never LLM.
DEFAULT_BASELINE_STRATEGY = Strategy(
    id="s_default",
    kind="builtin_splitter",
    instruction="Sentence-based splitting (LlamaIndex default, chunk_size=1024)",
    splitter="sentence",
)


def strategy_to_dict(s: Strategy) -> dict:
    return {
        "id": s.id,
        "kind": s.kind,
        "instruction": s.instruction,
        "splitter": getattr(s, "splitter", "sentence"),
        "model": getattr(s, "model", None),
    }


def normalize_strategy_kind(raw: object) -> str:
    """
    Return ``\"llm\"`` or ``\"builtin_splitter\"``.

    Missing, empty, or unrecognized values default to ``builtin_splitter`` so chunking
    does not silently require an LLM.
    """
    if raw is None:
        return "builtin_splitter"
    s = str(raw).strip()
    if not s:
        return "builtin_splitter"
    sl = s.lower()
    if sl == "llm":
        return "llm"
    if sl in ("builtin_splitter", "builtin"):
        return "builtin_splitter"
    return "builtin_splitter"


def strategy_definition_uses_llm(sd: dict) -> bool:
    """True only when normalized kind is ``llm`` (explicit instruction-driven chunking)."""
    return normalize_strategy_kind(sd.get("kind")) == "llm"


def dict_to_strategy(d: dict) -> Strategy:
    return Strategy(
        id=d["id"],
        kind=normalize_strategy_kind(d.get("kind")),
        instruction=d["instruction"],
        splitter=d.get("splitter", "sentence"),
        model=d.get("model"),
    )


def load_strategies(path: Path) -> list[Strategy] | None:
    """Load strategy set from JSON file. Returns None if file missing or invalid."""
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list) or len(data) == 0:
            return None
        return [dict_to_strategy(item) for item in data]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def save_strategies(path: Path, strategies: list[Strategy]) -> None:
    """Serialize strategy set to JSON."""
    data = [strategy_to_dict(s) for s in strategies]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

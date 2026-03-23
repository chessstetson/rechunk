"""Ensure ``src`` and project root are on ``sys.path`` (``temporal_*`` modules live at repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_src = _root / "src"
for p in (_src, _root):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

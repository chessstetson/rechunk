"""
Stable paths to the **project (checkout) root** and ``storage/`` layout.

Modules directly under ``rechunk/`` can use ``Path(__file__).parents[2]`` for the repo root, but
anything nested deeper (e.g. ``rechunk/vector_store/``) must not — ``parents[2]`` would be ``src/``.
Use :func:`project_root` everywhere we default to ``<repo>/storage/...``.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """
    Repository root: directory that contains ``pyproject.toml`` and ``storage/``.

    This file lives at ``<root>/src/rechunk/repo_paths.py``.
    """
    return Path(__file__).resolve().parent.parent.parent

"""
Derived corpus identity for vector collection keying (v12 architecture).

``corpus_snapshot_id`` is **not** stored or published. It is computed from the
current active content hashes exposed by :class:`ExtractedContentService`.

Use :func:`compute_corpus_snapshot_id` as the **only** implementation — do not
reimplement locally.
"""

from __future__ import annotations

import hashlib


def compute_corpus_snapshot_id(active_hashes: list[str]) -> str:
    """
    SHA-256 hex of sorted, lowercased active ``content_hash`` values, one per line.

    Order of ``active_hashes`` does not matter. Empty corpus yields a fixed digest
    of the empty payload.
    """
    normalized = sorted(h.strip().lower() for h in active_hashes)
    payload = "\n".join(normalized)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

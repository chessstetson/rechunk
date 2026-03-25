"""
Filesystem-backed :class:`VectorStore` for local dev (v12 Phase B).

**Rows** (reusable across corpus membership)::

    rows/{strategy_fingerprint}/{embedding_fingerprint}/{vector_schema_version}/{content_hash}.json

Each file holds a JSON object ``{"rows": [ {...}, ... ]}`` where each row includes at least
``content_hash``, ``embedding``, ``chunk_text``, ``metadata`` (with ``source_spans`` provenance).

**Collections** (materialized LlamaIndex persist layout)::

    collections/{corpus_snapshot_id}/{strategy_fingerprint}/{embedding_fingerprint}/{vector_schema_version}/

**Dev / reference implementation** — not the production durability target.

``get_collection`` returns a loaded :class:`~llama_index.core.VectorStoreIndex` when
``embed_model`` was passed to the constructor and persisted data exists; otherwise
``None``. If ``embed_model`` is omitted, ``get_collection`` returns the persist
:class:`~pathlib.Path` when the directory looks like a valid persist layout, else ``None``
(so callers can load manually in tests).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rechunk.derived_metadata import canonical_source_spans_merge_key
from rechunk.repo_paths import project_root


def _vector_store_root() -> Path:
    env = os.environ.get("RECHUNK_VECTOR_STORE_DEV_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / "storage" / "vector_store_dev"


def _rows_dir(
    root: Path,
    *,
    strategy_fingerprint: str,
    embedding_fingerprint: str,
    vector_schema_version: str,
) -> Path:
    return root / "rows" / strategy_fingerprint / embedding_fingerprint / vector_schema_version


def _collection_dir(
    root: Path,
    *,
    corpus_snapshot_id: str,
    strategy_fingerprint: str,
    embedding_fingerprint: str,
    vector_schema_version: str,
) -> Path:
    return (
        root
        / "collections"
        / corpus_snapshot_id
        / strategy_fingerprint
        / embedding_fingerprint
        / vector_schema_version
    )


def _collection_ready(p: Path) -> bool:
    return p.is_dir() and (p / "index_store.json").is_file()


class FilesystemVectorStore:
    def __init__(self, root: Path | None = None, *, embed_model: Any = None) -> None:
        self._root = (root or _vector_store_root()).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._embed_model = embed_model

    def get_collection(
        self,
        *,
        corpus_snapshot_id: str,
        strategy_fingerprint: str,
        embedding_fingerprint: str,
        vector_schema_version: str,
    ) -> Any | None:
        p = _collection_dir(
            self._root,
            corpus_snapshot_id=corpus_snapshot_id,
            strategy_fingerprint=strategy_fingerprint,
            embedding_fingerprint=embedding_fingerprint,
            vector_schema_version=vector_schema_version,
        )
        if not _collection_ready(p):
            return None
        if self._embed_model is None:
            return p
        from llama_index.core import StorageContext, load_index_from_storage

        ctx = StorageContext.from_defaults(persist_dir=str(p))
        return load_index_from_storage(ctx, embed_model=self._embed_model)

    def put_collection(
        self,
        *,
        corpus_snapshot_id: str,
        strategy_fingerprint: str,
        embedding_fingerprint: str,
        vector_schema_version: str,
        index_obj: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        p = _collection_dir(
            self._root,
            corpus_snapshot_id=corpus_snapshot_id,
            strategy_fingerprint=strategy_fingerprint,
            embedding_fingerprint=embedding_fingerprint,
            vector_schema_version=vector_schema_version,
        )
        p.mkdir(parents=True, exist_ok=True)
        index_obj.storage_context.persist(persist_dir=str(p))
        if metadata:
            meta_path = p / "rechunk_collection_meta.json"
            meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    @property
    def root(self) -> Path:
        """Root directory (``storage/vector_store_dev`` or ``RECHUNK_VECTOR_STORE_DEV_ROOT``)."""
        return self._root

    def list_row_strategy_fingerprints(self) -> list[str]:
        """
        Strategy fingerprint directory names under ``rows/`` (for diagnostics).

        Row bundles live at ``rows/<strategy_fp>/<embedding_fp>/<schema>/<hash>.json``.
        """
        r = self._root / "rows"
        if not r.is_dir():
            return []
        return sorted(p.name for p in r.iterdir() if p.is_dir())

    def list_vectorized_hashes(
        self,
        *,
        strategy_fingerprint: str,
        embedding_fingerprint: str,
        vector_schema_version: str,
    ) -> list[str]:
        d = _rows_dir(
            self._root,
            strategy_fingerprint=strategy_fingerprint,
            embedding_fingerprint=embedding_fingerprint,
            vector_schema_version=vector_schema_version,
        )
        if not d.is_dir():
            return []
        out: list[str] = []
        for f in d.glob("*.json"):
            out.append(f.stem.lower())
        return sorted(out)

    def upsert_rows(
        self,
        *,
        strategy_fingerprint: str,
        embedding_fingerprint: str,
        vector_schema_version: str,
        rows: list[dict[str, Any]],
    ) -> None:
        by_hash: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            h = row.get("content_hash")
            if not h or not isinstance(h, str):
                raise ValueError("each row must include string content_hash")
            by_hash.setdefault(h.lower(), []).append(row)

        for content_hash, new_rows in by_hash.items():
            path = _rows_dir(
                self._root,
                strategy_fingerprint=strategy_fingerprint,
                embedding_fingerprint=embedding_fingerprint,
                vector_schema_version=vector_schema_version,
            )
            path.mkdir(parents=True, exist_ok=True)
            fp = path / f"{content_hash}.json"
            existing: list[dict[str, Any]] = []
            if fp.is_file():
                data = json.loads(fp.read_text(encoding="utf-8"))
                existing = list(data.get("rows", []))

            def row_merge_key(r: dict[str, Any]) -> tuple:
                """Sorted ``(start, end)`` tuple from ``metadata['source_spans']``."""
                meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
                ck = canonical_source_spans_merge_key(meta)
                if ck is None:
                    raise ValueError("each row metadata must include valid source_spans")
                return ck

            merged = {row_merge_key(r): r for r in existing}
            for r in new_rows:
                k = row_merge_key(r)
                if k in merged:
                    old_ct = merged[k].get("chunk_text", "")
                    new_ct = r.get("chunk_text", "")
                    if old_ct != new_ct:
                        import sys

                        print(
                            f"  [WARN] vector_store upsert: replacing row merge_key={k!r} "
                            f"(chunk_text changed, len {len(old_ct)} → {len(new_ct)})",
                            file=sys.stderr,
                            flush=True,
                        )
                merged[k] = r

            payload = json.dumps(
                {"rows": [merged[k] for k in sorted(merged.keys())]},
                indent=2,
            )
            fd, tmp = tempfile.mkstemp(dir=str(path), suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                Path(tmp).replace(fp)
            except Exception:
                Path(tmp).unlink(missing_ok=True)
                raise

    def read_rows_for_hash(
        self,
        *,
        content_hash: str,
        strategy_fingerprint: str,
        embedding_fingerprint: str,
        vector_schema_version: str,
    ) -> list[dict[str, Any]]:
        """Load stored rows for one content hash (helper for retrieval assembly)."""
        fp = (
            _rows_dir(
                self._root,
                strategy_fingerprint=strategy_fingerprint,
                embedding_fingerprint=embedding_fingerprint,
                vector_schema_version=vector_schema_version,
            )
            / f"{content_hash.lower()}.json"
        )
        if not fp.is_file():
            return []
        data = json.loads(fp.read_text(encoding="utf-8"))
        return list(data.get("rows", []))

    def row_bundle_stat(
        self,
        *,
        content_hash: str,
        strategy_fingerprint: str,
        embedding_fingerprint: str,
        vector_schema_version: str,
    ) -> tuple[float, int] | None:
        """``(mtime, size)`` for the JSON bundle of rows for this hash, or ``None`` if missing."""
        fp = (
            _rows_dir(
                self._root,
                strategy_fingerprint=strategy_fingerprint,
                embedding_fingerprint=embedding_fingerprint,
                vector_schema_version=vector_schema_version,
            )
            / f"{content_hash.lower()}.json"
        )
        if not fp.is_file():
            return None
        st = fp.stat()
        return (st.st_mtime, st.st_size)

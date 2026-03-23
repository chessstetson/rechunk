"""
Filesystem-backed :class:`ExtractedContentService` for local dev (v12 Phase B).

Layout under ``root`` (default ``storage/ecs`` or ``RECHUNK_ECS_ROOT``)::

    content/ab/abcd...64.json   # sharded extracted records
    state/active_logical.json   # logical_doc_id -> {content_hash, source_kind}

**Dev / reference implementation** — not the production durability target.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rechunk.cache import compute_content_hash
from rechunk.doc_loader import extract_file_content
from rechunk.extracted_content.models import ExtractedContent, SourceDocumentRef
from rechunk.repo_paths import project_root


def _ecs_root() -> Path:
    env = os.environ.get("RECHUNK_ECS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / "storage" / "ecs"


def _content_path(root: Path, content_hash: str) -> Path:
    h = content_hash.lower()
    return root / "content" / h[:2] / f"{h}.json"


def _active_path(root: Path) -> Path:
    return root / "state" / "active_logical.json"


class FilesystemExtractedContentService:
    """
    Stores :class:`ExtractedContent` as JSON; active corpus = values in ``active_logical``.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or _ecs_root()).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _load_active(self) -> dict[str, dict[str, str]]:
        p = _active_path(self._root)
        if not p.is_file():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): dict(v) for k, v in data.items()}

    def _save_active(self, m: dict[str, dict[str, str]]) -> None:
        p = _active_path(self._root)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(m, indent=2, sort_keys=True) + "\n"
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            Path(tmp).replace(p)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _write_content_record(self, ec: ExtractedContent) -> None:
        path = _content_path(self._root, ec.content_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "content_hash": ec.content_hash,
                "logical_doc_id": ec.logical_doc_id,
                "mime_type": ec.mime_type,
                "extracted_text": ec.extracted_text,
                "structured_content": ec.structured_content,
                "source_hint": ec.source_hint,
                "provenance": ec.provenance,
            },
            indent=2,
        )
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            Path(tmp).replace(path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _extract_text_and_mime(self, ref: SourceDocumentRef) -> tuple[str, str]:
        if ref.path is not None:
            p = Path(ref.path).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Source path not found: {p}")
            text = extract_file_content(p)
            if not text or not text.strip():
                raise ValueError(f"No extractable text for {p}")
            ext = p.suffix.lower()
            mime = ref.mime_type or (
                "application/pdf"
                if ext == ".pdf"
                else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                if ext == ".docx"
                else "text/plain"
            )
            return text.strip(), mime

        if ref.raw_bytes is not None:
            mime = ref.mime_type or "application/octet-stream"
            if mime.startswith("text/") or mime in ("application/json", "application/xml"):
                text = ref.raw_bytes.decode("utf-8", errors="replace")
                if not text.strip():
                    raise ValueError("Empty text from raw_bytes")
                return text.strip(), mime
            raise ValueError(
                "raw_bytes ingestion for non-text MIME requires a path for full extraction "
                "(PDF/DOCX); use SourceDocumentRef.path for binary formats."
            )

        raise ValueError("SourceDocumentRef must provide path or raw_bytes")

    def ensure_content(self, source_doc_ref: SourceDocumentRef) -> ExtractedContent:
        text, mime = self._extract_text_and_mime(source_doc_ref)
        content_hash = compute_content_hash(text)
        hint = source_doc_ref.source_hint
        if source_doc_ref.path is not None and hint is None:
            hint = str(Path(source_doc_ref.path).resolve())

        provenance: dict[str, Any] = {
            "source_kind": source_doc_ref.source_kind,
            **(source_doc_ref.metadata or {}),
        }

        ec = ExtractedContent(
            content_hash=content_hash,
            logical_doc_id=source_doc_ref.logical_doc_id,
            mime_type=mime,
            extracted_text=text,
            structured_content=None,
            source_hint=hint,
            provenance=provenance,
        )
        self._write_content_record(ec)

        active = self._load_active()
        active[source_doc_ref.logical_doc_id] = {
            "content_hash": content_hash,
            "source_kind": source_doc_ref.source_kind,
        }
        self._save_active(active)
        return ec

    def deactivate_logical_doc(self, logical_doc_id: str, *, reason: str | None = None) -> None:
        active = self._load_active()
        if logical_doc_id in active:
            del active[logical_doc_id]
            self._save_active(active)

    def apply_source_inventory(
        self,
        source_kind: str,
        observed_logical_doc_ids: list[str],
    ) -> None:
        observed = set(observed_logical_doc_ids)
        active = self._load_active()
        to_remove = [
            lid
            for lid, meta in list(active.items())
            if meta.get("source_kind") == source_kind and lid not in observed
        ]
        for lid in to_remove:
            del active[lid]
        if to_remove:
            self._save_active(active)

    def get_content(self, content_hash: str) -> ExtractedContent:
        path = _content_path(self._root, content_hash)
        if not path.is_file():
            raise KeyError(f"No extracted content for hash {content_hash[:16]}…")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ExtractedContent(
            content_hash=raw["content_hash"],
            logical_doc_id=raw["logical_doc_id"],
            mime_type=raw["mime_type"],
            extracted_text=raw.get("extracted_text"),
            structured_content=raw.get("structured_content"),
            source_hint=raw.get("source_hint"),
            provenance=raw.get("provenance"),
        )

    def has_content(self, content_hash: str) -> bool:
        return _content_path(self._root, content_hash).is_file()

    def list_active_hashes(self) -> list[str]:
        active = self._load_active()
        hashes = {m["content_hash"] for m in active.values() if "content_hash" in m}
        return sorted(hashes)

    def list_active_content_refs(self) -> list:
        """
        One :class:`~rechunk.corpus.ContentRef` per distinct active ``content_hash``.

        ``source_hint`` is a representative logical document id (stable order by hash).
        """
        from rechunk.corpus import ContentRef

        active = self._load_active()
        hint_by_hash: dict[str, tuple[str, str]] = {}
        for lid, meta in sorted(active.items()):
            h = meta.get("content_hash")
            if not h or not isinstance(h, str):
                continue
            key = h.lower()
            if key not in hint_by_hash:
                hint_by_hash[key] = (h, str(lid))
        return [
            ContentRef(content_hash=pair[0], source_hint=pair[1])
            for pair in (hint_by_hash[k] for k in sorted(hint_by_hash.keys()))
        ]

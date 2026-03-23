"""
Load local files into LlamaIndex :class:`Document` objects with content hashes.

Uses :mod:`rechunk.doc_loader` for extraction so CLI and worker stay aligned.
"""

from __future__ import annotations

import sys
from pathlib import Path

from llama_index.core import Document

from rechunk.cache import compute_content_hash
from rechunk.doc_loader import extract_file_content


def load_documents(path: Path) -> list[Document]:
    """
    Load supported files from a file or directory into LlamaIndex Documents.

    For a directory, scans recursively. Supports .txt, .md, .pdf, .docx.
    Documents with identical content (same hash) are loaded only once.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    docs: list[Document] = []
    seen_hashes: set[str] = set()

    if path.is_file():
        content = extract_file_content(path)
        if content:
            h = compute_content_hash(content)
            docs.append(
                Document(
                    text=content,
                    id_=path.name,
                    metadata={"content_hash": h},
                )
            )
        else:
            raise FileNotFoundError(f"Unable to read or empty file: {path}")
    else:
        candidates = sorted(path.rglob("*"))
        for f in (p for p in candidates if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}):
            content = extract_file_content(f)
            if content:
                h = compute_content_hash(content)
                if h in seen_hashes:
                    print(f"[SKIP] Duplicate content (same hash): {f.relative_to(path)}", file=sys.stderr)
                    continue
                seen_hashes.add(h)
                doc_id = str(f.relative_to(path))
                docs.append(
                    Document(
                        text=content,
                        id_=doc_id,
                        metadata={"content_hash": h},
                    )
                )
            else:
                print(f"[WARN] Skipping {f}: could not extract text", file=sys.stderr)
        if not docs:
            raise FileNotFoundError(f"No supported files under {path}")

    return docs

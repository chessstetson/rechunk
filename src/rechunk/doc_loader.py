"""
Extract text from supported file types (.txt, .md, .pdf, .docx).

Used by the CLI and by Temporal worker activities so both use the same logic.
PDF and DOCX require PyPDF2 and python-docx respectively.
"""

from pathlib import Path


def extract_file_content(path: Path) -> str | None:
    """
    Extract text content from a file. Returns None if the format is unsupported,
    libraries are missing, or reading fails.
    """
    path = Path(path)
    ext = path.suffix.lower()

    try:
        if ext == ".pdf":
            try:
                import PyPDF2  # type: ignore[import]
            except ImportError:
                return None
            with path.open("rb") as f:
                reader = PyPDF2.PdfReader(f)
                parts: list[str] = []
                for page in reader.pages:
                    part = page.extract_text() or ""
                    parts.append(part)
            content = "\n".join(parts)
            return content if content.strip() else None

        if ext == ".docx":
            try:
                import docx  # type: ignore[import]
            except ImportError:
                return None
            doc = docx.Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs]
            content = "\n".join(paragraphs)
            return content if content.strip() else None

        if ext in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text if text.strip() else None

        # Fallback: try as plain text
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text if text.strip() else None
        except Exception:
            return None
    except Exception:
        return None

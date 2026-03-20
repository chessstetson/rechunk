#!/usr/bin/env python3
"""
Write a hash-only JSON manifest from a filesystem corpus (for use with run_with_docs --manifest).

Does not embed paths or titles — only SHA-256 hex strings of extracted text, matching
the retrieval boundary described in CORPUS_REFACTOR_PLAN.md.

Usage:
  python scripts/write_corpus_manifest.py <corpus_path> <out.json>

Example:
  python scripts/write_corpus_manifest.py docs corpus_hashes.json
  python scripts/run_with_docs.py --manifest corpus_hashes.json --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from rechunk.hash_manifest import write_manifest_from_filesystem_scan


def main() -> None:
    p = argparse.ArgumentParser(description="Write hash-only corpus manifest JSON from a filesystem tree.")
    p.add_argument("corpus_path", type=Path, help="Directory or single file (same rules as run_with_docs path)")
    p.add_argument("out_json", type=Path, help="Output path for JSON array of SHA-256 hex strings")
    args = p.parse_args()
    write_manifest_from_filesystem_scan(args.corpus_path, args.out_json)
    n = len(json.loads(args.out_json.read_text(encoding="utf-8")))
    print(f"Wrote hash-only manifest: {args.out_json.resolve()} ({n} content hash(es))")


if __name__ == "__main__":
    main()

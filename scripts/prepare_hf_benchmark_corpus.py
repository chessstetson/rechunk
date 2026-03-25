#!/usr/bin/env python3
"""
Download a **small subset** of public Hugging Face datasets and write ``.txt`` files suitable
for :func:`scripts.start_corpus_ingest` / ``docs/``-style corpora (``.txt`` under a single root).

Install deps (once)::

    pip install -e '.[benchmark-corpora]'

``benchmark-corpora`` pins ``datasets<4`` (4.x removed **loading scripts**; ``deepmind/pg19`` still uses one).
It also includes ``pdfplumber`` for CUAD Pdf columns when needed.

Examples (defaults write under ``docs/benchmark_corpora/<preset>/``)::

    python scripts/prepare_hf_benchmark_corpus.py wikipedia --n 300
    python scripts/prepare_hf_benchmark_corpus.py cuad --n 40
    python scripts/prepare_hf_benchmark_corpus.py pg19 --n 15 --split validation

Override output root::

    python scripts/prepare_hf_benchmark_corpus.py wikipedia --out storage/benchmark_corpora/wikipedia --n 300

Then ingest (with Temporal worker running)::

    python scripts/start_corpus_ingest.py docs/benchmark_corpora/wikipedia --wait

**Note:** ``docs/`` and ``storage/`` are gitignored in this repo; pick any ``--out`` you prefer.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
import time
from pathlib import Path


def _require_datasets():
    try:
        from datasets import Dataset, IterableDataset, load_dataset  # noqa: F401
    except ImportError as e:
        print(
            "Missing package `datasets`. Install with:\n"
            "  pip install -e '.[benchmark-corpora]'\n"
            "or: pip install datasets",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    from datasets import load_dataset

    return load_dataset


def _slug(name: str, *, max_len: int = 100) -> str:
    s = re.sub(r"[^\w\s.-]", "", name, flags=re.UNICODE)
    s = re.sub(r"[-\s.]+", "_", s).strip("._")
    if not s:
        s = "doc"
    return s[:max_len]


def _unique_path(dir_path: Path, base: str, ext: str = ".txt") -> Path:
    p = dir_path / f"{base}{ext}"
    if not p.exists():
        return p
    for i in range(2, 10_000):
        cand = dir_path / f"{base}_{i}{ext}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find free filename for {base}")


def _write_doc(
    out_dir: Path,
    stem: str,
    body: str,
    *,
    header_lines: list[str] | None = None,
    max_chars: int | None,
) -> Path:
    text = body if isinstance(body, str) else str(body)
    if max_chars is not None and len(text) > max_chars:
        text = (
            text[:max_chars]
            + f"\n\n[--- truncated to {max_chars} chars for benchmark export ---]\n"
        )
    parts: list[str] = []
    if header_lines:
        parts.extend(header_lines)
        parts.append("")
    parts.append(text)
    content = "\n".join(parts)
    path = _unique_path(out_dir, _slug(stem))
    path.write_text(content, encoding="utf-8", errors="replace")
    return path


def cmd_wikipedia(args: argparse.Namespace, load_dataset) -> int:
    # Standard HF English Wikipedia snapshot (large); use streaming to avoid full download.
    config = args.wiki_config
    stream = load_dataset(
        "wikimedia/wikipedia",
        config,
        split="train",
        streaming=True,
    )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    # Reservoir-style sample over stream: collect pool then sample (bounded memory via early stop).
    pool: list[dict] = []
    seen = 0
    scan_cap = max(args.n * args.scan_multiplier, args.n)
    for row in stream:
        seen += 1
        title = (row.get("title") or "").strip() or f"article_{seen}"
        text = (row.get("text") or "").strip()
        if len(text) < args.min_chars:
            continue
        pool.append({"title": title, "text": text, "id": row.get("id", "")})
        if len(pool) >= scan_cap:
            break
    if not pool:
        print(
            "No usable Wikipedia articles (check network, config name, or --min-chars).",
            file=sys.stderr,
        )
        return 1
    if len(pool) < args.n:
        print(
            f"Only collected {len(pool)} usable rows after scanning {seen} (try --scan-multiplier).",
            file=sys.stderr,
        )
    rng.shuffle(pool)
    chosen = pool[: args.n]
    for row in chosen:
        header = [f"title: {row['title']}", f"source: wikimedia/wikipedia ({config})"]
        if row.get("id"):
            header.append(f"id: {row['id']}")
        _write_doc(
            out,
            row["title"],
            row["text"],
            header_lines=header,
            max_chars=args.max_chars_per_doc,
        )
    print(f"Wrote {len(chosen)} articles to {out} (from {len(pool)} candidates, scanned {seen} rows).")
    return 0


def _cuad_load_split(
    load_dataset,
    hf_dataset: str,
    split: str,
    *,
    streaming: bool,
    force_redownload: bool,
    verification_mode: object | None = None,
):
    """
    Load CUAD (or ``cuad-qa``). ``verification_mode=NO_CHECKS`` skips split-size verification;
    streaming avoids verification when the materialized cache is broken.
    """
    from datasets import DownloadMode

    kw: dict = {"streaming": streaming}
    if force_redownload:
        kw["download_mode"] = DownloadMode.FORCE_REDOWNLOAD
    if verification_mode is not None:
        kw["verification_mode"] = verification_mode
    return load_dataset(hf_dataset, split=split, **kw)


def _cuad_select_text_columns_only(ds):
    """
    Read only ``title`` and ``context`` from Arrow so Pdf (and other) columns are never decoded.
    """
    feats = getattr(ds, "features", None)
    if not feats or "title" not in feats or "context" not in feats:
        return ds
    try:
        return ds.select_columns(["title", "context"])
    except Exception:
        return ds


def _cuad_drop_pdf_columns(ds):
    """
    CUAD exposes some columns as Pdf dtype; streaming iteration tries to decode them and
    requires pdfplumber. Dropping those columns before iteration avoids that (we only need text).
    """
    features = getattr(ds, "features", None)
    if not features:
        return ds
    try:
        from datasets.features import Pdf
    except ImportError:
        return ds
    to_remove = [name for name, feat in features.items() if isinstance(feat, Pdf)]
    if not to_remove:
        return ds
    try:
        return ds.remove_columns(to_remove)
    except Exception:
        return ds


def _cuad_prepare_for_scan(ds, *, streaming: bool):
    """
    Narrow to text columns for **materialized** splits only.

    ``select_columns(['title','context'])`` on a **streaming** CUAD IterableDataset can yield rows
    with empty ``context`` (511-row partial streams then dedupe to 0 contracts). For streaming, only
    drop Pdf columns so full rows decode correctly.
    """
    if streaming:
        return _cuad_drop_pdf_columns(ds)
    return _cuad_drop_pdf_columns(_cuad_select_text_columns_only(ds))


def _cuad_row_to_title_context(row: object) -> tuple[str, str]:
    """Normalize HF ``Row`` / dict and read clause text + contract title."""
    if isinstance(row, dict):
        d = row
    else:
        try:
            d = dict(row)
        except Exception:
            try:
                d = {k: row[k] for k in row.keys()}  # type: ignore[attr-defined]
            except Exception:
                return "", ""

    ctx = d.get("context")
    if ctx is None or (isinstance(ctx, str) and not ctx.strip()):
        ctx = d.get("passage") or d.get("paragraph") or d.get("document") or ""
    title = d.get("title") or d.get("document_title") or d.get("doc_title") or ""

    if not isinstance(ctx, str):
        ctx = str(ctx) if ctx is not None else ""
    if not isinstance(title, str):
        title = str(title) if title is not None else ""
    return title.strip(), ctx.strip()


def _cuad_row_iterator(ds):
    """Prefer indexed access for materialized splits (more reliable than ``__iter__`` for some caches)."""
    if hasattr(ds, "__len__") and hasattr(ds, "__getitem__"):
        try:
            ln = len(ds)
        except Exception:
            ln = None
        if ln is not None:
            for i in range(ln):
                yield ds[i]
            return
    yield from ds


def _cuad_validate_nonempty(ds, label: str) -> bool:
    """Reject materialized splits with 0 rows (common with partial cache + no_checks)."""
    if not hasattr(ds, "__len__"):
        return True
    try:
        ln = len(ds)
    except Exception:
        return True
    if ln == 0:
        print(
            f"  [!] {label}: loaded split has 0 rows (typical with a partial Hugging Face cache).\n"
            f"      Trying another load method…\n",
            flush=True,
        )
        return False
    print(f"  (materialized split: {ln} rows)", flush=True)
    return True


def _ensure_pdfplumber_cuad() -> bool:
    """Print install hint if pdfplumber is missing (HF decodes Pdf-typed columns while iterating)."""
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        print(
            "CUAD iteration requires `pdfplumber` (Hugging Face decodes Pdf-typed columns).\n"
            "Install with:\n"
            "  pip install -e '.[benchmark-corpora]'\n"
            "or: pip install pdfplumber",
            file=sys.stderr,
        )
        return False
    return True


def _cuad_scan_to_by_title(
    ds,
    *,
    max_rows: int | None,
    progress_every: int,
) -> tuple[dict[str, str], int]:
    """
    One full pass over the split (~84k train QA rows) to keep longest ``context`` per ``title``.
    Files are not written until this returns.
    """
    print(
        "\nCUAD: scanning QA rows to dedupe by contract title.\n"
        "  Train has ~84,325 rows; expect a few minutes. Nothing is written to --out until the scan finishes.\n"
        "  (Streaming is slower but tolerates a partial cache; materialized load is faster when the cache is complete.)\n",
        flush=True,
    )
    by_title: dict[str, str] = {}
    n = 0
    t0 = time.monotonic()
    first_row_debug: dict | None = None
    try:
        for row in _cuad_row_iterator(ds):
            n += 1
            if first_row_debug is None:
                try:
                    first_row_debug = row.copy() if isinstance(row, dict) else dict(row)
                except Exception:
                    first_row_debug = None
            if progress_every > 0 and n % progress_every == 0:
                elapsed = time.monotonic() - t0
                print(
                    f"  … {n:>6} rows scanned, {len(by_title):>4} unique contracts, {elapsed:>5.0f}s elapsed",
                    flush=True,
                )
            if max_rows is not None and n >= max_rows:
                print(
                    f"  … stopping at --max-stream-rows={max_rows} (sample may be biased toward early contracts).",
                    flush=True,
                )
                break
            title, ctx = _cuad_row_to_title_context(row)
            if not ctx:
                continue
            if not title:
                title = "contract_" + hashlib.sha256(ctx.encode("utf-8", errors="replace")).hexdigest()[:12]
            prev = by_title.get(title)
            if prev is None or len(ctx) > len(prev):
                by_title[title] = ctx
    except ImportError as e:
        err = str(e).lower()
        if "pdfplumber" in err or "decoding pdf" in err:
            _ensure_pdfplumber_cuad()
            raise
        raise
    elapsed = time.monotonic() - t0
    print(
        f"  … scan complete: {n} rows, {len(by_title)} unique contracts in {elapsed:.1f}s\n",
        flush=True,
    )
    if not by_title and n > 0 and first_row_debug is not None:
        keys = list(first_row_debug.keys())
        print(f"  [debug] first row keys ({len(keys)}): {keys[:25]}{'…' if len(keys) > 25 else ''}", flush=True)
        for k in ("context", "title", "question", "passage", "paragraph"):
            if k in first_row_debug:
                v = first_row_debug[k]
                ln = len(v) if isinstance(v, str) else len(str(v))
                print(f"  [debug] {k!r}: type={type(v).__name__} len={ln}", flush=True)
    return by_title, n


def cmd_cuad(args: argparse.Namespace, load_dataset) -> int:
    # QA rows repeat the same contract context; one file per unique contract (keyed by title).
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    from datasets import VerificationMode

    # Streaming first: works with partial caches. ``verification_mode=no_checks`` on a broken cache
    # often yields an empty Dataset (0 rows) — do not prefer it.
    attempts: list[tuple[str, dict]] = []
    if args.force_redownload:
        attempts.append(
            (
                "non-streaming + force_redownload",
                {"streaming": False, "force_redownload": True, "verification_mode": None},
            )
        )
        attempts.append(
            (
                "streaming + force_redownload",
                {"streaming": True, "force_redownload": True, "verification_mode": None},
            )
        )
    else:
        attempts.append(
            ("streaming (works with partial HF cache)", {"streaming": True, "force_redownload": False, "verification_mode": None}),
        )
        attempts.append(
            (
                "non-streaming + default verification (fast if cache is complete)",
                {"streaming": False, "force_redownload": False, "verification_mode": None},
            ),
        )
        attempts.append(
            (
                "non-streaming + force_redownload",
                {"streaming": False, "force_redownload": True, "verification_mode": None},
            ),
        )
        attempts.append(
            (
                "non-streaming + verification_mode=no_checks (last resort; may be empty if cache is bad)",
                {
                    "streaming": False,
                    "force_redownload": False,
                    "verification_mode": VerificationMode.NO_CHECKS,
                },
            ),
        )
        attempts.append(
            (
                "streaming + force_redownload",
                {"streaming": True, "force_redownload": True, "verification_mode": None},
            ),
        )

    ds = None
    used = ""
    used_streaming = False
    last_err: BaseException | None = None
    for label, opts in attempts:
        try:
            raw = _cuad_load_split(
                load_dataset,
                args.hf_dataset,
                args.split,
                streaming=opts["streaming"],
                force_redownload=opts["force_redownload"],
                verification_mode=opts.get("verification_mode"),
            )
            cand = _cuad_prepare_for_scan(raw, streaming=opts["streaming"])
            if not _cuad_validate_nonempty(cand, label):
                continue
            ds = cand
            used = label
            used_streaming = opts["streaming"]
            break
        except BaseException as e:
            last_err = e
            continue

    if ds is None:
        print(
            f"Failed to load {args.hf_dataset!r} after retries.\n"
            "Try:\n"
            f"  python scripts/prepare_hf_benchmark_corpus.py cuad --force-redownload --hf-dataset {args.hf_dataset!r}\n"
            "or clear the dataset cache, e.g.\n"
            f"  rm -rf ~/.cache/huggingface/datasets/*cuad*\n",
            file=sys.stderr,
        )
        if last_err is not None:
            print(f"Last error: {last_err!r}", file=sys.stderr)
        return 1

    if used:
        print(f"  (CUAD loaded via: {used})", flush=True)

    max_rows = args.max_stream_rows if args.max_stream_rows and args.max_stream_rows > 0 else None
    try:
        by_title, n_scanned = _cuad_scan_to_by_title(
            ds,
            max_rows=max_rows,
            progress_every=args.progress_every,
        )
    except ImportError:
        return 1

    # ~511 rows + 0 contracts = broken stream / partial cache; materialized + force_redownload fixes it.
    if (
        not by_title
        and n_scanned > 0
        and n_scanned < 5000
        and args.split == "train"
        and not args.force_redownload
        and used_streaming
    ):
        print(
            "  [!] Few rows and no contract text — typical of a partial CUAD cache + streaming.\n"
            "      Retrying once with non-streaming + force_redownload (slow but rebuilds cache)…\n",
            flush=True,
        )
        try:
            raw = _cuad_load_split(
                load_dataset,
                args.hf_dataset,
                args.split,
                streaming=False,
                force_redownload=True,
                verification_mode=None,
            )
            ds2 = _cuad_prepare_for_scan(raw, streaming=False)
            if _cuad_validate_nonempty(ds2, "non-streaming + force_redownload (auto)"):
                by_title, n_scanned = _cuad_scan_to_by_title(
                    ds2,
                    max_rows=max_rows,
                    progress_every=args.progress_every,
                )
                print("  (CUAD reloaded via: auto non-streaming + force_redownload)", flush=True)
        except BaseException as e:
            print(f"  [WARN] Auto-retry failed: {e!r}", file=sys.stderr, flush=True)

    if not by_title:
        print(
            "CUAD: no contract text collected (0 rows scanned or every clause/context field was empty).\n"
            "Try:\n"
            f"  python scripts/prepare_hf_benchmark_corpus.py cuad --force-redownload --n 40 --hf-dataset {args.hf_dataset!r}\n"
            "or the smaller SQuAD-style hub:\n"
            "  python scripts/prepare_hf_benchmark_corpus.py cuad --hf-dataset theatticusproject/cuad-qa --n 40\n",
            file=sys.stderr,
        )
        return 1
    titles = sorted(by_title.keys())
    rng = random.Random(args.seed)
    rng.shuffle(titles)
    titles = titles[: args.n]
    for t in titles:
        ctx = by_title[t]
        header = [
            f"title: {t}",
            f"source: {args.hf_dataset} (Hugging Face)",
            "license: CC BY 4.0 (see dataset card)",
        ]
        _write_doc(out, t, ctx, header_lines=header, max_chars=args.max_chars_per_doc)
    print(f"Wrote {len(titles)} unique contracts to {out} ({len(by_title)} distinct in split {args.split!r}).")
    return 0


def _pg19_row_to_dict(row: object) -> dict:
    if isinstance(row, dict):
        return row
    try:
        return dict(row)
    except Exception:
        return {}


def cmd_pg19(args: argparse.Namespace, load_dataset) -> int:
    """
    By default PG-19 is loaded **streaming** and we **stop after ``--n`` books**.

    A full materialized split pulls every shard on the hub (dozens of progress bars for
    validation); that is only used with ``--full-split`` when you need a shuffled sample
    over the entire split.
    """
    split = args.split
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    take = max(1, int(args.n))

    def _write_pg19_book(row: dict, *, fallback_id: str) -> None:
        title = (row.get("short_book_title") or "").strip() or fallback_id
        text = (row.get("text") or "").strip()
        url = row.get("url") or ""
        pub = row.get("publication_date")
        header = [
            f"title: {title}",
            "source: deepmind/pg19 (Hugging Face)",
            f"split: {split}",
        ]
        if url:
            header.append(f"url: {url}")
        if pub is not None:
            header.append(f"publication_date: {pub}")
        _write_doc(out, title, text, header_lines=header, max_chars=args.max_chars_per_doc)

    def _datasets_too_new(e: BaseException) -> bool:
        return "scripts are no longer supported" in str(e).lower()

    if args.full_split:
        try:
            ds = load_dataset("deepmind/pg19", split=split)
        except RuntimeError as e:
            if _datasets_too_new(e):
                print(
                    str(e),
                    "\n\nInstall `datasets<4`, e.g.\n"
                    "  pip install 'datasets>=2.16,<4.0.0'\n",
                    file=sys.stderr,
                )
                return 1
            raise
        n_available = len(ds)
        indices = list(range(n_available))
        rng = random.Random(args.seed)
        rng.shuffle(indices)
        n_write = min(take, n_available)
        for i in indices[:n_write]:
            row = _pg19_row_to_dict(ds[i])
            _write_pg19_book(row, fallback_id=f"book_{i}")
        print(
            f"Wrote {n_write} books to {out} (split={split!r}, --full-split, {n_available} in split, seed={args.seed})."
        )
        return 0

    # Streaming: only fetch rows until we have ``take`` usable books (far fewer hub files).
    try:
        stream = load_dataset("deepmind/pg19", split=split, streaming=True)
    except RuntimeError as e:
        if _datasets_too_new(e):
            print(
                str(e),
                "\n\nInstall `datasets<4`, e.g.\n"
                "  pip install 'datasets>=2.16,<4.0.0'\n",
                file=sys.stderr,
            )
            return 1
        raise
    except Exception as e:
        print(
            f"PG-19 streaming failed ({e!r}); retry with --full-split to materialize the split.\n",
            file=sys.stderr,
        )
        return 1

    print(
        f"PG-19: streaming split={split!r}, stopping after {take} book(s) "
        "(avoids downloading the whole split). Use --full-split for shuffle over entire split.\n",
        flush=True,
    )
    written = 0
    seen = 0
    for row in stream:
        seen += 1
        d = _pg19_row_to_dict(row)
        text = (d.get("text") or "").strip()
        if len(text) < 1:
            continue
        _write_pg19_book(d, fallback_id=f"book_{seen}")
        written += 1
        if written >= take:
            break

    if written == 0:
        print(
            "PG-19: no non-empty books collected. Try --full-split or another --split.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Wrote {written} book(s) to {out} (split={split!r}, scanned {seen} stream row(s); order = hub order, not shuffled)."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a small Hugging Face dataset subset as .txt corpus files.",
    )
    sub = parser.add_subparsers(dest="preset", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--out",
            type=Path,
            help="Output directory (default: docs/benchmark_corpora/<preset> under repo root)",
        )
        p.add_argument(
            "--n",
            type=int,
            default=100,
            help="Number of documents to write (default: 100)",
        )
        p.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling/sample")
        p.add_argument(
            "--max-chars-per-doc",
            type=int,
            default=400_000,
            metavar="N",
            help="Truncate each file after N characters (default: 400000)",
        )

    p_wiki = sub.add_parser("wikipedia", help="wikimedia/wikipedia (English snapshot, streaming sample)")
    add_common(p_wiki)
    p_wiki.add_argument(
        "--wiki-config",
        default="20231101.en",
        help="Dataset config name (default: 20231101.en). Try 20231101.simple for smaller Simple English wiki.",
    )
    p_wiki.add_argument(
        "--scan-multiplier",
        type=int,
        default=50,
        metavar="K",
        help="Scan up to n*K streaming rows to find enough non-short articles (default: 50)",
    )
    p_wiki.add_argument(
        "--min-chars",
        type=int,
        default=400,
        help="Skip articles with body shorter than this (default: 400)",
    )
    p_wiki.set_defaults(func=cmd_wikipedia)

    p_cuad = sub.add_parser("cuad", help="theatticusproject/cuad (legal contracts, deduped by title)")
    add_common(p_cuad)
    p_cuad.add_argument(
        "--hf-dataset",
        default="theatticusproject/cuad",
        metavar="ID",
        help="HF dataset id (default: theatticusproject/cuad). Try theatticusproject/cuad-qa if context is missing.",
    )
    p_cuad.add_argument("--split", default="train", choices=("train", "test"), help="CUAD split (default: train)")
    p_cuad.add_argument(
        "--force-redownload",
        action="store_true",
        help="Purge and re-fetch CUAD (fixes NonMatchingSplitsSizesError / partial cache).",
    )
    p_cuad.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        metavar="N",
        help="Print a status line every N rows while scanning (default: 10000; 0 to disable).",
    )
    p_cuad.add_argument(
        "--max-stream-rows",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N rows (faster debug run; biases toward early contracts; default: scan all).",
    )
    p_cuad.set_defaults(func=cmd_cuad)

    p_pg = sub.add_parser("pg19", help="deepmind/pg19 (long-form fiction; default split validation)")
    add_common(p_pg)
    p_pg.add_argument(
        "--split",
        default="validation",
        choices=("train", "validation", "test"),
        help="PG-19 split (default: validation, 50 books — train is huge)",
    )
    p_pg.add_argument(
        "--full-split",
        action="store_true",
        help=(
            "Download and cache the entire split, then shuffle with --seed and sample --n "
            "(slow, many files). Default is streaming: stop after --n books (minimal download)."
        ),
    )
    p_pg.set_defaults(func=cmd_pg19)

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.out is None:
        args.out = repo_root / "docs" / "benchmark_corpora" / str(args.preset)
    load_dataset = _require_datasets()
    return int(args.func(args, load_dataset))


if __name__ == "__main__":
    raise SystemExit(main())

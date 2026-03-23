# Benchmark corpora (Hugging Face → `.txt` tree)

Use this when you want **fixed, reproducible subsets** of public datasets to stress-test chunking / ingest (Wikipedia-style articles, legal contracts, long fiction).

## Install

```bash
pip install -e ".[benchmark-corpora]"
```

### `datasets` version (**must be &lt; 4**)

Hugging Face **`datasets` 4.x** removed support for hubs that ship a Python **loading script** (e.g. **`deepmind/pg19`**). This repo pins **`datasets>=2.16,<4`** under `[benchmark-corpora]`.

If you already upgraded globally, run:

```bash
pip install 'datasets>=2.16,<4.0.0'
```

The extras also include **`pdfplumber`** for **CUAD** when Pdf columns are decoded; the script prefers **`select_columns(["title","context"])`** when possible.

## Export

Each subcommand writes **only `.txt`** files under `--out` (default: `storage/benchmark_corpora/<preset>/`).  
That layout matches what `scripts/start_corpus_ingest.py` expects (`.txt` under a single root).

| Preset        | Hugging Face dataset        | Good for |
|---------------|-----------------------------|----------|
| `wikipedia`   | `wikimedia/wikipedia`       | Many mid-length articles; streaming sample from English dump |
| `cuad`        | `theatticusproject/cuad`    | Legal / contract prose (deduped by contract title) |
| `pg19`        | `deepmind/pg19`             | Very long narratives (default: **validation** split, 50 books) |

Examples:

```bash
# English Wikipedia (streaming; does not download the full 6M articles)
python scripts/prepare_hf_benchmark_corpus.py wikipedia --n 250 --wiki-config 20231101.en

# Smaller / faster wiki variant (Simple English)
python scripts/prepare_hf_benchmark_corpus.py wikipedia --n 200 --wiki-config 20231101.simple

# CUAD — unique contracts from train split (~84k QA rows scanned once; no files until scan ends)
python scripts/prepare_hf_benchmark_corpus.py cuad --n 50 --split train
# Status every 5k rows; quick biased sample:
python scripts/prepare_hf_benchmark_corpus.py cuad --n 20 --progress-every 5000 --max-stream-rows 25000

# PG-19 — streams until --n books (few shard downloads); hub order, not shuffled
python scripts/prepare_hf_benchmark_corpus.py pg19 --n 20 --split validation
python scripts/prepare_hf_benchmark_corpus.py pg19 --n 5  --split test
# Entire split + shuffled sample (many downloads; old behavior)
python scripts/prepare_hf_benchmark_corpus.py pg19 --n 5 --split validation --full-split
```

Custom output dir (e.g. mimic `docs/`):

```bash
python scripts/prepare_hf_benchmark_corpus.py wikipedia --out ./docs_wiki_sample --n 100
```

## Ingest + chunk (Temporal)

With `temporal server start-dev` and `python temporal_workers.py` running:

```bash
python scripts/start_corpus_ingest.py storage/benchmark_corpora/wikipedia --wait
python scripts/start_strategy_chunking.py s_default --wait
python scripts/run_with_ecs.py --interactive
```

## Troubleshooting (CUAD)

If you see `NonMatchingSplitsSizesError` (expected ~84k train rows but only a few hundred recorded), the Hugging Face **cache is usually partial or stale**. This script loads CUAD in **streaming** mode first to avoid that check; if it still fails:

```bash
python scripts/prepare_hf_benchmark_corpus.py cuad --force-redownload --n 40
# or:
rm -rf ~/.cache/huggingface/datasets/theatticusproject___cuad
```

If the scan finishes with **~511 rows and 0 unique contracts** (every `context` empty), that was usually **streaming + column narrowing** on a partial cache. The exporter avoids `select_columns` while streaming and may **auto-retry** once with non-streaming + `force_redownload`. You can also pass **`--hf-dataset theatticusproject/cuad-qa`** (SQuAD-style fields) or clear caches under **`~/.cache/huggingface/hub/`** as well as `datasets/` — an empty `datasets/` folder does not mean nothing is cached.

## Notes

- **`storage/`** is gitignored here; exported corpora stay local unless you copy them elsewhere.
- Wikipedia export **truncates** very long articles (`--max-chars-per-doc`, default 400k) so single files stay manageable.
- CUAD is **CC BY 4.0**; cite the dataset card when publishing results.
- PG-19 **train** has tens of thousands of books — prefer `validation` / `test` or keep `--n` small.

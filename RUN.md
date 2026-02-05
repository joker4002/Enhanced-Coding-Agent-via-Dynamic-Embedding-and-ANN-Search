# RUN

This repo contains scripts for:

- Building document/code chunks.
- Embedding chunks and building FAISS ANN indices.
- Querying one or multiple indices.
- Rebuilding `test_20_zh.csv` discussion questions from `discussions_with_replies.csv` and exporting solutions.

## Prerequisites

- Python 3.10+ recommended.
- Git (optional).

### FAISS note (important)

- On Linux/macOS, `faiss-cpu` is installed automatically from `requirements.txt`.
- On Windows, FAISS wheels are usually not available via pip. Use **conda**:

```powershell
conda create -n ann python=3.11 -y
conda activate ann
conda install -c conda-forge faiss-cpu -y
pip install -r requirements.txt
```

## Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

## 1) Build chunks (docs + code)

### Docs chunks

This reads `./docs` (RST/MD) and writes `docs_chunks.jsonl`.

```powershell
python extract_docs_chunks.py --docs d:\499\docs --out d:\499\docs_chunks.jsonl
```

### Code chunks

This extracts class/function/method blocks from Flask source and writes `flask_chunks.jsonl`.

```powershell
python extract_code_chunks.py --src d:\499\src\flask-3.1.2\flask-3.1.2\src --out d:\499\flask_chunks.jsonl --include-methods
```

## 2) Embed chunks and build FAISS indices

### Docs index

```powershell
python embed_chunks.py --mode embed --in d:\499\docs_chunks.jsonl --normalize --faiss-index d:\499\docs.index
```

### Code index

```powershell
python embed_chunks.py --mode embed --in d:\499\flask_chunks.jsonl --normalize --faiss-index d:\499\flask.index
```

Outputs:

- `*.vectors.npy` (embeddings)
- `*.meta.jsonl` (row -> metadata)
- `*.index` (FAISS index)

## 3) Query indices

### Single-index query (simple)

```powershell
python embed_chunks.py --mode query --index d:\499\docs.index --chunks d:\499\docs_chunks.jsonl --query "What is app context?" --top-k 5 --normalize
```

### Multi-index query (merged)

`ann_query_multi.py` expects each `--source` in the format:

```
label|index_path|chunks_jsonl
```

Example (docs + code merged):

```powershell
python ann_query_multi.py \
  --query "Why teardown callbacks should not swallow exceptions?" \
  --source "docs|d:\499\docs.index|d:\499\docs_chunks.jsonl" \
  --source "code|d:\499\flask.index|d:\499\flask_chunks.jsonl" \
  --top-k 10 --per-index-k 20 --normalize --score-norm zscore --format compact
```

Optional flags you may find useful:

- `--split`: show per-source tables.
- `--pick label:rank,label:rank`: expand full snippets for selected ranks when using `--split`.

## 4) Rebuild `test_20_zh.csv` discussion questions from replies + export solutions

Input files used:

- `test_20_zh.csv`
- `questions.csv`
- `comment_questions.csv`
- `comment_questions.jsonl`
- `discussions_with_replies.csv`

This will:

- Replace the original 20 discussion questions in `test_20_zh.csv` with 20 discussions that have concrete solution replies.
- Remove Issue rows that have **no local comment-based solution** and fill the gaps with Issue rows that do.
- Write a solutions CSV for the final dataset.

Run:

```powershell
python rebuild_test_20_from_replies.py \
  --test-in d:\499\test_20_zh.csv \
  --questions d:\499\questions.csv \
  --comment-csv d:\499\comment_questions.csv \
  --comment-jsonl d:\499\comment_questions.jsonl \
  --discussions d:\499\discussions_with_replies.csv \
  --test-out d:\499\test_20_zh.updated.csv \
  --solutions-out d:\499\solutions_from_test_20.csv
```

Outputs:

- `test_20_zh.updated.csv`
- `solutions_from_test_20.csv`

## Troubleshooting

- If you see `faiss is not available`, install FAISS (see the Windows conda note above).
- If `sentence-transformers` pulls in a large `torch` dependency: this is expected.

import argparse
import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def _iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _build_content(
    rec: Dict,
    include_qualname: bool,
    include_docstring: bool,
    include_code: bool,
) -> str:
    parts: List[str] = []

    if include_qualname and rec.get("qualname"):
        parts.append(str(rec["qualname"]))

    if include_docstring and rec.get("docstring"):
        parts.append(str(rec["docstring"]))

    if include_code and rec.get("code"):
        parts.append(str(rec["code"]))

    return "\n\n".join(parts)


def embed_chunks(
    in_jsonl: str,
    model_name: str,
    batch_size: int,
    normalize: bool,
    include_qualname: bool,
    include_docstring: bool,
    include_code: bool,
) -> Tuple[np.ndarray, List[Dict]]:
    records: List[Dict] = []
    texts: List[str] = []

    for rec in _iter_jsonl(in_jsonl):
        content = _build_content(
            rec,
            include_qualname=include_qualname,
            include_docstring=include_docstring,
            include_code=include_code,
        )
        if not content.strip():
            continue

        records.append(rec)
        texts.append(content)

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )

    vectors_np = np.asarray(vectors, dtype=np.float32)
    return vectors_np, records


def write_meta_jsonl(records: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for i, rec in enumerate(records):
            meta = {
                "row": i,
                "chunk_id": rec.get("chunk_id"),
                "file_path": rec.get("file_path"),
                "symbol_type": rec.get("symbol_type"),
                "name": rec.get("name"),
                "qualname": rec.get("qualname"),
                "start_line": rec.get("start_line"),
                "end_line": rec.get("end_line"),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def maybe_write_faiss(vectors: np.ndarray, out_index_path: str) -> None:
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "faiss is not available. Install faiss-cpu (or faiss) to use --faiss-index."
        ) from e

    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    os.makedirs(os.path.dirname(out_index_path) or ".", exist_ok=True)
    faiss.write_index(index, out_index_path)


def _ensure_faiss() -> "object":
    try:
        import faiss  # type: ignore

        return faiss
    except Exception as e:
        raise RuntimeError(
            "faiss is not available. Install faiss-cpu (or faiss) to use --mode query."
        ) from e


def query_chunks(
    index_path: str,
    chunks_jsonl: str,
    query: str,
    model_name: str,
    top_k: int,
    normalize: bool,
    show_code: bool,
    max_code_chars: int,
) -> None:
    faiss = _ensure_faiss()
    index = faiss.read_index(index_path)

    model = SentenceTransformer(model_name)
    q = model.encode([query], normalize_embeddings=normalize)
    q = np.asarray(q, dtype=np.float32)

    scores, ids = index.search(q, top_k)
    scores = scores[0]
    ids = ids[0]

    chunks = list(_iter_jsonl(chunks_jsonl))

    for rank, (score, row) in enumerate(zip(scores.tolist(), ids.tolist()), start=1):
        if row < 0 or row >= len(chunks):
            continue

        rec = chunks[row]
        header = {
            "rank": rank,
            "score": score,
            "row": row,
            "chunk_id": rec.get("chunk_id"),
            "symbol_type": rec.get("symbol_type"),
            "qualname": rec.get("qualname"),
            "file_path": rec.get("file_path"),
            "start_line": rec.get("start_line"),
            "end_line": rec.get("end_line"),
        }
        print(json.dumps(header, ensure_ascii=False))

        if show_code:
            code = rec.get("code") or ""
            if max_code_chars > 0 and len(code) > max_code_chars:
                code = code[: max_code_chars - 20] + "\n...<truncated>...\n"
            print(code)
            print("-" * 80)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["embed", "query"], default="embed")
    p.add_argument("--in", dest="in_jsonl", required=False)
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--normalize", action="store_true")

    p.add_argument("--no-qualname", action="store_true")
    p.add_argument("--no-docstring", action="store_true")
    p.add_argument("--no-code", action="store_true")

    p.add_argument("--out-vectors", default=None)
    p.add_argument("--out-meta", default=None)
    p.add_argument("--faiss-index", default=None)

    p.add_argument("--index", dest="index_path", default=None)
    p.add_argument("--chunks", dest="chunks_jsonl", default=None)
    p.add_argument("--query", dest="query_text", default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--show-code", action="store_true")
    p.add_argument("--max-code-chars", type=int, default=1600)

    args = p.parse_args(argv)

    if args.mode == "query":
        if not args.index_path:
            raise SystemExit("--index is required when --mode query")
        if not args.chunks_jsonl:
            raise SystemExit("--chunks is required when --mode query")
        if not args.query_text:
            raise SystemExit("--query is required when --mode query")

        query_chunks(
            index_path=args.index_path,
            chunks_jsonl=args.chunks_jsonl,
            query=args.query_text,
            model_name=args.model,
            top_k=args.top_k,
            normalize=args.normalize,
            show_code=args.show_code,
            max_code_chars=args.max_code_chars,
        )
        return 0

    if not args.in_jsonl:
        raise SystemExit("--in is required when --mode embed")

    out_vectors = args.out_vectors or os.path.splitext(args.in_jsonl)[0] + ".vectors.npy"
    out_meta = args.out_meta or os.path.splitext(args.in_jsonl)[0] + ".meta.jsonl"

    vectors, records = embed_chunks(
        in_jsonl=args.in_jsonl,
        model_name=args.model,
        batch_size=args.batch_size,
        normalize=args.normalize,
        include_qualname=not args.no_qualname,
        include_docstring=not args.no_docstring,
        include_code=not args.no_code,
    )

    os.makedirs(os.path.dirname(out_vectors) or ".", exist_ok=True)
    np.save(out_vectors, vectors)
    write_meta_jsonl(records, out_meta)

    print(f"rows={len(records)}")
    print(f"dim={int(vectors.shape[1]) if vectors.size else 0}")
    print(f"vectors_npy={out_vectors}")
    print(f"meta_jsonl={out_meta}")

    if args.faiss_index:
        maybe_write_faiss(vectors, args.faiss_index)
        print(f"faiss_index={args.faiss_index}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

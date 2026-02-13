from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

@dataclass(frozen=True)
class CodeLocation:
    file: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class GroundTruthEntry:
    answer_source: str
    code_locations: List[CodeLocation]
    doc_locations: List[CodeLocation]


def _ensure_faiss() -> "object":
    try:
        import faiss  # type: ignore

        return faiss
    except Exception as e:
        raise RuntimeError(
            "faiss is not available. Install faiss-cpu (or faiss) to query indices."
        ) from e


def _iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_csv_dict(path: str) -> Iterator[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {k: (v or "") for k, v in row.items()}


def _parse_code_location_token(token: str) -> Optional[CodeLocation]:
    tok = (token or "").strip()
    if not tok:
        return None

    if ":" not in tok:
        return None

    file_part, rest = tok.rsplit(":", 1)
    file_part = file_part.strip().replace("\\", "/").strip("/")
    rest = rest.strip()
    if not file_part or not rest:
        return None

    if "-" in rest:
        a, b = rest.split("-", 1)
        try:
            start = int(a)
            end = int(b)
        except ValueError:
            return None
    else:
        try:
            start = int(rest)
        except ValueError:
            return None
        end = start

    if start <= 0 or end <= 0:
        return None
    if end < start:
        start, end = end, start
    return CodeLocation(file=file_part.lower(), start_line=start, end_line=end)


def _parse_code_locations(raw: str) -> List[CodeLocation]:
    s = (raw or "").strip()
    if not s:
        return []
    out: List[CodeLocation] = []
    for part in s.split(";"):
        loc = _parse_code_location_token(part)
        if loc is not None:
            out.append(loc)
    return out


def _dedupe_locations(locs: List[CodeLocation]) -> List[CodeLocation]:
    if not locs:
        return []
    seen = set()
    uniq: List[CodeLocation] = []
    for loc in locs:
        key = (loc.file, loc.start_line, loc.end_line)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(loc)
    return uniq


def _normalize_answer_source(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in {"code", "docs", "mixed", "none"}:
        return s
    return ""


def _load_groundtruth_entries(path: str) -> Dict[str, GroundTruthEntry]:
    merged: Dict[str, GroundTruthEntry] = {}
    for row in _iter_csv_dict(path):
        rid = (row.get("id") or "").strip()
        if not rid:
            continue

        answer_source = _normalize_answer_source(row.get("answer_source") or "")
        code_locs = _dedupe_locations(_parse_code_locations(row.get("code_locations") or ""))
        doc_locs = _dedupe_locations(_parse_code_locations(row.get("doc_locations") or ""))

        if not answer_source:
            if code_locs and doc_locs:
                answer_source = "mixed"
            elif code_locs:
                answer_source = "code"
            elif doc_locs:
                answer_source = "docs"
            else:
                answer_source = "none"

        merged[rid] = GroundTruthEntry(
            answer_source=answer_source,
            code_locations=code_locs,
            doc_locations=doc_locs,
        )

    return merged


def _load_groundtruth_locations(path: str) -> Dict[str, List[CodeLocation]]:
    entries = _load_groundtruth_entries(path)
    return {
        rid: entry.code_locations
        for rid, entry in entries.items()
        if entry.code_locations
    }


def _normalize_result_file_path(file_path: object) -> str:
    if not isinstance(file_path, str) or not file_path.strip():
        return ""

    p = file_path.replace("\\", "/")
    lo = p.lower()
    for marker in ("/src/flask/", "/docs/", "/doc/"):
        idx = lo.rfind(marker)
        if idx >= 0:
            rel = p[idx + len(marker) :]
            return rel.strip("/").lower()
    return os.path.basename(p).lower()


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _build_rerank_text(rec: Dict) -> str:
    parts: List[str] = []
    qn = rec.get("qualname")
    if isinstance(qn, str) and qn.strip():
        parts.append(qn)
    ds = rec.get("docstring")
    if isinstance(ds, str) and ds.strip():
        parts.append(ds)
    snippet = _get_snippet(rec)
    if snippet:
        parts.append(snippet)
    fp = rec.get("file_path")
    if isinstance(fp, str) and fp.strip():
        parts.append(os.path.basename(fp))
    return "\n".join(parts)


def _bm25_scores(query_tokens: List[str], docs_tokens: List[List[str]]) -> List[float]:
    if not query_tokens or not docs_tokens:
        return [0.0 for _ in docs_tokens]

    n_docs = len(docs_tokens)
    doc_lens = [len(toks) for toks in docs_tokens]
    avgdl = (sum(doc_lens) / float(n_docs)) if n_docs else 0.0
    if avgdl <= 1e-12:
        return [0.0 for _ in docs_tokens]

    q_terms = list(dict.fromkeys(query_tokens))
    df: Dict[str, int] = {t: 0 for t in q_terms}
    for toks in docs_tokens:
        s = set(toks)
        for t in q_terms:
            if t in s:
                df[t] += 1

    k1 = 1.5
    b = 0.75

    idf: Dict[str, float] = {}
    for t in q_terms:
        n = df.get(t, 0)
        idf[t] = math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))

    out: List[float] = []
    for toks, dl in zip(docs_tokens, doc_lens):
        tf: Dict[str, int] = {}
        for tok in toks:
            if tok in df:
                tf[tok] = tf.get(tok, 0) + 1

        score = 0.0
        denom_const = k1 * (1.0 - b + b * (float(dl) / avgdl))
        for t in q_terms:
            f = tf.get(t, 0)
            if f <= 0:
                continue
            score += idf[t] * (float(f) * (k1 + 1.0)) / (float(f) + denom_const)
        out.append(score)

    return out


def _minmax_norm(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    rng = hi - lo
    if rng <= 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / rng for s in scores]


def _rerank_merged_bm25(
    merged: List[Tuple[float, float, str, int, Dict]],
    query_text: str,
    pool_k: int,
    alpha: float,
) -> List[Tuple[float, float, str, int, Dict]]:
    if not merged:
        return merged
    if pool_k <= 0:
        return merged

    pool = merged[: min(int(pool_k), len(merged))]
    q_tokens = _tokenize(query_text)
    if not q_tokens:
        return merged

    docs = [_tokenize(_build_rerank_text(rec)) for (_s, _rs, _lab, _row, rec) in pool]
    lex_scores = _bm25_scores(q_tokens, docs)
    vec_scores = [float(s) for (s, _rs, _lab, _row, _rec) in pool]

    lex_norm = _minmax_norm(lex_scores)
    vec_norm = _minmax_norm(vec_scores)

    scored: List[Tuple[float, int]] = []
    for i, (vn, ln) in enumerate(zip(vec_norm, lex_norm)):
        combined = float(alpha) * float(vn) + (1.0 - float(alpha)) * float(ln)
        scored.append((combined, i))

    scored.sort(key=lambda t: t[0], reverse=True)
    reranked_pool = [pool[i] for _c, i in scored]
    return reranked_pool + merged[len(pool) :]


def _rerank_source_bm25(
    items: List[Tuple[float, float, int, Dict]],
    query_text: str,
    pool_k: int,
    alpha: float,
) -> List[Tuple[float, float, int, Dict]]:
    if not items:
        return items
    if pool_k <= 0:
        return items

    pool = items[: min(int(pool_k), len(items))]
    q_tokens = _tokenize(query_text)
    if not q_tokens:
        return items

    docs = [_tokenize(_build_rerank_text(rec)) for (_s, _rs, _row, rec) in pool]
    lex_scores = _bm25_scores(q_tokens, docs)
    vec_scores = [float(s) for (s, _rs, _row, _rec) in pool]

    lex_norm = _minmax_norm(lex_scores)
    vec_norm = _minmax_norm(vec_scores)

    scored: List[Tuple[float, int]] = []
    for i, (vn, ln) in enumerate(zip(vec_norm, lex_norm)):
        combined = float(alpha) * float(vn) + (1.0 - float(alpha)) * float(ln)
        scored.append((combined, i))

    scored.sort(key=lambda t: t[0], reverse=True)
    reranked_pool = [pool[i] for _c, i in scored]
    return reranked_pool + items[len(pool) :]


def _select_top_results(
    sources: Sequence[SourceSpec],
    q: "object",
    query_text: str,
    top_k: int,
    per_index_k: int,
    score_norm: str,
    rerank: str,
    rerank_pool: int,
    rerank_alpha: float,
    chunks_cache: Dict[str, List[Dict]],
    index_cache: Dict[str, "object"],
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    faiss = _ensure_faiss()

    merged: List[Tuple[float, float, str, int, Dict]] = []
    per_source: Dict[str, List[Tuple[float, float, int, Dict]]] = {}

    for src in sources:
        if src.index_path not in index_cache:
            index_cache[src.index_path] = faiss.read_index(src.index_path)

        index = index_cache[src.index_path]
        scores, ids = index.search(q, per_index_k)
        scores0 = scores[0].tolist()
        ids0 = ids[0].tolist()

        if score_norm == "zscore":
            norm_scores0 = _zscore([float(s) for s in scores0])
        elif score_norm == "minmax":
            norm_scores0 = _minmax([float(s) for s in scores0])
        else:
            norm_scores0 = [float(s) for s in scores0]

        if src.chunks_jsonl not in chunks_cache:
            chunks_cache[src.chunks_jsonl] = list(_iter_jsonl(src.chunks_jsonl))
        chunks = chunks_cache[src.chunks_jsonl]

        for raw_score, norm_score, row in zip(scores0, norm_scores0, ids0):
            if row is None or row < 0 or row >= len(chunks):
                continue
            rec = chunks[row]
            merged.append((float(norm_score), float(raw_score), src.label, int(row), rec))
            per_source.setdefault(src.label, []).append(
                (float(norm_score), float(raw_score), int(row), rec)
            )

    merged.sort(key=lambda t: t[0], reverse=True)

    for label in per_source:
        per_source[label].sort(key=lambda t: t[0], reverse=True)

    if rerank == "bm25":
        merged = _rerank_merged_bm25(
            merged=merged,
            query_text=query_text,
            pool_k=int(rerank_pool),
            alpha=float(rerank_alpha),
        )
        for label in list(per_source.keys()):
            per_source[label] = _rerank_source_bm25(
                items=per_source[label],
                query_text=query_text,
                pool_k=int(rerank_pool),
                alpha=float(rerank_alpha),
            )

    top = merged[: max(top_k, 0)]

    top_headers: List[Dict] = []
    for rank, (score, raw_score, label, row, rec) in enumerate(top, start=1):
        top_headers.append(
            {
                "rank": rank,
                "score": score,
                "raw_score": raw_score,
                "source": label,
                "row": row,
                "chunk_id": rec.get("chunk_id"),
                "symbol_type": rec.get("symbol_type"),
                "qualname": rec.get("qualname"),
                "file_path": rec.get("file_path"),
                "start_line": rec.get("start_line"),
                "end_line": rec.get("end_line"),
            }
        )

    per_source_headers: Dict[str, List[Dict]] = {}
    for label, items in per_source.items():
        headers: List[Dict] = []
        table = items[: max(top_k, 0)]
        for rank, (score, raw_score, row, rec) in enumerate(table, start=1):
            headers.append(
                {
                    "rank": rank,
                    "score": score,
                    "raw_score": raw_score,
                    "source": label,
                    "row": row,
                    "chunk_id": rec.get("chunk_id"),
                    "symbol_type": rec.get("symbol_type"),
                    "qualname": rec.get("qualname"),
                    "file_path": rec.get("file_path"),
                    "start_line": rec.get("start_line"),
                    "end_line": rec.get("end_line"),
                }
            )
        per_source_headers[label] = headers

    return top_headers, per_source_headers


def _evaluate_recall_at_k(
    results: List[Dict],
    gt_locs: List[CodeLocation],
    ks: Sequence[int],
    allowed_sources: Optional[set[str]],
) -> Dict[int, float]:
    if not gt_locs:
        return {int(k): 0.0 for k in ks}

    gt_by_file: Dict[str, List[CodeLocation]] = {}
    for loc in gt_locs:
        gt_by_file.setdefault(loc.file, []).append(loc)

    out: Dict[int, float] = {}
    for k in ks:
        k_int = int(k)
        matched = set()
        for header in results[: max(k_int, 0)]:
            if allowed_sources is not None:
                src = str(header.get("source") or "")
                if src not in allowed_sources:
                    continue

            f = _normalize_result_file_path(header.get("file_path"))
            if not f or f not in gt_by_file:
                continue

            start = header.get("start_line")
            end = header.get("end_line")

            if not isinstance(start, int) or start <= 0:
                continue
            if not isinstance(end, int) or end < start:
                end = start

            for loc in gt_by_file[f]:
                if _ranges_overlap(start, end, loc.start_line, loc.end_line):
                    matched.add((loc.file, loc.start_line, loc.end_line))

        out[k_int] = float(len(matched)) / float(len(gt_locs))

    return out


def _parse_ks(raw: str) -> List[int]:
    s = (raw or "").strip()
    if not s:
        return []
    parts = [p for p in s.replace(",", " ").split() if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError:
            continue
        if v > 0 and v not in out:
            out.append(v)
    return out


def _iter_questions_csv(
    path: str,
    id_col: str,
    question_col: str,
    limit: Optional[int],
) -> Iterator[Tuple[str, str]]:
    n = 0
    for row in _iter_csv_dict(path):
        rid = (row.get(id_col) or "").strip()
        q = (row.get(question_col) or "").strip()
        if not rid or not q:
            continue
        yield rid, q
        n += 1
        if limit is not None and n >= int(limit):
            return


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...<truncated>...\n"


def _get_snippet(rec: Dict) -> str:
    code = rec.get("code")
    if isinstance(code, str) and code.strip():
        return code

    for k in ("text", "content", "chunk", "body"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v

    return ""


def _zscore(scores: List[float]) -> List[float]:
    if not scores:
        return []
    m = sum(scores) / float(len(scores))
    var = sum((s - m) ** 2 for s in scores) / float(len(scores))
    std = var ** 0.5
    if std <= 1e-12:
        return [0.0 for _ in scores]
    return [(s - m) / std for s in scores]


def _minmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    rng = hi - lo
    if rng <= 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / rng for s in scores]


@dataclass
class SourceSpec:
    label: str
    index_path: str
    chunks_jsonl: str


def _parse_source_spec(raw: str) -> SourceSpec:
    # Use | as separator to avoid conflicts with Windows drive letters.
    # Format: label|index_path|chunks_jsonl
    parts = raw.split("|", 2)
    if len(parts) != 3 or not all(p.strip() for p in parts):
        raise ValueError(
            "Invalid --source. Expected format: label|index_path|chunks_jsonl"
        )
    return SourceSpec(label=parts[0].strip(), index_path=parts[1].strip(), chunks_jsonl=parts[2].strip())


def query_multi(
    sources: Sequence[SourceSpec],
    query: str,
    model_name: str,
    top_k: int,
    per_index_k: int,
    normalize: bool,
    score_norm: str,
    output_format: str,
    out_path: Optional[str],
    quiet: bool,
    show_snippet: bool,
    interactive: bool,
    pick_ranks: Optional[List[int]],
    split: bool,
    pick: Optional[Dict[str, List[int]]],
    max_code_chars: int,
    rerank: str,
    rerank_pool: int,
    rerank_alpha: float,
) -> None:
    faiss = _ensure_faiss()

    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)
    q = model.encode([query], normalize_embeddings=normalize)

    try:
        import numpy as np  # type: ignore

        q = np.asarray(q, dtype=np.float32)
    except Exception as e:
        raise RuntimeError("numpy is required") from e

    merged: List[Tuple[float, float, str, int, Dict]] = []
    per_source: Dict[str, List[Tuple[float, float, int, Dict]]] = {}

    # Load chunks per source only once (needed for row -> record mapping).
    chunks_cache: Dict[str, List[Dict]] = {}

    for src in sources:
        index = faiss.read_index(src.index_path)
        scores, ids = index.search(q, per_index_k)
        scores0 = scores[0].tolist()
        ids0 = ids[0].tolist()

        if score_norm == "zscore":
            norm_scores0 = _zscore([float(s) for s in scores0])
        elif score_norm == "minmax":
            norm_scores0 = _minmax([float(s) for s in scores0])
        else:
            norm_scores0 = [float(s) for s in scores0]

        if src.chunks_jsonl not in chunks_cache:
            chunks_cache[src.chunks_jsonl] = list(_iter_jsonl(src.chunks_jsonl))
        chunks = chunks_cache[src.chunks_jsonl]

        for raw_score, norm_score, row in zip(scores0, norm_scores0, ids0):
            if row is None or row < 0 or row >= len(chunks):
                continue
            rec = chunks[row]
            merged.append((float(norm_score), float(raw_score), src.label, int(row), rec))
            per_source.setdefault(src.label, []).append((float(norm_score), float(raw_score), int(row), rec))

    merged.sort(key=lambda t: t[0], reverse=True)

    for label in per_source:
        per_source[label].sort(key=lambda t: t[0], reverse=True)

    if rerank == "bm25":
        merged = _rerank_merged_bm25(
            merged=merged,
            query_text=query,
            pool_k=int(rerank_pool),
            alpha=float(rerank_alpha),
        )
        for label in list(per_source.keys()):
            per_source[label] = _rerank_source_bm25(
                items=per_source[label],
                query_text=query,
                pool_k=int(rerank_pool),
                alpha=float(rerank_alpha),
            )

    top = merged[: max(top_k, 0)]

    out_f: Optional[TextIO] = None
    if out_path:
        out_f = open(out_path, "w", encoding="utf-8", newline="\n")

    try:
        if split:
            for src in sources:
                label = src.label
                items = per_source.get(label) or []
                table = items[: max(top_k, 0)]

                if output_format == "compact" and not quiet:
                    print(f"source={label}")
                    print("rank\tscore\traw_score\tqualname\tlocation")

                for rank, (score, raw_score, row, rec) in enumerate(table, start=1):
                    header = {
                        "rank": rank,
                        "score": score,
                        "raw_score": raw_score,
                        "source": label,
                        "row": row,
                        "chunk_id": rec.get("chunk_id"),
                        "symbol_type": rec.get("symbol_type"),
                        "qualname": rec.get("qualname"),
                        "file_path": rec.get("file_path"),
                        "start_line": rec.get("start_line"),
                        "end_line": rec.get("end_line"),
                    }

                    if show_snippet:
                        snippet = _get_snippet(rec)
                        if snippet:
                            header["snippet"] = _truncate(snippet, max_chars=max_code_chars)

                    if out_f is not None:
                        out_f.write(json.dumps(header, ensure_ascii=False) + "\n")

                    if quiet:
                        continue

                    if output_format == "jsonl":
                        print(json.dumps(header, ensure_ascii=False))
                    else:
                        qualname = header.get("qualname") or ""
                        file_path = header.get("file_path") or ""
                        start_line = header.get("start_line")
                        end_line = header.get("end_line")

                        base = os.path.basename(str(file_path)) if file_path else ""
                        loc = base or str(file_path) or ""
                        if isinstance(start_line, int) and isinstance(end_line, int) and start_line > 0 and end_line >= start_line:
                            loc = f"{loc}:{start_line}-{end_line}"
                        elif isinstance(start_line, int) and start_line > 0:
                            loc = f"{loc}:{start_line}"

                        print(
                            f"{rank}\t{float(score):.6f}\t{float(raw_score):.6f}\t{qualname}\t{loc}".rstrip()
                        )

                    if show_snippet and header.get("snippet"):
                        print(header["snippet"])
                        print("-" * 80)

            if pick and not quiet and output_format == "compact":
                for label, ranks in pick.items():
                    items = per_source.get(label) or []
                    table = items[: max(top_k, 0)]
                    if not table:
                        print(f"No results available for source={label}.")
                        continue

                    wanted = []
                    seen = set()
                    for r in ranks:
                        if r in seen:
                            continue
                        seen.add(r)
                        wanted.append(r)

                    for sel in wanted:
                        if sel < 1 or sel > len(table):
                            print(f"Rank out of range (skipped): {label}:{sel}. Valid range: 1-{len(table)}")
                            continue
                        score, raw_score, row, rec = table[sel - 1]
                        snippet = _get_snippet(rec)
                        if not snippet:
                            print(f"No snippet content found for {label}:{sel}.")
                            continue

                        file_path = rec.get("file_path") or ""
                        start_line = rec.get("start_line")
                        end_line = rec.get("end_line")
                        base = os.path.basename(str(file_path)) if file_path else ""
                        loc = base or str(file_path) or ""
                        if isinstance(start_line, int) and isinstance(end_line, int) and start_line > 0 and end_line >= start_line:
                            loc = f"{loc}:{start_line}-{end_line}"
                        elif isinstance(start_line, int) and start_line > 0:
                            loc = f"{loc}:{start_line}"

                        qualname = rec.get("qualname") or ""
                        print(
                            f"pick={label}:{sel}\tscore={float(score):.6f}\traw_score={float(raw_score):.6f}\t{qualname}\t{loc}".rstrip()
                        )
                        print(snippet)
                        print("-" * 80)
        else:
            if output_format == "compact" and not quiet:
                print("rank\tscore\traw_score\tsource\tqualname\tlocation")

            for rank, (score, raw_score, label, row, rec) in enumerate(top, start=1):
                header = {
                    "rank": rank,
                    "score": score,
                    "raw_score": raw_score,
                    "source": label,
                    "row": row,
                    "chunk_id": rec.get("chunk_id"),
                    "symbol_type": rec.get("symbol_type"),
                    "qualname": rec.get("qualname"),
                    "file_path": rec.get("file_path"),
                    "start_line": rec.get("start_line"),
                    "end_line": rec.get("end_line"),
                }

                if show_snippet:
                    snippet = _get_snippet(rec)
                    if snippet:
                        header["snippet"] = _truncate(snippet, max_chars=max_code_chars)

                if out_f is not None:
                    out_f.write(json.dumps(header, ensure_ascii=False) + "\n")

                if quiet:
                    continue

                if output_format == "jsonl":
                    print(json.dumps(header, ensure_ascii=False))
                else:
                    qualname = header.get("qualname") or ""
                    file_path = header.get("file_path") or ""
                    start_line = header.get("start_line")
                    end_line = header.get("end_line")

                    base = os.path.basename(str(file_path)) if file_path else ""
                    loc = base or str(file_path) or ""
                    if isinstance(start_line, int) and isinstance(end_line, int) and start_line > 0 and end_line >= start_line:
                        loc = f"{loc}:{start_line}-{end_line}"
                    elif isinstance(start_line, int) and start_line > 0:
                        loc = f"{loc}:{start_line}"

                    print(
                        f"{rank}\t{float(score):.6f}\t{float(raw_score):.6f}\t{label}\t{qualname}\t{loc}".rstrip()
                    )

                if show_snippet and header.get("snippet"):
                    print(header["snippet"])
                    print("-" * 80)

            if pick_ranks and not quiet and output_format == "compact" and len(top) > 0:
                wanted = []
                seen = set()
                for r in pick_ranks:
                    if r in seen:
                        continue
                    seen.add(r)
                    wanted.append(r)

                for sel in wanted:
                    if sel < 1 or sel > len(top):
                        print(f"Rank out of range (skipped): {sel}. Valid range: 1-{len(top)}")
                        continue

                    score, raw_score, label, row, rec = top[sel - 1]
                    snippet = _get_snippet(rec)
                    if not snippet:
                        print(f"No snippet content found for rank {sel}.")
                        continue

                    file_path = rec.get("file_path") or ""
                    start_line = rec.get("start_line")
                    end_line = rec.get("end_line")
                    base = os.path.basename(str(file_path)) if file_path else ""
                    loc = base or str(file_path) or ""
                    if isinstance(start_line, int) and isinstance(end_line, int) and start_line > 0 and end_line >= start_line:
                        loc = f"{loc}:{start_line}-{end_line}"
                    elif isinstance(start_line, int) and start_line > 0:
                        loc = f"{loc}:{start_line}"

                    qualname = rec.get("qualname") or ""
                    print(
                        f"rank={sel}\tscore={float(score):.6f}\traw_score={float(raw_score):.6f}\tsource={label}\t{qualname}\t{loc}".rstrip()
                    )
                    print(snippet)
                    print("-" * 80)

            if interactive and not quiet and output_format == "compact" and len(top) > 0:
                stdin_is_tty = bool(
                    sys.stdin is not None
                    and hasattr(sys.stdin, "isatty")
                    and sys.stdin.isatty()
                )
                if not stdin_is_tty:
                    print(
                        "Warning: stdin is not reported as a TTY. If your runner redirects stdin (IDE/conda run), "
                        "interactive input may not work; in that case the prompt will exit automatically."
                    )

                if show_snippet:
                    print(
                        "Interactive mode note: --show-snippet is enabled, so snippets were already printed for each result."
                    )

                while True:
                    try:
                        raw = input(
                            f"Select a rank to show snippet (1-{len(top)}), or press Enter to exit: "
                        ).strip()
                    except EOFError:
                        break
                    if raw == "" or raw.lower() in {"q", "quit", "exit"}:
                        break

                    try:
                        sel = int(raw)
                    except ValueError:
                        print("Please enter an integer rank, or press Enter to exit.")
                        continue

                    if sel < 1 or sel > len(top):
                        print(f"Rank out of range. Valid range: 1-{len(top)}")
                        continue

                    score, raw_score, label, row, rec = top[sel - 1]
                    snippet = _get_snippet(rec)
                    if not snippet:
                        print("No snippet content found in the selected record.")
                        continue

                    file_path = rec.get("file_path") or ""
                    start_line = rec.get("start_line")
                    end_line = rec.get("end_line")
                    base = os.path.basename(str(file_path)) if file_path else ""
                    loc = base or str(file_path) or ""
                    if isinstance(start_line, int) and isinstance(end_line, int) and start_line > 0 and end_line >= start_line:
                        loc = f"{loc}:{start_line}-{end_line}"
                    elif isinstance(start_line, int) and start_line > 0:
                        loc = f"{loc}:{start_line}"

                    qualname = rec.get("qualname") or ""
                    print(
                        f"rank={sel}\tscore={float(score):.6f}\traw_score={float(raw_score):.6f}\tsource={label}\t{qualname}\t{loc}".rstrip()
                    )
                    print(snippet)
                    print("-" * 80)
    finally:
        if out_f is not None:
            out_f.close()


def evaluate_questions_recall(
    sources: Sequence[SourceSpec],
    questions_csv: str,
    groundtruth_csv: str,
    eval_target: str,
    model_name: str,
    top_k: int,
    per_index_k: int,
    normalize: bool,
    score_norm: str,
    ks: Sequence[int],
    allowed_sources: Optional[set[str]],
    id_col: str,
    question_col: str,
    limit: Optional[int],
    include_empty_gt: bool,
    details: bool,
    rerank: str,
    rerank_pool: int,
    rerank_alpha: float,
) -> None:
    from sentence_transformers import SentenceTransformer  # type: ignore

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("numpy is required") from e

    eval_target_norm = (eval_target or "").strip().lower()
    if eval_target_norm not in {"code", "docs"}:
        raise RuntimeError(f"Invalid eval_target: {eval_target!r}. Expected 'code' or 'docs'.")

    gt_entries = _load_groundtruth_entries(groundtruth_csv)

    model = SentenceTransformer(model_name)
    chunks_cache: Dict[str, List[Dict]] = {}
    index_cache: Dict[str, "object"] = {}

    totals: Dict[int, float] = {int(k): 0.0 for k in ks}
    n_total = 0
    n_evaluated = 0
    n_skipped_no_gt = 0
    n_skipped_out_of_target = 0

    for rid, qtext in _iter_questions_csv(
        questions_csv, id_col=id_col, question_col=question_col, limit=limit
    ):
        n_total += 1

        entry = gt_entries.get(rid)
        if entry is None:
            entry = GroundTruthEntry(answer_source="none", code_locations=[], doc_locations=[])

        if eval_target_norm == "code":
            if entry.answer_source not in {"code", "mixed"}:
                n_skipped_out_of_target += 1
                continue
            locs = entry.code_locations
        else:
            if entry.answer_source not in {"docs", "mixed"}:
                n_skipped_out_of_target += 1
                continue
            locs = entry.doc_locations

        if not locs and not include_empty_gt:
            n_skipped_no_gt += 1
            continue

        q = model.encode([qtext], normalize_embeddings=normalize)
        q = np.asarray(q, dtype=np.float32)

        results, _per_source = _select_top_results(
            sources=sources,
            q=q,
            query_text=qtext,
            top_k=top_k,
            per_index_k=per_index_k,
            score_norm=score_norm,
            rerank=rerank,
            rerank_pool=rerank_pool,
            rerank_alpha=rerank_alpha,
            chunks_cache=chunks_cache,
            index_cache=index_cache,
        )

        per = _evaluate_recall_at_k(
            results=results,
            gt_locs=locs,
            ks=ks,
            allowed_sources=allowed_sources,
        )

        if details:
            print(
                json.dumps(
                    {
                        "metric": "recall@k_per_query",
                        "id": rid,
                        "eval_target": eval_target_norm,
                        "answer_source": entry.answer_source,
                        "n_gt": len(locs),
                        "ks": [int(k) for k in ks],
                        "values": {str(k): per[int(k)] for k in ks},
                    },
                    ensure_ascii=False,
                )
            )

        for k, v in per.items():
            totals[int(k)] = totals.get(int(k), 0.0) + float(v)

        n_evaluated += 1

    if n_evaluated <= 0:
        print(
            "No queries evaluated. Either the CSV is empty or all rows were skipped (out-of-target or empty groundtruth for the selected eval target).",
            file=sys.stderr,
        )
        return

    print(
        json.dumps(
            {
                "metric": "recall@k",
                "eval_target": eval_target_norm,
                "n_total": n_total,
                "n_evaluated": n_evaluated,
                "n_skipped_no_gt": n_skipped_no_gt,
                "n_skipped_out_of_target": n_skipped_out_of_target,
                "ks": [int(k) for k in ks],
                "values": {str(k): (totals[int(k)] / float(n_evaluated)) for k in ks},
            },
            ensure_ascii=False,
        )
    )



def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        action="append",
        default=None,
        help="Repeatable. Format: label|index_path|chunks_jsonl",
    )
    p.add_argument("--query", default=None)
    p.add_argument(
        "--questions-csv",
        default=None,
        help="Run evaluation over a CSV of questions (e.g. test_questions.csv).",
    )
    p.add_argument(
        "--groundtruth-csv",
        default=None,
        help="Groundtruth CSV containing 'id' and locations. Supports columns: code_locations, doc_locations, answer_source.",
    )
    p.add_argument(
        "--eval-ks",
        default="1,3,5,10",
        help="Comma/space-separated ks for recall@k when --questions-csv is provided.",
    )
    p.add_argument(
        "--eval-target",
        choices=["code", "docs"],
        default="code",
        help="Which groundtruth field to evaluate: code uses code_locations; docs uses doc_locations. Also filters rows by answer_source (code/mixed vs docs/mixed).",
    )
    p.add_argument(
        "--eval-sources",
        default=None,
        help="Optional: only count hits from these source labels (comma/space-separated), e.g. 'code'.",
    )
    p.add_argument(
        "--eval-id-col",
        default="id",
        help="ID column name in --questions-csv (default: id).",
    )
    p.add_argument(
        "--eval-question-col",
        default="question",
        help="Question column name in --questions-csv (default: question).",
    )
    p.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional: limit number of rows read from --questions-csv.",
    )
    p.add_argument(
        "--eval-include-empty-gt",
        action="store_true",
        help="Include rows whose groundtruth has empty locations for the selected --eval-target (they contribute recall=0). Default: skip.",
    )
    p.add_argument(
        "--eval-details",
        action="store_true",
        help="Print per-query recall@k as JSONL before printing the final summary.",
    )
    p.add_argument(
        "--rerank",
        choices=["none", "bm25"],
        default="none",
        help="Optional reranking method applied to ANN candidates before truncating to top-k.",
    )
    p.add_argument(
        "--rerank-pool",
        type=int,
        default=100,
        help="How many top ANN candidates to rerank (from the merged list / each source list).",
    )
    p.add_argument(
        "--rerank-alpha",
        type=float,
        default=0.6,
        help="Combine score = alpha * vector + (1-alpha) * lexical. Only used for --rerank bm25.",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--per-index-k",
        type=int,
        default=None,
        help="How many results to fetch from each index before merging. Default: top_k*3",
    )
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--normalize", action="store_true")
    p.add_argument(
        "--score-norm",
        choices=["none", "zscore", "minmax"],
        default="none",
        help="Per-source score normalization before merging results. Sorting uses normalized scores.",
    )
    p.add_argument("--format", choices=["compact", "jsonl"], default="compact")
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--show-snippet", action="store_true")
    p.add_argument("--show-code", action="store_true", dest="show_snippet")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="After printing the top-k table (compact format), prompt to select a rank and print only that snippet.",
    )
    p.add_argument(
        "--pick-rank",
        default=None,
        help="Comma/space-separated ranks to expand after the table (e.g. '2' or '2,5,7'). Does not require stdin.",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help="Print separate top-k tables per source label instead of a merged ranking.",
    )
    p.add_argument(
        "--pick",
        default=None,
        help="In --split mode: expand selected ranks per source (e.g. 'code:1,2 docs:3' or 'docs:1').",
    )
    p.add_argument("--max-code-chars", type=int, default=1600)
    args = p.parse_args(argv)

    if not args.source:
        raise SystemExit("At least one --source is required")

    if not args.query and not args.questions_csv:
        raise SystemExit("Either --query (single query) or --questions-csv (batch evaluation) is required")

    try:
        sources = [_parse_source_spec(s) for s in args.source]
    except ValueError as e:
        raise SystemExit(str(e))

    per_index_k = args.per_index_k
    if per_index_k is None:
        per_index_k = max(int(args.top_k) * 3, int(args.top_k))

    rerank = str(args.rerank)
    if rerank != "none":
        per_index_k = max(int(per_index_k), int(args.rerank_pool), int(args.top_k))

    pick_ranks: Optional[List[int]] = None
    if args.pick_rank:
        raw = str(args.pick_rank)
        parts = [p for p in raw.replace(",", " ").split() if p.strip()]
        parsed: List[int] = []
        for part in parts:
            try:
                parsed.append(int(part))
            except ValueError:
                raise SystemExit(f"Invalid --pick-rank value: {part!r}. Expected integers like '2' or '2,5,7'.")
        pick_ranks = parsed

    pick: Optional[Dict[str, List[int]]] = None
    if args.pick:
        pick = {}
        raw = str(args.pick)
        tokens = [t for t in raw.replace(";", " ").split() if t.strip()]
        for tok in tokens:
            if ":" not in tok:
                raise SystemExit(
                    f"Invalid --pick token: {tok!r}. Expected format like 'code:1,2' or 'docs:3'."
                )
            label, rest = tok.split(":", 1)
            label = label.strip()
            if not label:
                raise SystemExit(f"Invalid --pick token: {tok!r}. Empty label.")
            parts = [p for p in rest.replace(",", " ").split() if p.strip()]
            if not parts:
                raise SystemExit(f"Invalid --pick token: {tok!r}. No ranks provided.")
            ranks: List[int] = []
            for part in parts:
                try:
                    ranks.append(int(part))
                except ValueError:
                    raise SystemExit(
                        f"Invalid --pick token: {tok!r}. Rank must be integer, got {part!r}."
                    )
            pick.setdefault(label, []).extend(ranks)

    if args.questions_csv:
        ks = _parse_ks(str(args.eval_ks))
        if not ks:
            raise SystemExit("--eval-ks did not contain any valid positive integers")

        top_k_eval = max(int(args.top_k), max(ks))
        per_index_k_eval = per_index_k
        if per_index_k_eval < top_k_eval:
            per_index_k_eval = top_k_eval

        allowed_sources: Optional[set[str]] = None
        if args.eval_sources:
            labels = [t for t in str(args.eval_sources).replace(",", " ").split() if t.strip()]
            if labels:
                allowed_sources = set(labels)

        if not args.groundtruth_csv:
            raise SystemExit("--groundtruth-csv is required when --questions-csv is provided")

        evaluate_questions_recall(
            sources=sources,
            questions_csv=str(args.questions_csv),
            groundtruth_csv=str(args.groundtruth_csv),
            eval_target=str(args.eval_target),
            model_name=args.model,
            top_k=top_k_eval,
            per_index_k=per_index_k_eval,
            normalize=args.normalize,
            score_norm=args.score_norm,
            ks=ks,
            allowed_sources=allowed_sources,
            id_col=str(args.eval_id_col),
            question_col=str(args.eval_question_col),
            limit=args.eval_limit,
            include_empty_gt=bool(args.eval_include_empty_gt),
            details=bool(args.eval_details),
            rerank=rerank,
            rerank_pool=int(args.rerank_pool),
            rerank_alpha=float(args.rerank_alpha),
        )
    else:
        query_multi(
            sources=sources,
            query=str(args.query),
            model_name=args.model,
            top_k=args.top_k,
            per_index_k=per_index_k,
            normalize=args.normalize,
            score_norm=args.score_norm,
            output_format=args.format,
            out_path=args.out,
            quiet=args.quiet,
            show_snippet=args.show_snippet,
            interactive=args.interactive,
            pick_ranks=pick_ranks,
            split=args.split,
            pick=pick,
            max_code_chars=args.max_code_chars,
            rerank=rerank,
            rerank_pool=int(args.rerank_pool),
            rerank_alpha=float(args.rerank_alpha),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

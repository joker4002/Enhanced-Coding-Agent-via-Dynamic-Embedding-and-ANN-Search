import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, TextIO, Tuple


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


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        action="append",
        default=None,
        help="Repeatable. Format: label|index_path|chunks_jsonl",
    )
    p.add_argument("--query", required=True)
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

    try:
        sources = [_parse_source_spec(s) for s in args.source]
    except ValueError as e:
        raise SystemExit(str(e))

    per_index_k = args.per_index_k
    if per_index_k is None:
        per_index_k = max(int(args.top_k) * 3, int(args.top_k))

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

    query_multi(
        sources=sources,
        query=args.query,
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
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

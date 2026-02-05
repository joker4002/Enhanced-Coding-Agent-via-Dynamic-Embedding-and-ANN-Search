import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Iterator, List, Optional, Sequence, Tuple


@dataclass
class DocChunk:
    chunk_id: str
    file_path: str
    symbol_type: str
    name: str
    qualname: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    code: str
    code_truncated: bool


def _iter_text_files(root: str, exts: Sequence[str]) -> Iterator[str]:
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "_build", "build", "dist", "__pycache__", ".venv", "venv", "node_modules"}]
        for fn in files:
            lower = fn.lower()
            if any(lower.endswith(ext) for ext in exts):
                yield os.path.join(dirpath, fn)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _is_heading_underline(s: str) -> bool:
    t = s.rstrip("\n")
    if len(t) < 3:
        return False
    ch = t[0]
    if ch not in "=-~^\"'`:+#.*_":
        return False
    return all(c == ch for c in t)


def _split_rst_sections(lines: List[str]) -> List[Tuple[str, int, int]]:
    sections: List[Tuple[str, int, int]] = []
    starts: List[Tuple[str, int]] = []

    i = 0
    while i + 1 < len(lines):
        title = lines[i].rstrip("\n")
        underline = lines[i + 1]
        if title.strip() and _is_heading_underline(underline):
            starts.append((title.strip(), i + 1))
            i += 2
            continue
        i += 1

    if not starts:
        return [("DOCUMENT", 1, len(lines))]

    first_title, first_ul = starts[0]
    if first_ul > 2:
        sections.append(("INTRO", 1, first_ul - 2))

    for idx, (title, underline_line_no) in enumerate(starts):
        start_line = underline_line_no + 1
        if idx + 1 < len(starts):
            next_title, next_ul = starts[idx + 1]
            end_line = max(next_ul - 2, start_line)
        else:
            end_line = len(lines)
        sections.append((title, start_line, end_line))

    return sections


def _get_segment(lines: List[str], start_line: int, end_line: int) -> str:
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    return "".join(lines[start_idx:end_idx]).strip("\n")


def _truncate(text: str, max_chars: int) -> Tuple[str, bool]:
    if max_chars <= 0:
        return text, False
    if len(text) <= max_chars:
        return text, False
    return text[: max_chars - 20] + "\n...<truncated>...\n", True


def extract_docs_chunks(
    docs_dir: str,
    exts: Sequence[str],
    limit: int,
    max_chars: int,
    path_contains: Optional[str],
) -> List[DocChunk]:
    chunks: List[DocChunk] = []

    for file_path in _iter_text_files(docs_dir, exts=exts):
        if path_contains and path_contains not in file_path:
            continue

        try:
            text = _read_text(file_path)
        except OSError:
            continue

        lines = text.splitlines(keepends=True)
        sections = _split_rst_sections(lines)
        rel = os.path.relpath(file_path, docs_dir)

        for title, start_line, end_line in sections:
            body = _get_segment(lines, start_line, end_line)
            if not body.strip():
                continue

            qualname = f"{rel}#{title}" if title else rel
            stable_id_src = f"{file_path}:{qualname}:{start_line}:{end_line}"
            chunk_id = hashlib.sha1(stable_id_src.encode("utf-8")).hexdigest()

            code, truncated = _truncate(body, max_chars=max_chars)

            chunks.append(
                DocChunk(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    symbol_type="doc",
                    name=title or os.path.basename(file_path),
                    qualname=qualname,
                    start_line=start_line,
                    end_line=end_line,
                    docstring=None,
                    code=code,
                    code_truncated=truncated,
                )
            )

    chunks.sort(key=lambda c: (c.file_path, c.start_line, c.qualname))
    return chunks[: max(limit, 0)]


def write_jsonl(chunks: List[DocChunk], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def write_csv(chunks: List[DocChunk], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fieldnames = list(asdict(chunks[0]).keys()) if chunks else [
        "chunk_id",
        "file_path",
        "symbol_type",
        "name",
        "qualname",
        "start_line",
        "end_line",
        "docstring",
        "code",
        "code_truncated",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in chunks:
            w.writerow(asdict(c))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default=os.path.join(os.getcwd(), "docs"))
    p.add_argument("--out", default=os.path.join(os.getcwd(), "docs_chunks.jsonl"))
    p.add_argument("--csv", default=None)
    p.add_argument("--limit", type=int, default=1000000)
    p.add_argument("--max-chars", type=int, default=8000)
    p.add_argument("--path-contains", default=None)
    p.add_argument("--ext", action="append", default=None)
    args = p.parse_args(argv)

    exts = tuple(args.ext) if args.ext else (".rst", ".md")

    chunks = extract_docs_chunks(
        docs_dir=args.docs,
        exts=exts,
        limit=args.limit,
        max_chars=args.max_chars,
        path_contains=args.path_contains,
    )

    write_jsonl(chunks, args.out)
    if args.csv:
        write_csv(chunks, args.csv)

    print(f"docs_chunks={len(chunks)}")
    print(f"jsonl={args.out}")
    if args.csv:
        print(f"csv={args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

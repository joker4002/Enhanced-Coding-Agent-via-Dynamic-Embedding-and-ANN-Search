import argparse
import ast
import csv
import hashlib
import json
import os
import sys
import tokenize
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, List, Optional, Tuple


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    symbol_type: str  # function | class | method
    name: str
    qualname: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    code: str
    code_truncated: bool


def _compute_end_lineno(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if isinstance(end, int) and end > 0:
        return end

    max_lineno = getattr(node, "lineno", 0) or 0
    for child in ast.walk(node):
        lineno = getattr(child, "lineno", None)
        if isinstance(lineno, int) and lineno > max_lineno:
            max_lineno = lineno
        child_end = getattr(child, "end_lineno", None)
        if isinstance(child_end, int) and child_end > max_lineno:
            max_lineno = child_end
    return max_lineno


def _read_text(path: str) -> str:
    with tokenize.open(path) as f:
        return f.read()


def _get_code_segment(source: str, start_line: int, end_line: int) -> str:
    lines = source.splitlines(keepends=True)
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    return "".join(lines[start_idx:end_idx]).rstrip("\n")


def _truncate_code(code: str, max_chars: int) -> Tuple[str, bool]:
    if max_chars <= 0:
        return code, False
    if len(code) <= max_chars:
        return code, False
    return code[: max_chars - 20] + "\n...<truncated>...\n", True


def _iter_py_files(src_dir: str) -> Iterator[str]:
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "venv", "node_modules"}]
        for fn in files:
            if fn.endswith(".py"):
                yield os.path.join(root, fn)


def _iter_definitions(
    file_path: str,
    source: str,
    include_methods: bool,
) -> Iterator[Tuple[str, str, str, ast.AST]]:
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return

    def visit(node: ast.AST, parents: List[str]) -> Iterable[Tuple[str, str, str, ast.AST]]:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                qual = ".".join(parents + [child.name])
                yield ("class", child.name, qual, child)
                yield from visit(child, parents + [child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_method = len(parents) > 0
                if is_method and not include_methods:
                    continue
                symbol_type = "method" if is_method else "function"
                qual = ".".join(parents + [child.name])
                yield (symbol_type, child.name, qual, child)

    yield from visit(tree, [])


def extract_chunks(
    src_dir: str,
    limit: int,
    include_methods: bool,
    max_chars: int,
    path_contains: Optional[str],
) -> List[CodeChunk]:
    chunks: List[CodeChunk] = []

    for file_path in _iter_py_files(src_dir):
        if path_contains and path_contains not in file_path:
            continue

        try:
            source = _read_text(file_path)
        except (OSError, UnicodeDecodeError):
            continue

        for symbol_type, name, qualname, node in _iter_definitions(file_path, source, include_methods):
            start_line = getattr(node, "lineno", None)
            if not isinstance(start_line, int) or start_line <= 0:
                continue

            end_line = _compute_end_lineno(node)
            if end_line <= 0:
                continue

            code = _get_code_segment(source, start_line, end_line)
            code, truncated = _truncate_code(code, max_chars=max_chars)

            doc = ast.get_docstring(node) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else None

            stable_id_src = f"{file_path}:{qualname}:{start_line}:{end_line}"
            chunk_id = hashlib.sha1(stable_id_src.encode("utf-8")).hexdigest()

            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    symbol_type=symbol_type,
                    name=name,
                    qualname=qualname,
                    start_line=start_line,
                    end_line=end_line,
                    docstring=doc,
                    code=code,
                    code_truncated=truncated,
                )
            )

    chunks.sort(key=lambda c: (c.file_path, c.start_line, c.qualname))
    return chunks[: max(limit, 0)]


def write_jsonl(chunks: List[CodeChunk], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def write_csv(chunks: List[CodeChunk], out_path: str) -> None:
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
    p.add_argument("--src", default=os.path.join(os.getcwd(), "src"))
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", default=os.path.join(os.getcwd(), "chunks.jsonl"))
    p.add_argument("--csv", default=None)
    p.add_argument("--include-methods", action="store_true")
    p.add_argument("--max-chars", type=int, default=8000)
    p.add_argument("--path-contains", default=None)
    args = p.parse_args(argv)

    chunks = extract_chunks(
        src_dir=args.src,
        limit=args.limit,
        include_methods=args.include_methods,
        max_chars=args.max_chars,
        path_contains=args.path_contains,
    )

    write_jsonl(chunks, args.out)
    if args.csv:
        write_csv(chunks, args.csv)

    print(f"extracted_chunks={len(chunks)}")
    print(f"jsonl={args.out}")
    if args.csv:
        print(f"csv={args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

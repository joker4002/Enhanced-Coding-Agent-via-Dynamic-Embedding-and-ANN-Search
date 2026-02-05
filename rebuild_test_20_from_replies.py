import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _split_indexed_items(text: str) -> List[Tuple[int, str]]:
    parts = re.split(r"\[(\d+)\]\s*", text or "")
    out: List[Tuple[int, str]] = []
    for i in range(1, len(parts), 2):
        idx_s = parts[i]
        body = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        if not body:
            continue
        try:
            idx = int(idx_s)
        except ValueError:
            continue
        out.append((idx, body))
    return out


_NEG_PHRASES = [
    "can't reproduce",
    "cannot reproduce",
    "show the full",
    "need more info",
    "provide a minimum",
    "could you elaborate",
    "what exactly is your problem",
    "please provide",
    "not sure",
    "i don't understand",
]

_POS_PHRASES = [
    "use ",
    "set ",
    "return ",
    "methods=",
    "methods ",
    "add ",
    "pass ",
    "guard ",
    "fix ",
    "solution",
    "you need",
    "you should",
    "the correct",
    "works if",
    "because you",
    "override",
    "disable",
    "remove",
    "delete",
    "re-raise",
]


def _score_solution_text(t: str) -> int:
    low = (t or "").lower()
    if any(p in low for p in _NEG_PHRASES):
        return -5
    score = 0
    for p in _POS_PHRASES:
        if p in low:
            score += 2
    if "```" in t or "def " in t or "@app." in t or "if __name__" in t:
        score += 3
    if "http" in low:
        score += 1
    if ":" in t[:40]:
        score += 1
    return score


def _extract_issue_number_from_issue_url(url: str) -> Optional[int]:
    m = re.search(r"/issues/(\d+)", url or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _cn_discussion_question(title: str, text: str) -> str:
    t = (title or "").strip()
    if t:
        return f"{t}：原因是什么？如何解决？"
    x = (text or "").strip().replace("\n", " ")
    x = re.sub(r"\s+", " ", x)
    if len(x) > 80:
        x = x[:80] + "…"
    return f"{x} 问题原因是什么？如何解决？"


def _cn_issue_question_from_title(title: str) -> str:
    t = (title or "").strip()
    if not t:
        return "这个 Flask Issue 的问题是什么？如何解决？"
    return f"{t}：这是什么原因？如何修复/规避？"


@dataclass
class Solution:
    solution_url: str
    solution_text: str


def _pick_best_solution(items: Sequence[Tuple[str, str]]) -> Optional[Solution]:
    best: Optional[Solution] = None
    best_score = -10**9
    for url, text in items:
        s = _score_solution_text(text)
        if s > best_score:
            best_score = s
            best = Solution(solution_url=url, solution_text=text)
    return best


def _load_issue_solutions(comment_csv: Path, comment_jsonl: Path) -> Dict[int, List[Tuple[str, str]]]:
    by_issue: Dict[int, List[Tuple[str, str]]] = {}

    for row in _read_csv(comment_csv):
        issue_no = _extract_issue_number_from_issue_url(row.get("issue_url", ""))
        if issue_no is None:
            continue
        by_issue.setdefault(issue_no, []).append((row.get("comment_url", ""), row.get("text", "")))

    for rec in _iter_jsonl(comment_jsonl):
        issue_no = rec.get("issue_number")
        if not isinstance(issue_no, int):
            continue
        by_issue.setdefault(issue_no, []).append((rec.get("comment_url", ""), rec.get("text", "")))

    return by_issue


def _load_discussion_candidates(disc_csv: Path) -> List[Dict[str, str]]:
    rows = _read_csv(disc_csv)
    out: List[Dict[str, str]] = []
    for r in rows:
        if (r.get("category") or "").strip() != "Q&A":
            continue
        replies_text = r.get("replies_text") or ""
        replies = _split_indexed_items(replies_text)
        if not replies:
            continue
        best = max(replies, key=lambda it: _score_solution_text(it[1]))
        best_score = _score_solution_text(best[1])
        if best_score < 3:
            continue
        r2 = dict(r)
        r2["best_reply_index"] = str(best[0])
        r2["best_reply_text"] = best[1]
        r2["best_reply_score"] = str(best_score)
        out.append(r2)

    out.sort(key=lambda x: (int(x.get("best_reply_score", "0")), int(x.get("number", "0"))), reverse=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-in", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--comment-csv", required=True)
    ap.add_argument("--comment-jsonl", required=True)
    ap.add_argument("--discussions", required=True)
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--solutions-out", required=True)
    args = ap.parse_args()

    test_in = Path(args.test_in)
    questions_csv = Path(args.questions)
    comment_csv = Path(args.comment_csv)
    comment_jsonl = Path(args.comment_jsonl)
    discussions_csv = Path(args.discussions)
    test_out = Path(args.test_out)
    solutions_out = Path(args.solutions_out)

    test_rows = _read_csv(test_in)
    issue_solutions = _load_issue_solutions(comment_csv, comment_jsonl)

    candidates = _load_discussion_candidates(discussions_csv)
    if len(candidates) < 20:
        raise SystemExit(f"Not enough discussion candidates with solutions: {len(candidates)}")
    chosen_disc = candidates[:20]

    kept: List[Dict[str, str]] = []
    removed_issue_rows: List[Dict[str, str]] = []

    for r in test_rows:
        src = (r.get("source") or "").strip()
        if src == "discussion_questions.csv":
            continue
        if src == "questions.csv":
            try:
                issue_no = int(r.get("issue_number") or "0")
            except ValueError:
                issue_no = 0
            if issue_no not in issue_solutions:
                removed_issue_rows.append(r)
                continue
        kept.append(r)

    needed = len(removed_issue_rows)

    existing_issue_numbers = set()
    for r in kept:
        if (r.get("repo") or "").strip() != "pallets/flask":
            continue
        try:
            existing_issue_numbers.add(int(r.get("issue_number") or "0"))
        except ValueError:
            pass

    questions_rows = _read_csv(questions_csv)
    question_by_issue: Dict[int, Dict[str, str]] = {}
    for qr in questions_rows:
        try:
            no = int(qr.get("issue_number") or "0")
        except ValueError:
            continue
        question_by_issue[no] = qr

    fill_rows: List[Dict[str, str]] = []
    if needed > 0:
        for issue_no in sorted(issue_solutions.keys(), reverse=True):
            if issue_no in existing_issue_numbers:
                continue
            qrec = question_by_issue.get(issue_no)
            if not qrec:
                continue
            fill_rows.append(
                {
                    "source": "questions.csv",
                    "repo": qrec.get("repo", "pallets/flask"),
                    "issue_number": str(issue_no),
                    "url": qrec.get("url", ""),
                    "question": _cn_issue_question_from_title(qrec.get("title", "")),
                }
            )
            existing_issue_numbers.add(issue_no)
            if len(fill_rows) >= needed:
                break

    if len(fill_rows) < needed:
        raise SystemExit(f"Not enough issue rows with local comment solutions to fill gaps. needed={needed} got={len(fill_rows)}")

    discussion_rows_out: List[Dict[str, str]] = []
    for d in chosen_disc:
        discussion_rows_out.append(
            {
                "source": "discussions_with_replies.csv",
                "repo": d.get("repo", "pallets/flask"),
                "issue_number": d.get("number", ""),
                "url": d.get("url", ""),
                "question": _cn_discussion_question(d.get("question_title", ""), d.get("question_text", "")),
            }
        )

    final_rows = kept + fill_rows + discussion_rows_out

    if len(final_rows) != len(test_rows):
        raise SystemExit(f"Row count changed: in={len(test_rows)} out={len(final_rows)}")

    _write_csv(test_out, final_rows, fieldnames=["source", "repo", "issue_number", "url", "question"])

    solutions: List[Dict[str, str]] = []

    for r in final_rows:
        src = (r.get("source") or "").strip()
        repo = r.get("repo", "")
        no = r.get("issue_number", "")
        url = r.get("url", "")
        question = r.get("question", "")

        if src == "discussions_with_replies.csv":
            dmatch = next((d for d in chosen_disc if d.get("number") == no), None)
            if dmatch:
                reply_idx = dmatch.get("best_reply_index", "")
                sol_url = url
                sol_text = dmatch.get("best_reply_text", "")
                solutions.append(
                    {
                        "type": "discussion",
                        "repo": repo,
                        "number": no,
                        "question_url": url,
                        "question": question,
                        "solution_url": sol_url,
                        "solution_reply_index": reply_idx,
                        "solution_text": sol_text,
                    }
                )
            continue

        try:
            issue_no = int(no)
        except ValueError:
            continue

        if src in {"comment_questions.csv", "comment_questions.jsonl"}:
            solutions.append(
                {
                    "type": "issue",
                    "repo": repo,
                    "number": no,
                    "question_url": url,
                    "question": question,
                    "solution_url": url,
                    "solution_reply_index": "",
                    "solution_text": question,
                }
            )
            continue

        if src == "questions.csv":
            best = _pick_best_solution(issue_solutions.get(issue_no, []))
            if best:
                solutions.append(
                    {
                        "type": "issue",
                        "repo": repo,
                        "number": no,
                        "question_url": url,
                        "question": question,
                        "solution_url": best.solution_url,
                        "solution_reply_index": "",
                        "solution_text": best.solution_text,
                    }
                )

    _write_csv(
        solutions_out,
        solutions,
        fieldnames=[
            "type",
            "repo",
            "number",
            "question_url",
            "question",
            "solution_url",
            "solution_reply_index",
            "solution_text",
        ],
    )


if __name__ == "__main__":
    main()

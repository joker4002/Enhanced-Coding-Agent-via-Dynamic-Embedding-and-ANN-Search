import csv
import json
import subprocess
from pathlib import Path

EVAL_CSV = "eval_from_discussions.csv"

# 你们的 sources：按你现有文件来（需要哪个就留哪个）
SOURCES = [
    "flask|flask.index|flask_chunks.jsonl",
    "docs|docs.index|docs_chunks.jsonl",
]

TOPKS = [1, 5, 10]

def run_query(query: str, top_k: int) -> list[dict]:
    """Call ann_query_multi.py and return parsed jsonl results."""
    tmp_out = Path(f"_tmp_results_{top_k}.jsonl")
    if tmp_out.exists():
        tmp_out.unlink()

    cmd = ["python", "ann_query_multi.py"]
    for s in SOURCES:
        cmd += ["--source", s]
    cmd += ["--query", query, "--top-k", str(top_k), "--format", "jsonl", "--out", str(tmp_out), "--quiet"]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        raise RuntimeError(f"Query failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

    results = []
    with tmp_out.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results

def hit_any_gold(result: dict, gold_terms: list[str]) -> bool:
    """Check if result matches any gold term by qualname/location."""
    qual = str(result.get("qualname", "")).lower()
    loc = str(result.get("location", "")).lower()
    for t in gold_terms:
        t = t.strip().lower()
        if not t:
            continue
        if t in qual or t in loc:
            return True
    return False

def main():
    total = 0
    hits = {k: 0 for k in TOPKS}

    with open(EVAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = (row.get("query") or "").strip()
            gold_terms = (row.get("gold_terms") or "").split("|")

            if not query:
                continue

            total += 1

            # 跑一次 top10，然后复用给 1/5/10（省时间）
            results10 = run_query(query, top_k=10)

            for k in TOPKS:
                topk = results10[:k]
                ok = any(hit_any_gold(r, gold_terms) for r in topk)
                if ok:
                    hits[k] += 1

            if total % 5 == 0:
                print(f"Processed {total} queries...")

    print("\n=== Recall Results ===")
    for k in TOPKS:
        r = hits[k] / total if total else 0
        print(f"Recall@{k}: {hits[k]}/{total} = {r:.3f}")

if __name__ == "__main__":
    main()
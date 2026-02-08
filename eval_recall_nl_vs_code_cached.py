import csv
import json
import subprocess
from pathlib import Path

EVAL_CSV = "eval_from_discussions.csv"
TOPKS = [1, 5, 10]

SOURCES = [
    "flask|flask.index|flask_chunks.jsonl",
    "docs|docs.index|docs_chunks.jsonl",
]

CACHE_NL = Path("cache_nl_top10.jsonl")
CACHE_CODE = Path("cache_code_top10.jsonl")


def run_query_to_file(query: str, out_path: Path):
    cmd = ["python", "ann_query_multi.py"]
    for s in SOURCES:
        cmd += ["--source", s]
    cmd += ["--query", query, "--top-k", "10", "--format", "jsonl", "--out", str(out_path), "--quiet"]

    # 关键：不要 capture_output（减少编码/缓冲问题）
    subprocess.run(cmd)


def hit_any_gold(result: dict, gold_terms: list[str]) -> bool:
    qual = str(result.get("qualname", "")).lower()
    loc = str(result.get("location", "")).lower()
    for t in gold_terms:
        t = t.strip().lower()
        if t and (t in qual or t in loc):
            return True
    return False


def pick_code_query(gold_terms: list[str]) -> str:
    gold_terms = [t.strip() for t in gold_terms if t.strip()]
    for t in gold_terms:
        if "." in t and 3 <= len(t) <= 80:
            return t
    for t in gold_terms:
        if "_" in t and t.islower() and 3 <= len(t) <= 40:
            return t
    return gold_terms[0] if gold_terms else ""


def build_cache(cache_path: Path, mode: str):
    # mode: "nl" or "code"
    if cache_path.exists():
        cache_path.unlink()

    with open(EVAL_CSV, "r", encoding="utf-8") as f, cache_path.open("w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        total = 0

        for row in reader:
            query = (row.get("query") or "").strip()
            gold_terms = (row.get("gold_terms") or "").split("|")

            if not query:
                continue

            if mode == "nl":
                q = query
            else:
                q = pick_code_query(gold_terms)
                if not q:
                    continue

            tmp = Path("_tmp_top10.jsonl")
            if tmp.exists():
                tmp.unlink()

            run_query_to_file(q, tmp)

            # 把这一条的 top10 + 原始 query id 写进 cache
            results = []
            with tmp.open("r", encoding="utf-8") as r:
                for line in r:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))

            record = {
                "query": query,
                "mode": mode,
                "used_query": q,
                "gold_terms": gold_terms,
                "results": results,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            total += 1
            if total % 5 == 0:
                print(f"[{mode}] cached {total} queries...")


def compute_recall(cache_path: Path):
    hits = {k: 0 for k in TOPKS}
    total = 0

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gold_terms = rec["gold_terms"]
            results = rec["results"]

            total += 1
            for k in TOPKS:
                if any(hit_any_gold(r, gold_terms) for r in results[:k]):
                    hits[k] += 1

    return total, hits


def main():
    print("Building NL cache...")
    build_cache(CACHE_NL, "nl")

    print("Building CODE cache...")
    build_cache(CACHE_CODE, "code")

    total_nl, hits_nl = compute_recall(CACHE_NL)
    total_code, hits_code = compute_recall(CACHE_CODE)

    print("\n=== NL Query Recall ===")
    for k in TOPKS:
        print(f"Recall@{k}: {hits_nl[k]}/{total_nl} = {hits_nl[k]/total_nl:.3f}")

    print("\n=== Code Query Recall ===")
    for k in TOPKS:
        print(f"Recall@{k}: {hits_code[k]}/{total_code} = {hits_code[k]/total_code:.3f}")


if __name__ == "__main__":
    main()
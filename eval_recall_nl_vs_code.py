import csv
import json
import subprocess
from pathlib import Path

EVAL_CSV = "eval_from_discussions.csv"

SOURCES = [
    "flask|flask.index|flask_chunks.jsonl",
    "docs|docs.index|docs_chunks.jsonl",
]

TOPKS = [1, 5, 10]


def run_query(query: str, top_k: int) -> list[dict]:
    tmp_out = Path(f"_tmp_results_{top_k}.jsonl")
    if tmp_out.exists():
        tmp_out.unlink()

    cmd = ["python", "ann_query_multi.py"]
    for s in SOURCES:
        cmd += ["--source", s]

    cmd += [
        "--query", query,
        "--top-k", str(top_k),
        "--format", "jsonl",
        "--out", str(tmp_out),
        "--quiet",
    ]

    subprocess.run(cmd)

    results = []
    with tmp_out.open("r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def hit_any_gold(result: dict, gold_terms: list[str]) -> bool:
    qual = str(result.get("qualname", "")).lower()
    loc = str(result.get("location", "")).lower()

    for t in gold_terms:
        t = t.strip().lower()
        if t and (t in qual or t in loc):
            return True
    return False


def pick_code_query(gold_terms):
    for t in gold_terms:
        if "." in t:
            return t
    for t in gold_terms:
        if "_" in t:
            return t
    return gold_terms[0] if gold_terms else ""


def main():

    hits_nl = {k: 0 for k in TOPKS}
    hits_code = {k: 0 for k in TOPKS}

    total = 0

    with open(EVAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            query = row["query"]
            gold_terms = row["gold_terms"].split("|")

            code_query = pick_code_query(gold_terms)

            res_nl = run_query(query, 10)
            res_code = run_query(code_query, 10) if code_query else []

            for k in TOPKS:
                if any(hit_any_gold(r, gold_terms) for r in res_nl[:k]):
                    hits_nl[k] += 1

                if res_code and any(hit_any_gold(r, gold_terms) for r in res_code[:k]):
                    hits_code[k] += 1

            total += 1
            if total % 5 == 0:
                print(f"Processed {total} queries...")

    print("\n=== NL Query Recall ===")
    for k in TOPKS:
        print(f"Recall@{k}: {hits_nl[k]}/{total} = {hits_nl[k]/total:.3f}")

    print("\n=== Code Query Recall ===")
    for k in TOPKS:
        print(f"Recall@{k}: {hits_code[k]}/{total} = {hits_code[k]/total:.3f}")


if __name__ == "__main__":
    main()
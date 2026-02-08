import json
import pandas as pd

IN_PATH = "cache_nl_top10.jsonl"
OUT_PATH = "cache_nl_top10_readable.csv"

rows = []
with open(IN_PATH, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        # 兼容不同字段命名：query / question / text
        query = obj.get("query") or obj.get("question") or obj.get("text") or ""

        # 兼容不同字段命名：results / hits / topk
        hits = obj.get("results") or obj.get("hits") or obj.get("topk") or []

        for rank, h in enumerate(hits, start=1):
            rows.append({
                "line_no": line_no,
                "query": query,
                "rank": rank,
                "score": h.get("score"),
                "raw_score": h.get("raw_score"),
                "source": h.get("source"),
                "qualname": h.get("qualname"),
                "location": h.get("location"),
                "file_path": h.get("file_path"),
                "chunk_id": h.get("chunk_id"),
                "text": h.get("text") or h.get("chunk_text") or "",
            })

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print("Wrote:", OUT_PATH, "rows=", len(df))
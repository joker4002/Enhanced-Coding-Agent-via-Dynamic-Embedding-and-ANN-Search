import pandas as pd
import re

inp = "Ground_choice_discussion_replies.csv"
out = "eval_from_discussions.csv"

df = pd.read_csv(inp)

def extract_gold_terms(reply: str) -> str:
    if not isinstance(reply, str):
        return ""
    # 1) 抓反引号里的内容：`send_file`、`Blueprint.add_url_rule`
    backticks = re.findall(r"`([^`]{2,80})`", reply)

    # 2) 抓类似 Flask.url_for / Blueprint.add_url_rule 这种点号路径
    dotted = re.findall(r"\b[A-Z][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b", reply)

    # 3) 抓 snake_case 函数名：send_file、send_from_directory
    snake = re.findall(r"\b[a-z_]{3,40}\b", reply)

    # 过滤一些太泛的词（你也可以再加）
    stop = {"this","that","with","from","your","have","will","when","then","just","also","there","here"}
    terms = []
    for t in backticks + dotted + snake:
        t = t.strip()
        if len(t) < 3:
            continue
        if t.lower() in stop:
            continue
        terms.append(t)

    # 去重，保留前 20 个（够用了）
    seen = set()
    uniq = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(t)
        if len(uniq) >= 20:
            break
    return "|".join(uniq)

out_df = pd.DataFrame({
    "repo": df["repo"],
    "number": df["number"],
    "url": df["url"],
    "query": df["question_text"],
    "gold_terms": df["replies_text"].apply(extract_gold_terms),
})
out_df.to_csv(out, index=False, encoding="utf-8")
print("Wrote:", out)
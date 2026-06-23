"""

Collect r/movies posts from HuggingFace Reddit dataset (no API credentials needed).
Filters by keyword groups, saves to raw_posts.csv.

Usage:
    python collect_data.py
"""

import csv
import hashlib
from datasets import load_dataset

REACTION_KEYWORDS = [
    "just saw", "just watched", "trailer", "opening weekend",
    "just finished", "first time", "box office", "casting",
]
HOT_TAKE_KEYWORDS = [
    "overrated", "underrated", "unpopular opinion", "change my mind",
    "hot take", "fight me", "most people", "disagree",
    "actually bad", "actually good", "is trash", "is terrible",
]
ANALYSIS_KEYWORDS = [
    "cinematography", "narrative", "thematic", "screenplay",
    "character arc", "motif", "symbolism", "subtext",
    "mise en scène", "editing", "visual language", "allegory",
    "structure", "directorial", "score",
]

MIN_CHARS = 100
TARGET_PER_LABEL = {"reaction": 80, "hot_take": 70, "analysis": 60}


def classify_hint(text: str) -> str | None:
    low = text.lower()
    if any(k in low for k in ANALYSIS_KEYWORDS):
        return "analysis"
    if any(k in low for k in HOT_TAKE_KEYWORDS):
        return "hot_take"
    if any(k in low for k in REACTION_KEYWORDS):
        return "reaction"
    return None


def make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:10]


def main():
    print("Loading Reddit dataset from HuggingFace (streaming)…")
    print("This may take a minute on first run.\n")

    # sentence-transformers/reddit-title-body: Parquet format, has subreddit field
    ds = load_dataset(
        "sentence-transformers/reddit-title-body",
        split="train",
        streaming=True,
    )

    buckets: dict[str, list[dict]] = {k: [] for k in TARGET_PER_LABEL}
    seen: set[str] = set()
    checked = 0

    for item in ds:
        if all(len(buckets[k]) >= TARGET_PER_LABEL[k] for k in TARGET_PER_LABEL):
            break

        # filter to r/movies
        subreddit = (item.get("subreddit") or "").lower()
        if subreddit != "movies":
            continue

        body  = (item.get("body")  or "").strip()
        title = (item.get("title") or "").strip()
        text = f"{title}\n\n{body}".strip() if body else title

        if len(text) < MIN_CHARS:
            continue

        hint = classify_hint(text)
        if hint and len(buckets[hint]) < TARGET_PER_LABEL[hint]:
            pid = make_id(text)
            if pid in seen:
                continue
            seen.add(pid)
            buckets[hint].append({
                "id":         pid,
                "text":       text,
                "label_hint": hint,
                "label":      "",
                "prelabeled": "false",
                "hard_case":  "false",
                "notes":      "",
            })

        checked += 1
        if checked % 5000 == 0:
            totals = {k: len(v) for k, v in buckets.items()}
            print(f"  Scanned {checked:,} posts — {totals}")

    rows = [row for label in TARGET_PER_LABEL for row in buckets[label]]

    out_path = "raw_posts.csv"
    fieldnames = ["id", "text", "label_hint", "label", "prelabeled", "hard_case", "notes"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} posts → {out_path}")
    for label in TARGET_PER_LABEL:
        count = sum(1 for r in rows if r["label_hint"] == label)
        print(f"  {label}: {count}")

    if len(rows) < 200:
        print("\n[!] 不够 200 条，可能该数据集里 r/movies 帖子较少。")
        print("    建议运行后手动从 Reddit 网页补充剩余条目到 raw_posts.csv。")


if __name__ == "__main__":
    main()

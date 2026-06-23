"""
Pre-label raw_posts.csv using local Ollama (llama3.1).
No API key, no token limits, incremental save — safe to re-run after crashes.

Requirements:
    ollama serve   (running in a separate terminal)

Usage:
    python prelabel.py
"""

import csv
import json
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "llama3.1:latest"

SYSTEM_PROMPT = """Classify r/movies Reddit posts into exactly one label.

analysis: structured argument with specific verifiable evidence (cinematography, narrative, thematic content, directorial choices, comparisons to other films). The reasoning would hold up even without the opinion framing.

hot_take: bold opinion asserted without real supporting evidence. The claim may be valid but the post asserts rather than argues. Any evidence is decorative, not load-bearing.

reaction: immediate emotional or hype-driven response to a specific event (trailer drop, just saw a film, box office result, casting announcement). Expresses a feeling, not an argument.

Edge case rules:
- Names craft elements but only asserts, no evidence → hot_take
- "just saw / just watched" + personal feeling, no generalizable claim → reaction

Reply with ONLY one word: analysis, hot_take, or reaction. Nothing else."""


def classify(text: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text[:500]},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            raw  = data["message"]["content"].strip().lower()
            raw  = raw.split()[0] if raw else ""
            return raw if raw in {"analysis", "hot_take", "reaction"} else "REVIEW"
    except Exception as e:
        print(f"  [error] {e}")
        return "REVIEW"


def main():
    in_path  = "raw_posts.csv"
    out_path = "prelabeled_posts.csv"

    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # resume: skip already-labeled rows
    existing: dict[str, dict] = {}
    if Path(out_path).exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("label"):
                    existing[r["id"]] = r

    todo = [r for r in rows if r["id"] not in existing]
    print(f"Total: {len(rows)} | Already labeled: {len(existing)} | Remaining: {len(todo)}\n")

    fieldnames = ["id", "text", "label_hint", "label", "prelabeled", "hard_case", "notes"]
    done = dict(existing)

    for i, row in enumerate(todo, 1):
        label = classify(row["text"])
        row["label"]      = label
        row["prelabeled"] = "true"
        done[row["id"]]   = row

        # incremental save after every label
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(done.values())

        if i % 10 == 0 or i == len(todo):
            print(f"  {len(done)}/{len(rows)} labeled")

    print(f"\nDone. Saved {len(done)} rows → {out_path}")

    from collections import Counter
    counts = Counter(r["label"] for r in done.values())
    print("\nLabel distribution (pre-review):")
    for label in ["reaction", "hot_take", "analysis", "REVIEW"]:
        n = counts.get(label, 0)
        if n:
            print(f"  {label}: {n} ({n/len(done)*100:.1f}%)")

    print("\nNEXT STEP: open prelabeled_posts.csv in a spreadsheet.")
    print("Review every row, correct wrong labels, mark hard cases,")
    print("then save as dataset.csv.")


if __name__ == "__main__":
    main()

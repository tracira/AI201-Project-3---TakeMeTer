"""
Re-classify the uncertain rows in dataset.csv using Ollama.
Writes the final dataset.csv with all rows labeled.

Usage:
    python fix_uncertain.py
"""

import csv, json, urllib.request
from pathlib import Path
from collections import Counter

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "llama3.1:latest"

SYSTEM_PROMPT = """You classify r/movies Reddit posts. Pick exactly one label.

REACTION — ONLY if the post is purely emotional / experiential with NO quality judgment:
  Asking for movie recommendations, sharing personal viewing experience with no claims,
  news/announcements without opinion, polls asking who people's favorites are.

HOT_TAKE — when the post makes any quality judgment or opinion claim:
  Calling something overrated/underrated/best/worst, saying a director is good or bad,
  asking "does anyone else think X is [judgment]?", expressing disappointment or praise
  AS A CLAIM, comparing two things and picking a winner, saying "you must see this."
  Also: discussion threads asking what people think of X, "agree or disagree?" posts.

ANALYSIS — structured argument backed by specific craft evidence:
  References cinematography, narrative structure, thematic content, symbolism, editing,
  sound design, shot composition — AND uses these to build an actual argument.

Reply with ONE word only: reaction, hot_take, or analysis."""


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
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read())["message"]["content"].strip().lower()
            word = raw.split()[0] if raw else ""
            return word if word in {"analysis", "hot_take", "reaction"} else "hot_take"
    except Exception as e:
        print(f"  [error] {e}")
        return "hot_take"


def main():
    path = "dataset.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    uncertain = [r for r in rows if "NEEDS REVIEW" in r.get("notes", "")]
    certain   = [r for r in rows if "NEEDS REVIEW" not in r.get("notes", "")]

    print(f"Certain: {len(certain)} | Re-classifying uncertain: {len(uncertain)}\n")

    fieldnames = ["id", "text", "label_hint", "label", "prelabeled", "hard_case", "notes"]

    for i, row in enumerate(uncertain, 1):
        label = classify(row["text"])
        row["label"]     = label
        row["hard_case"] = "false"
        row["notes"]     = f"ollama-final (was uncertain)"

        all_rows = certain + uncertain
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        if i % 10 == 0 or i == len(uncertain):
            counts = Counter(r["label"] for r in all_rows)
            print(f"  {i}/{len(uncertain)} — "
                  f"reaction:{counts['reaction']} "
                  f"hot_take:{counts['hot_take']} "
                  f"analysis:{counts['analysis']}")

    all_rows = certain + uncertain
    counts = Counter(r["label"] for r in all_rows)
    total  = len(all_rows)
    print(f"\nFinal dataset.csv ({total} rows):")
    for label in ["reaction", "hot_take", "analysis"]:
        n = counts[label]
        flag = "  ← OVER 70% LIMIT" if n/total > 0.70 else ""
        print(f"  {label}: {n} ({n/total*100:.1f}%){flag}")
    print("\n✓ dataset.csv is ready for Colab training.")


if __name__ == "__main__":
    main()

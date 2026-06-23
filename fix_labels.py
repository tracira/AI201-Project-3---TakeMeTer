"""
Re-classify rows currently labeled 'reaction' using a sharper prompt.
analysis labels are kept as-is. Overwrites prelabeled_posts.csv.

Usage:
    python fix_labels.py
"""

import csv
import json
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "llama3.1:latest"

SYSTEM_PROMPT = """You classify r/movies Reddit posts. Read carefully and pick exactly one label.

REACTION — the post is PURELY an emotional response to a specific personal event.
  It only expresses a feeling: excitement, sadness, confusion, nostalgia.
  There is NO opinion claim about quality, ranking, or the filmmaker.
  Examples: "Just saw Inception, my mind is blown!", "Just finished Dear Zachary... wow."

HOT_TAKE — the post makes a JUDGMENT or OPINION CLAIM, even if framed as a reaction.
  Any of these → hot_take:
  • calls something overrated, underrated, best, worst, greatest, terrible, flawed
  • says the director is good/bad/genius/overrated
  • compares two films and picks a winner
  • argues the film deserved more/less praise or box office
  • asks "does anyone else think X is [judgment]?"
  • expresses disappointment or praise AS A CLAIM (not just a feeling)
  Examples: "Just saw Inception — Nolan is overrated, it's just a heist movie."
            "Am I the only one who thinks Crystal Skull wasn't that bad?"
            "Just saw Predators. That movie was beyond fucking horrible [lists reasons]."

ANALYSIS — structured argument with specific craft evidence (cinematography, narrative
  structure, thematic content, directorial choices). Evidence would support the claim
  even without the opinion framing.

The key test: does the post make a CLAIM about quality/ranking/the filmmaker?
  YES → hot_take (or analysis if evidence-backed)
  NO  → reaction

Reply with ONLY one word: reaction, hot_take, or analysis."""


def classify(text: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text[:600]},
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
            data = json.loads(resp.read())
            raw  = data["message"]["content"].strip().lower().split()[0]
            return raw if raw in {"analysis", "hot_take", "reaction"} else "REVIEW"
    except Exception as e:
        print(f"  [error] {e}")
        return "REVIEW"


def main():
    path = "prelabeled_posts.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    to_recheck = [r for r in rows if r["label"] == "reaction"]
    keep       = [r for r in rows if r["label"] != "reaction"]
    print(f"Keeping {len(keep)} non-reaction labels.")
    print(f"Re-classifying {len(to_recheck)} reaction-labeled posts…\n")

    fieldnames = ["id", "text", "label_hint", "label", "prelabeled", "hard_case", "notes"]

    for i, row in enumerate(to_recheck, 1):
        new_label = classify(row["text"])
        if new_label != "reaction":
            row["notes"] = f"reclassified reaction→{new_label}"
        row["label"] = new_label
        to_recheck[i - 1] = row

        # save after every label
        all_rows = keep + to_recheck
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        if i % 10 == 0 or i == len(to_recheck):
            from collections import Counter
            counts = Counter(r["label"] for r in all_rows)
            print(f"  {i}/{len(to_recheck)} done — "
                  f"reaction:{counts['reaction']} "
                  f"hot_take:{counts['hot_take']} "
                  f"analysis:{counts['analysis']}")

    all_rows = keep + to_recheck
    from collections import Counter
    counts = Counter(r["label"] for r in all_rows)
    total  = len(all_rows)
    print(f"\nFinal distribution ({total} rows):")
    for label in ["reaction", "hot_take", "analysis", "REVIEW"]:
        n = counts.get(label, 0)
        if n:
            flag = " ← OVER 70% LIMIT" if n / total > 0.70 else ""
            print(f"  {label}: {n} ({n/total*100:.1f}%){flag}")


if __name__ == "__main__":
    main()

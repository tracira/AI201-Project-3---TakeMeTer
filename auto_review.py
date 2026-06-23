"""
Auto-fix obvious mislabels and write dataset.csv.
Only prints rows that genuinely need human eyes (~20-30 rows).

Usage:
    python auto_review.py
"""

import csv, re
from pathlib import Path

JUDGMENT_WORDS = [
    "overrated", "underrated", "best movie", "worst movie", "greatest film",
    "best film", "best ever", "worst ever", "most underrated", "most overrated",
    "doesn't deserve", "does not deserve", "is terrible", "is awful",
    "is garbage", "is trash", "is brilliant", "is perfect", "is flawed",
    "seriously flawed", "piss anyone", "guilty pleasure", "does anyone else think",
    "am i the only one", "am i wrong", "am i alone", "dae ", "dea ",
    "anyone else feel", "anyone else think", "fight me", "change my mind",
    "unpopular opinion", "hot take", "controversial opinion",
]
ANALYSIS_TERMS = [
    "cinematography", "mise en scène", "narrative structure", "thematic",
    "screenplay", "character arc", "motif", "symbolism", "allegory",
    "subtext", "visual language", "visual metaphor", "directorial",
    "editing technique", "sound design", "color palette", "long take",
    "depth of field", "shot composition", "production design",
]
REACTION_STARTERS = [
    "just saw", "just watched", "just finished", "just view",
    "i just saw", "i just watched", "i just finished",
]
RECOMMENDATION_PATTERNS = [
    r"recommend.*movie", r"what.*should i (watch|see)",
    r"looking for (movie|film|recommendation)",
    r"can (you|anyone|someone) (name|help|identify|find|recommend)",
    r"what (movie|film|are some)",
    r"any (good |other )?(movie|film|recommendation)",
]


def analyze(text: str, current_label: str) -> tuple[str, str, bool]:
    """Return (new_label, reason, needs_human_review)."""
    low   = text.lower()
    nl    = low.find("\n")
    title = low[:nl].strip() if nl > 0 else low[:120]
    wc    = len(text.split())

    # ── strong ANALYSIS: 2+ craft terms ───────────────────────────
    hits = [t for t in ANALYSIS_TERMS if t in low]
    if len(hits) >= 2:
        return "analysis", f"craft terms: {hits[:2]}", False
    if len(hits) == 1 and wc >= 300:
        return "analysis", f"craft term + long ({wc}w)", False

    # ── strong REACTION: pure "just saw/watched" no judgment ──────
    is_reaction_opener = any(title.startswith(s) for s in REACTION_STARTERS) or \
                         any(s in low[:150] for s in REACTION_STARTERS)
    has_judgment = any(j in low for j in JUDGMENT_WORDS)

    if is_reaction_opener and not has_judgment:
        return "reaction", "reaction opener, no judgment", False

    # ── strong HOT_TAKE: judgment words present ────────────────────
    if has_judgment:
        matched = next(j for j in JUDGMENT_WORDS if j in low)
        return "hot_take", f"judgment word: '{matched}'", False

    # ── recommendation / question posts → reaction ─────────────────
    if any(re.search(p, low) for p in RECOMMENDATION_PATTERNS):
        return "reaction", "recommendation/question post", False

    # ── no clear signal: flag for human review ─────────────────────
    return current_label, "uncertain — please review", True


def main():
    in_path  = "prelabeled_posts.csv"
    out_path = "dataset.csv"

    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = ["id", "text", "label_hint", "label", "prelabeled", "hard_case", "notes"]
    results    = []
    review_rows = []

    for row in rows:
        label, reason, needs_review = analyze(row["text"], row["label"])
        row["label"]      = label
        row["prelabeled"] = "true"
        if needs_review:
            row["hard_case"] = "true"
            row["notes"]     = "NEEDS REVIEW — " + reason
            review_rows.append(row)
        else:
            row["notes"] = reason
        results.append(row)

    # write dataset.csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    from collections import Counter
    counts = Counter(r["label"] for r in results)
    total  = len(results)

    print(f"Saved {total} rows → {out_path}\n")
    print("Label distribution:")
    for label in ["reaction", "hot_take", "analysis"]:
        n = counts[label]
        flag = "  ← OVER 70% LIMIT" if n/total > 0.70 else ""
        print(f"  {label}: {n} ({n/total*100:.1f}%){flag}")

    print(f"\n{'='*55}")
    print(f"NEEDS HUMAN REVIEW ({len(review_rows)} rows):")
    print(f"{'='*55}")
    for r in review_rows:
        print(f"\n[hint={r['label_hint']} | auto={r['label']}]")
        print(r["text"][:220].strip())
        print("  → Your call: reaction / hot_take / analysis?")

    print(f"\n{'='*55}")
    print(f"Edit dataset.csv for the {len(review_rows)} rows above,")
    print("then you're done with annotation.")


if __name__ == "__main__":
    main()

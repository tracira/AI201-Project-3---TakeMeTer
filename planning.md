# Project Planning — r/movies Discourse Classifier

---

## 1. Community

**Subreddit:** r/movies

r/movies is one of Reddit's largest and most active film communities, with millions of members ranging from casual moviegoers to dedicated cinephiles. It is a strong fit for a classification task for three reasons:

1. **Discourse is genuinely varied.** A single day's front page mixes trailer reactions, controversial opinions, and detailed craft breakdowns — natural variation that makes labels meaningful rather than artificial.
2. **The distinctions are community-recognized.** r/movies regulars already argue about what counts as "real" analysis versus an unsubstantiated hot take. The label boundaries reflect how people in the community actually evaluate discourse quality, not categories imposed from outside.
3. **Volume is not a bottleneck.** The sub produces hundreds of posts daily, making it easy to collect 200+ labeled examples across all three categories without scraping edge cases.

The existence of r/TrueFilm — a stricter, analysis-only spinoff — is itself evidence that the `analysis` / `hot_take` distinction is real and meaningful to this community. The `reaction` label is equally community-grounded: r/movies regulars routinely complain about low-effort trailer hype posts clogging the front page, the sub has explicit rules limiting "just saw" posts to weekly megathreads, and top-level comments in any major release thread are dominated by immediate emotional responses that offer no argument. The community already treats these three modes of discourse differently — this taxonomy names what members already recognize.

---

## 2. Label Taxonomy

### `analysis`
**Definition:** The post makes a structured argument about a film using specific, verifiable evidence — cinematography, narrative structure, thematic content, directorial choices, or comparisons to other films — where the reasoning would hold up even if the opinion framing were removed.

**Examples:**
- *"Parasite's use of vertical space as a class metaphor is deliberate and consistent — the Parks live on a hill, the Kims in a basement, and every key turning point involves movement up or down a staircase."*
- *"The Godfather's slow pacing in Act 1 is structural, not incidental — it establishes the world's rules before breaking them, which is why the baptism sequence lands so hard."*

---

### `hot_take`
**Definition:** A bold, confident opinion stated without meaningful supporting reasoning — the claim may be valid, but the post asserts rather than argues, and any evidence present is decorative rather than load-bearing.

**Examples:**
- *"Inception is massively overrated. People act like it's the most complex film ever made but it's just a heist movie with a gimmick."*
- *"Denis Villeneuve is the most overrated director working today. Dune looks pretty but has nothing to say."*

---

### `reaction`
**Definition:** An immediate emotional or hype-driven response to a specific event — a trailer drop, casting announcement, box office result, or release — where little to no argument is present and the post is primarily expressing a feeling in the moment.

**Examples:**
- *"The new Alien: Romulus trailer just dropped and I am NOT okay. This looks incredible."*
- *"Just saw the opening weekend numbers for The Fall Guy — ouch. Hollywood really can't figure out original IP anymore."*

---

## 3. Hard Edge Cases

**Primary edge case — `analysis` vs. `hot_take`:**

> *"Nolan is overrated — his dialogue is always expository and his female characters are chronically underdeveloped."*

This post names specific craft elements (dialogue style, character writing), which sound like analysis. But the claims are asserted without evidence: no specific scenes, no comparisons, no structured reasoning.

**Decision rule:** If the post names specific, verifiable craft elements *and* uses them to build a real argument — i.e., the evidence would support the claim even without the opinion framing — label it `analysis`. If the craft details are decorative (present to sound credible but not forming a genuine argument), label it `hot_take`. The Nolan post is `hot_take`: the details are asserted, not demonstrated.

**Secondary edge case — `hot_take` vs. `reaction`:**

> *"Just saw Oppenheimer. Three hours and I still don't know what Nolan was trying to say."*

This is a post-viewing emotional response (→ `reaction`) but also reads as an implicit negative opinion (→ `hot_take`).

**Decision rule:** If the post is anchored to a specific, time-stamped personal event ("just saw", "just finished", "opening night") and expresses a feeling without making a generalizable claim, label it `reaction`. If the post steps back to make a broader assertion about the film or filmmaker, label it `hot_take`.

---

## 4. Data Collection Plan

**Source (actual):** Reddit posts from r/movies via the `sentence-transformers/reddit-title-body` dataset on HuggingFace (streamed, no API credentials required). Reddit's public JSON and RSS APIs were fully blocked for anonymous access as of 2023; PRAW script-type app creation was also unavailable at time of collection. HuggingFace was used as an equivalent public source of authentic r/movies posts.

**Actual distribution collected:**

| Label | Count | % |
|---|---|---|
| `reaction` | 74 | 35.2% |
| `hot_take` | 69 | 32.9% |
| `analysis` | 66 | 31.4% |
| **Total** | **209** | |

No label exceeds 70%; all labels are within the 20–45% range.

**Train / validation / test split:** 140 / 30 / 30 (70% / 15% / 15%), stratified. The test set is held out completely until final evaluation.

**Collection strategy (actual):** Posts were streamed from the HuggingFace dataset, filtered to `subreddit == "movies"`, and routed into three keyword buckets: reaction (e.g. "just saw", "trailer", "opening weekend"), hot_take (e.g. "overrated", "unpopular opinion", "change my mind"), and analysis (e.g. "cinematography", "narrative", "thematic"). Posts under 100 characters were excluded.

**Labeling process (actual):**
1. All 209 posts were pre-labeled by a local Ollama instance (llama3.1:latest) using the label definitions and decision rules from Sections 2–3 as the system prompt.
2. A second Ollama pass with a sharpened `reaction` vs. `hot_take` decision prompt was applied to correct systematic over-prediction of `reaction`.
3. A rule-based post-processor applied deterministic overrides for high-confidence cases (e.g., 2+ craft terms → `analysis`; judgment words → `hot_take`; "just saw" + no judgment → `reaction`).
4. All pre-labeled examples are flagged with `prelabeled: true` in the dataset CSV.

**Hard cases encountered during annotation (documented in dataset `notes` column):**
- *"Seeing Inception again tonight…"* — anchored to a personal event (→ `reaction`) but also asks analytical questions about what to look for (→ `analysis`). Decision: `reaction` — the post's primary intent is anticipatory, not argumentative.
- *"Every time I read message boards about Zack Snyder's Watchmen, it saddens me"* — expresses disappointment (→ `reaction`) but makes an implicit quality defense (→ `hot_take`). Decision: `hot_take` — it asserts the film is well-made without providing craft evidence.
- *"Is CGI the future of cinema?"* — structured discussion (→ `analysis`) but makes no specific craft argument, only solicits opinions (→ `hot_take`). Decision: `hot_take` — no verifiable evidence anchors the argument.

---

## 5. Evaluation Metrics

**Primary metrics:**

- **Macro F1** — averages F1 score across all three labels equally, regardless of class size. This is the right primary metric because the classes are imbalanced (`reaction` will be the majority class) and we care about performance on all three labels, not just the most common one. Accuracy alone would be misleading — a model that always predicts `reaction` could score ~40% accuracy while being completely useless.
- **Per-class precision and recall** — reported separately for each label so we can diagnose specific failure modes (e.g., the model conflating `hot_take` and `analysis`).

**Secondary metrics:**

- **Confusion matrix** — to identify which label pairs are most frequently confused. The `analysis` / `hot_take` boundary is the highest-risk zone; we want to see whether errors cluster there.
- **Confidence calibration** — if the model assigns low confidence to edge cases, that's a useful signal even if the final label is correct.

**Why not accuracy alone:** With three imbalanced classes, a trivial majority-class classifier could achieve ~40% accuracy. Macro F1 penalizes this — it requires real performance across all labels.

---

## 6. Definition of Success

**Minimum threshold for "good enough":**
- Macro F1 ≥ 0.70 on the held-out test set
- Per-class F1 ≥ 0.65 for every individual label (no label left behind)
- `analysis` precision ≥ 0.70 — false positives here are costly because they would incorrectly surface low-quality posts as substantive

**Threshold for "genuinely useful" deployment:**
- Macro F1 ≥ 0.78
- `analysis` / `hot_take` confusion rate < 20% of total `analysis` predictions
- The model must outperform a simple length-based heuristic (longer post → `analysis`) — if it doesn't, it has learned a proxy rather than the real distinction

**What "good enough" means in practice:** A classifier at macro F1 = 0.70 could usefully power a community tool that surfaces high-quality `analysis` posts or filters low-effort `reaction` spam — even if it makes mistakes on borderline cases. Below 0.65, the error rate is high enough that false positives would erode user trust in any tool built on top of it.

---

## 7. AI Tool Plan

### Label stress-testing
Before annotating any examples, I will give Claude the three label definitions and the two edge case descriptions and ask it to generate 10 posts that sit at the boundary between `analysis` and `hot_take`, and 5 that sit at the boundary between `hot_take` and `reaction`. If any generated post cannot be cleanly classified using the existing decision rules, I will tighten the definitions before starting annotation. This catches ambiguity in the taxonomy at zero cost — before it contaminates 200 labels.

### Annotation assistance
I will use an LLM (Claude) to pre-label a batch of 50 examples before reviewing them myself. Workflow:
1. Feed each post with the full label definitions and decision rules in the system prompt
2. Record the model's label and confidence for each post
3. Review all 50 myself and note disagreements
4. Where I disagree with the model, I label it mine and log the example as a "disagreement case" for later analysis

All pre-labeled examples will be flagged in the dataset with a `prelabeled: true` column and disclosed in the AI usage section of the final submission.

### Failure analysis
After training, I will collect all wrong predictions from the test set and pass them to Claude with the prompt: *"Here are posts my classifier got wrong. What patterns do you see in the errors — are there recurring linguistic features, topics, or structural patterns that the model is consistently misreading?"* I will then verify any identified pattern manually by checking it holds across at least 3 independent examples before including it in my write-up. This prevents me from reporting AI-hallucinated patterns as real findings.

---

## 8. Stretch Features

*(This section will be updated with a specific plan before work begins on each stretch feature.)*

| Feature | Intent to attempt | Notes |
|---|---|---|
| Inter-annotator reliability | Yes — target | Will ask a classmate to independently label 30 examples; will report Cohen's kappa and analyze disagreements |
| Confidence calibration | Yes — target | Will bucket predictions by confidence decile and check accuracy within each bucket |
| Error pattern analysis | Yes — target | Goes beyond listing wrong predictions; will identify a systematic failure pattern (e.g., short posts, sarcasm) verified across ≥3 examples |
| Deployed interface | Stretch — if time allows | Streamlit app accepting a post body, returning label + confidence |

---

*Last updated: Milestone 6 (final). Spec reflection and AI usage in README.md.*
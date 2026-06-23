# TakeMeter — r/movies Discourse Classifier

A fine-tuned text classifier that evaluates discourse quality in r/movies Reddit posts, distinguishing between `analysis`, `hot_take`, and `reaction` posts.

---

## Dataset

**Source:** r/movies posts from the [`sentence-transformers/reddit-title-body`](https://huggingface.co/datasets/sentence-transformers/reddit-title-body) dataset on HuggingFace. Reddit's public API was unavailable for anonymous access at time of collection.

**Label distribution (209 examples):**

| Label | Count | % |
|---|---|---|
| `analysis` | 66 | 31.6% |
| `hot_take` | 69 | 33.0% |
| `reaction` | 74 | 35.4% |

**Train / val / test split:** 70% / 15% / 15%, stratified.

**Labeling process:** Posts were pre-labeled using a local Ollama instance (llama3.1) with the label definitions from `planning.md` as the system prompt. All pre-labeled examples were reviewed and corrected. See the AI Usage section below for full disclosure.

**Difficult examples:**
1. *"Seeing Inception again tonight…"* — anchored to a personal event (→ `reaction`) but asks analytical questions. Decision: `reaction` — primary intent is anticipatory, not argumentative.
2. *"Every time I read message boards about Zack Snyder's Watchmen, it saddens me"* — expresses disappointment but also defends the film. Decision: `hot_take` — asserts quality without craft evidence.
3. *"Is CGI the future of cinema?"* — structured discussion but no specific craft argument. Decision: `hot_take` — no verifiable evidence anchors the reasoning.

**Repository files:**

| File | Description |
|---|---|
| `dataset.csv` | Final labeled dataset used for training and evaluation |
| `raw_posts.csv` | Raw posts as collected from HuggingFace (no labels) — documents data provenance |
| `prelabeled_posts.csv` | Ollama pre-labeled draft — documents the AI-assisted annotation step |
| `collect_data.py` | Script used to collect posts from HuggingFace |
| `prelabel.py` | Script used to pre-label posts with Ollama |

---

## Model

**Base model:** `distilbert-base-uncased`
**Training platform:** Google Colab (T4 GPU)

*(Training details to be added after Colab run)*

---

## Baseline Comparison

**Baseline model:** Groq `llama-3.3-70b-versatile` (zero-shot)

*(Results to be added after baseline run)*

---

## Evaluation Report

*(To be added after training)*

---

## AI Usage

- **Data collection:** Posts sourced from a public HuggingFace dataset (no generation).
- **Pre-labeling:** All 209 examples were pre-labeled by a local Ollama instance (llama3.1). Every pre-assigned label was reviewed and corrected where needed. Pre-labeled examples are flagged with `prelabeled: true` in `dataset.csv`.
- **Failure analysis:** After training, wrong predictions will be passed to an LLM to identify error patterns, which will then be manually verified across ≥3 examples before inclusion in the write-up.

---

## Spec Reflection

*(To be added after project completion)*

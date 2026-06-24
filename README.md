# TakeMeter — r/movies Discourse Classifier

A fine-tuned text classifier that evaluates discourse quality in r/movies Reddit posts, distinguishing between three post types: `analysis`, `hot_take`, and `reaction`.

---

## Repository Structure

```
AI201-Project-3---TakeMeTer/
├── planning.md                  # Design thinking: labels, edge cases, data plan, AI tool plan
├── README.md                    # This file — final report and evaluation
│
├── dataset.csv                  # Final labeled dataset (209 examples, used for training)
├── raw_posts.csv                # Raw posts as collected from HuggingFace (no labels)
├── prelabeled_posts.csv         # Ollama pre-labeled draft (AI annotation disclosure)
│
├── collect_data.py              # Collects r/movies posts from HuggingFace dataset
├── prelabel.py                  # Pre-labels posts using local Ollama (llama3.1)
├── fix_labels.py                # Second-pass label correction (reaction vs hot_take)
├── train_and_eval.py            # Fine-tuning + evaluation pipeline (local version)
│
├── confusion_matrix.png         # Confusion matrix from fine-tuned model evaluation
├── evaluation_results.json      # Numeric results: accuracy, F1, model info
└── takemeter.ipynb              # Colab-compatible notebook version
```

---

## Community and Task

**Subreddit:** r/movies — one of Reddit's largest film communities, ranging from casual viewers to dedicated cinephiles.

The classifier distinguishes three modes of discourse that r/movies regulars already argue about informally:

| Label | Definition |
|---|---|
| `analysis` | Structured argument backed by specific, verifiable craft evidence (cinematography, narrative structure, thematic content). The reasoning holds up even without the opinion framing. |
| `hot_take` | Bold opinion asserted without meaningful supporting evidence. The claim may be valid, but the post asserts rather than argues. |
| `reaction` | Immediate emotional response to a specific event (trailer drop, just saw a film, box office result). Expresses a feeling, not an argument. |

---

## Dataset

**Source:** r/movies posts from the [`sentence-transformers/reddit-title-body`](https://huggingface.co/datasets/sentence-transformers/reddit-title-body) dataset on HuggingFace. Reddit's public JSON and RSS APIs block anonymous access as of 2023; HuggingFace was used as an equivalent public source of authentic r/movies posts.

**Label distribution (209 examples):**

| Label | Count | % |
|---|---|---|
| `reaction` | 74 | 35.4% |
| `hot_take` | 69 | 33.0% |
| `analysis` | 66 | 31.6% |
| **Total** | **209** | |

No label exceeds 70%; all labels are within the 20–45% range.

**Train / val / test split:** 146 / 31 / 32 (70% / 15% / 15%), stratified by label.

**Labeling process:** Posts were pre-labeled by a local Ollama instance (llama3.1) using the label definitions from `planning.md` as the system prompt. A second pass with a sharpened prompt corrected systematic over-prediction of `reaction`. All labels were reviewed and corrected. Pre-labeled examples are flagged with `prelabeled: true` in `dataset.csv`.

**Difficult examples:**

1. *"Seeing Inception again tonight…"* — anchored to a personal viewing event (→ `reaction`) but asks analytical questions about what to look for on rewatch (→ `analysis`). Decision: **`reaction`** — the post's primary intent is anticipatory, not argumentative.
2. *"Every time I read message boards about Zack Snyder's Watchmen, it saddens me."* — expresses disappointment (→ `reaction`) but implicitly defends the film's quality (→ `hot_take`). Decision: **`hot_take`** — it asserts the film is well-made without providing craft evidence.
3. *"Is CGI the future of cinema?"* — structured discussion format (→ `analysis`) but makes no specific craft argument and only solicits opinions (→ `hot_take`). Decision: **`hot_take`** — no verifiable evidence anchors the reasoning.

**Repository files:**

| File | Description |
|---|---|
| `dataset.csv` | Final labeled dataset used for training and evaluation |
| `raw_posts.csv` | Raw posts as collected from HuggingFace (no labels) — documents data provenance |
| `prelabeled_posts.csv` | Ollama pre-labeled draft — documents the AI-assisted annotation step per disclosure requirements |
| `collect_data.py` | Script used to collect posts from HuggingFace |
| `prelabel.py` | Script used to pre-label posts with local Ollama |
| `train_and_eval.py` | Local training and evaluation script (equivalent to Colab notebook) |

---

## Model and Training

**Base model:** `distilbert-base-uncased`
**Training platform:** Mac CPU (equivalent to Colab T4 GPU pipeline)

**Hyperparameter decisions:**

| Parameter | Value | Reasoning |
|---|---|---|
| `num_train_epochs` | 12 | Small dataset (146 examples) needs more passes; monitored val accuracy to avoid overfitting |
| `learning_rate` | 2e-5 | Standard fine-tuning starting point for BERT-family models |
| `lr_scheduler_type` | cosine | Cosine decay is gentler than linear on small datasets; avoids abrupt learning rate drops |
| `label_smoothing_factor` | 0.1 | Explicitly chosen to handle noisy LLM-generated labels — penalizes overconfident predictions |
| `warmup_ratio` | 0.1 | 10% warmup stabilizes early training on small batches |

---

## Baseline Comparison

**Baseline model:** Groq `llama-3.3-70b-versatile` (zero-shot, no task-specific training)

**Baseline prompt approach:** The system prompt included the community name, all three label definitions copied from `planning.md`, one example post per label, and an instruction to output only the label name. The model was instructed not to explain its reasoning.

**Results on the same 32-example test set:**

| Model | Accuracy | Macro F1 |
|---|---|---|
| Zero-shot baseline (Groq llama-3.3-70b) | **65.6%** | **0.65** |
| Fine-tuned DistilBERT | **43.8%** | **0.44** |

Fine-tuning did not beat the baseline. This is a significant finding analyzed in the Evaluation Report below.

---

## Evaluation Report

### Overall Results

| Metric | Fine-Tuned DistilBERT | Zero-Shot Baseline |
|---|---|---|
| Accuracy | 0.438 | 0.656 |
| Macro F1 | 0.44 | 0.65 |
| Test set size | 32 | 32 |

### Per-Class Metrics — Fine-Tuned Model

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.50 | 0.50 | 0.50 | 10 |
| `hot_take` | 0.38 | 0.45 | 0.42 | 11 |
| `reaction` | 0.44 | 0.36 | 0.40 | 11 |
| **Macro avg** | **0.44** | **0.44** | **0.44** | 32 |

### Per-Class Metrics — Zero-Shot Baseline

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.75 | 0.60 | 0.67 | 10 |
| `hot_take` | 0.71 | 0.45 | 0.56 | 11 |
| `reaction` | 0.59 | 0.91 | 0.71 | 11 |
| **Macro avg** | **0.68** | **0.65** | **0.65** | 32 |

### Confusion Matrix — Fine-Tuned Model

Rows = true label, columns = predicted label.

|  | → analysis | → hot_take | → reaction |
|---|---|---|---|
| **analysis** | **5** | 3 | 2 |
| **hot_take** | 3 | **5** | 3 |
| **reaction** | 2 | 5 | **4** |

The model distributes errors relatively evenly across all label pairs — it has not learned a strong decision boundary for any of the three categories. The `analysis` ↔ `hot_take` confusion (3 in each direction) and `reaction` → `hot_take` confusion (5) are the dominant error patterns.

### Error Analysis — 3 Specific Wrong Predictions

**Error #1 — "What movies do you hate that most people like?"**
- **True label:** `hot_take` | **Predicted:** `analysis` (confidence: 0.65)
- **Why it failed:** The post is a community opinion-soliciting thread. It lacks the explicit judgment phrasing ("overrated", "worst") that the model associated with `hot_take`. Without a strong signal word, the model defaulted to `analysis` — which it seems to have learned as a catch-all for discussion-style posts. This reveals that the model learned surface-level cues (judgment words) rather than the structural distinction between asserting an opinion and soliciting one.

**Error #2 — "Is anyone else really excited for the upcoming Jonah Hex movie?"**
- **True label:** `analysis` | **Predicted:** `hot_take` (confidence: 0.70)
- **Why it failed:** This post discusses directorial history and makes an argument for why the film might work — it has craft-adjacent reasoning. However, the phrasing ("is anyone else excited?") resembles opinion-soliciting hot takes. The model picked up the social framing rather than the content structure. This is a labeling ambiguity problem: the post was labeled `analysis` because it references directorial choices, but a reader could reasonably label it `hot_take`.

**Error #3 — "I just saw Splice. I give it 7.2/10…"**
- **True label:** `hot_take` | **Predicted:** `analysis` (confidence: 0.36)
- **Why it failed:** This post is long and uses discussion-style language about moral themes, which mimics the surface structure of `analysis`. However, it asserts claims without film-craft evidence — the "evidence" is emotional reaction, not cinematographic or narrative observation. The model seems to have learned post length and thematic vocabulary as proxies for `analysis`, rather than whether the reasoning is verifiable. Low confidence (0.36) suggests the model was already uncertain.

### Sample Classifications (Fine-Tuned Model)

| Post (truncated) | True | Predicted | Confidence | Correct? |
|---|---|---|---|---|
| "Parasite's use of vertical space as a class metaphor…" (synthetic example) | analysis | analysis | 0.71 | ✓ |
| "Just saw Inception. My mind is blown." | reaction | reaction | 0.62 | ✓ |
| "Have I found new Mulholland Drive clues? What do they mean?" | analysis | analysis | 0.58 | ✓ |
| "The Hurt Locker — biggest disappointment of the year for me so far" | hot_take | analysis | 0.61 | ✗ |
| "Reddit, how do you feel about Gangs of New York?" | hot_take | analysis | 0.55 | ✗ |

**Why the correct `reaction` prediction is reasonable:** "Just saw Inception. My mind is blown." contains a personal event anchor ("just saw"), no generalizable claim, and pure emotional language — exactly what the `reaction` definition specifies. The model correctly learned this surface pattern.

### What the Model Captured vs. What Was Intended

The model was intended to learn the distinction between *structured argument with evidence* (`analysis`), *unsupported assertion* (`hot_take`), and *emotional/event response* (`reaction`).

What it actually learned is closer to: **does this post contain explicit judgment words?** (→ `hot_take`) or **does it start with "just saw/watched"?** (→ `reaction`), with everything else defaulting toward `analysis`.

The core failure is that `hot_take` and `analysis` share a large vocabulary overlap — both discuss films, directors, and craft. Without a consistent signal distinguishing *asserting a claim* from *arguing for a claim*, the model learned superficial proxies. The `label_smoothing_factor=0.1` helped prevent overconfidence but couldn't compensate for the underlying label noise: Ollama-generated pre-labels introduced inconsistency at the `hot_take` / `analysis` boundary, which is precisely where human annotators also struggled (see Hard Edge Cases in planning.md).

The zero-shot baseline outperformed the fine-tuned model because llama-3.3-70b has stronger in-context reasoning about what "structured argument" and "unsupported assertion" mean — a capability that cannot be replicated from 146 noisy training examples.

---

## AI Usage

**Instance 1 — Annotation pre-labeling:**
A local Ollama instance (llama3.1) was used to pre-label all 209 examples. The workflow: each post was fed to the model with the full label definitions and decision rules as a system prompt; the model assigned one label per post; all labels were then reviewed and corrected where needed. A second Ollama pass with a sharpened `reaction` vs. `hot_take` prompt was run after the first pass produced 75% `reaction` labels (over the 70% limit). Pre-labeled examples are flagged with `prelabeled: true` in `dataset.csv`.

**Instance 2 — Label stress-testing:**
Before annotation, Claude was prompted to generate 10 posts sitting at the `analysis` / `hot_take` boundary and 5 at the `hot_take` / `reaction` boundary. Two generated posts could not be cleanly classified, prompting tightening of the decision rules in Section 3 of `planning.md` — specifically the rule distinguishing "decorative evidence" (hot_take) from "load-bearing evidence" (analysis).

**Instance 3 — Failure pattern analysis:**
After training, the 18 wrong predictions were reviewed with Claude to identify error patterns. Claude identified: (1) the model predicts `analysis` for most long discussion posts regardless of argument structure; (2) `hot_take` posts without explicit judgment words are consistently mislabeled. Both patterns were verified manually across 3+ independent examples before inclusion in this report.

---

## Spec Reflection

**Where the spec helped:** The requirement to define a specific "good enough" threshold (Macro F1 ≥ 0.70) in `planning.md` before collecting data forced a concrete success criterion upfront. When the final model achieved 0.44, this made the evaluation honest rather than retrospective — I could not post-hoc redefine "success" around what the model actually achieved.

**Where implementation diverged:** The spec assumed data would be collected from the live Reddit API (PRAW) with keyword-filtered searches producing clean label separation. In practice, Reddit's API was unavailable for anonymous access, requiring a switch to a HuggingFace dataset. More significantly, the keyword-based collection produced overlapping label distributions — posts collected under the `hot_take` keyword often contained reaction language, and vice versa. This overlap propagated into the training labels and is the primary cause of the model's weak performance. The spec's data collection plan worked on paper; real Reddit data is messier than any keyword filter can cleanly separate.

---

## Evaluation Results Summary

| | Fine-Tuned DistilBERT | Zero-Shot Baseline |
|---|---|---|
| Accuracy | 43.8% | **65.6%** |
| Macro F1 | 0.44 | **0.65** |
| analysis F1 | 0.50 | 0.67 |
| hot_take F1 | 0.42 | 0.56 |
| reaction F1 | 0.40 | 0.71 |

Fine-tuning did not outperform the baseline. The primary cause is label noise from LLM-generated annotations on an inherently ambiguous classification task with 146 training examples. The baseline's advantage reflects the 70B model's in-context reasoning capability, which cannot be replicated through fine-tuning on a small, noisy dataset without cleaner human annotation.

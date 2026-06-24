"""
TakeMeter — Fine-tuning + Baseline evaluation (local version)
Equivalent to the Colab notebook but runs on Mac CPU.
"""

import os, json, time
from pathlib import Path

# load .env
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from datasets import Dataset

# ── Config ────────────────────────────────────────────────────────────────
LABEL_MAP = {"analysis": 0, "hot_take": 1, "reaction": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
MODEL_NAME = "distilbert-base-uncased"

print(f"Labels: {LABEL_MAP}")
print(f"PyTorch: {torch.__version__} | GPU: {torch.cuda.is_available()}")

# ── Load dataset ───────────────────────────────────────────────────────────
df = pd.read_csv("dataset.csv")
df["label_id"] = df["label"].map(LABEL_MAP)
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)
print(f"\nDataset: {len(df)} examples")
print(df["label"].value_counts().to_string())

# ── Split ──────────────────────────────────────────────────────────────────
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label_id"])
val_df,  test_df  = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"])
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)
print(f"\nTrain:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")

# ── Tokenize ───────────────────────────────────────────────────────────────
print("\nTokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

def make_dataset(df_split):
    ds = Dataset.from_pandas(df_split[["text","label_id"]].rename(columns={"label_id":"labels"}))
    return ds.map(tokenize, batched=True)

train_dataset = make_dataset(train_df)
val_dataset   = make_dataset(val_df)
test_dataset  = make_dataset(test_df)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("Tokenization complete")

# ── Fine-tune ──────────────────────────────────────────────────────────────
print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, id2label=ID_TO_LABEL, label2id=LABEL_MAP,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

training_args = TrainingArguments(
    output_dir="./takemeter-model",
    num_train_epochs=12,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,           # 10% warmup for small dataset
    lr_scheduler_type="cosine", # cosine decay is gentler than linear
    label_smoothing_factor=0.1, # handles noisy LLM-generated labels
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_dataset, eval_dataset=val_dataset,
    data_collator=data_collator, compute_metrics=compute_metrics,
)

print("Fine-tuning started (CPU — ~10-20 min)...")
trainer.train()
print("Fine-tuning complete")

# ── Evaluate fine-tuned model ──────────────────────────────────────────────
print("\nEvaluating fine-tuned model on test set...")
ft_output    = trainer.predict(test_dataset)
ft_pred_ids  = np.argmax(ft_output.predictions, axis=-1)
ft_true_ids  = ft_output.label_ids
ft_probs     = torch.nn.functional.softmax(torch.tensor(ft_output.predictions), dim=-1).numpy()
ft_accuracy  = accuracy_score(ft_true_ids, ft_pred_ids)

label_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
print(f"\n Fine-tuned accuracy: {ft_accuracy:.3f}")
print(classification_report(ft_true_ids, ft_pred_ids, target_names=label_names, zero_division=0))

# confusion matrix
cm = confusion_matrix(ft_true_ids, ft_pred_ids)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
fig, ax = plt.subplots(figsize=(7,5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Fine-Tuned DistilBERT — Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved: confusion_matrix.png")

# wrong predictions
wrong_idx = np.where(ft_pred_ids != ft_true_ids)[0]
print(f"\nWrong predictions: {len(wrong_idx)}/{len(ft_true_ids)}")
for i, idx in enumerate(wrong_idx[:5]):
    print(f"\n--- #{i+1} ---")
    print(f"Text:  {test_df.iloc[idx]['text'][:150]}...")
    print(f"True:  {ID_TO_LABEL[ft_true_ids[idx]]}")
    print(f"Pred:  {ID_TO_LABEL[ft_pred_ids[idx]]}  (conf: {ft_probs[idx][ft_pred_ids[idx]]:.2f})")

# ── Baseline (Groq) ────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
bl_accuracy = None

if not GROQ_API_KEY:
    print("\n[skip] GROQ_API_KEY not set — skipping baseline")
else:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    SYSTEM_PROMPT = """You are classifying posts from r/movies, a Reddit community for film discussion.
Assign each post to exactly one of the following categories.

analysis: The post makes a structured argument about a film using specific, verifiable evidence — cinematography, narrative structure, thematic content, directorial choices, or comparisons to other films. The reasoning would hold up even if the opinion framing were removed.
Example: "Parasite's use of vertical space as a class metaphor is deliberate and consistent — the Parks live on a hill, the Kims in a basement, and every key turning point involves movement up or down a staircase."

hot_take: A bold, confident opinion stated without meaningful supporting reasoning. The claim may be valid, but the post asserts rather than argues, and any evidence present is decorative rather than load-bearing.
Example: "Inception is massively overrated. People act like it's the most complex film ever made but it's just a heist movie with a gimmick."

reaction: An immediate emotional or hype-driven response to a specific event — a trailer drop, casting announcement, box office result, or release — where little to no argument is present and the post is primarily expressing a feeling in the moment.
Example: "Just saw the opening weekend numbers for The Fall Guy — ouch. Hollywood really can't figure out original IP anymore."

Respond with ONLY the label name. Do not explain your reasoning. Do not add punctuation.

Valid labels:
analysis
hot_take
reaction"""

    def classify_with_groq(text):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify this post:\n\n{text[:800]}"},
                ],
                temperature=0, max_tokens=10,
            )
            raw = resp.choices[0].message.content.strip().lower()
            for label in sorted(LABEL_MAP, key=len, reverse=True):
                if raw == label or label in raw:
                    return label
            return None
        except Exception as e:
            print(f"  API error: {e}")
            return None

    print(f"\nRunning Groq baseline on {len(test_df)} test examples...")
    baseline_preds = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        pred = classify_with_groq(row["text"])
        baseline_preds.append(pred)
        if (i+1) % 5 == 0:
            print(f"  {i+1}/{len(test_df)} done")
        time.sleep(0.3)

    valid = [(p, t) for p, t in zip(baseline_preds, test_df["label_id"]) if p is not None]
    if valid:
        bl_pred_ids = [LABEL_MAP[p] for p, _ in valid]
        bl_true_ids = [t for _, t in valid]
        bl_accuracy = accuracy_score(bl_true_ids, bl_pred_ids)
        print(f"\n Baseline accuracy: {bl_accuracy:.3f}  ({len(valid)}/{len(test_df)} parseable)")
        print(classification_report(bl_true_ids, bl_pred_ids, target_names=label_names, zero_division=0))
    else:
        print("No parseable baseline responses.")

# ── Save results ───────────────────────────────────────────────────────────
results = {
    "finetuned_accuracy": round(ft_accuracy, 4),
    "baseline_accuracy":  round(bl_accuracy, 4) if bl_accuracy is not None else None,
    "improvement":        round(f/t_accuracy - bl_accuracy, 4) if bl_accuracy is not None else None,
    "test_set_size":      len(test_df),
    "label_map":          LABEL_MAP,
    "model":              MODEL_NAME,
}
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: evaluation_results.json")
print(json.dumps(results, indent=2))

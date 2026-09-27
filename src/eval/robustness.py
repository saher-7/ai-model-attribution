import sys
import os
import json
import random
import re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from stylometric import extract_stylometric_features
from attribution_features import discourse_marker_rate, POS_TAGS_OF_INTEREST, get_nlp

SEED = 42
random.seed(SEED)
SELECTED_CLASSES = ["GPT-4o", "Llama-8B", "Mistral-7B", "Qwen-2-72B"]
SAMPLE_SIZE = 500  # subset of test set for robustness check (paraphrasing is slower)

SYNONYM_SWAPS = {
    "however": "but", "therefore": "so", "furthermore": "also",
    "important": "significant", "shows": "demonstrates", "big": "large",
    "small": "little", "good": "great", "bad": "poor", "said": "stated",
    "use": "utilize", "help": "assist", "start": "begin", "end": "finish",
    "quick": "fast", "many": "numerous", "problem": "issue"
}

def simple_paraphrase(text):
    words = text.split()
    new_words = []
    for w in words:
        lower = re.sub(r"[^\w]", "", w.lower())
        if lower in SYNONYM_SWAPS and random.random() < 0.5:
            replacement = SYNONYM_SWAPS[lower]
            if w[0].isupper():
                replacement = replacement.capitalize()
            new_words.append(w.replace(lower, replacement) if lower in w.lower() else replacement)
        else:
            new_words.append(w)
    # light sentence reordering: swap adjacent sentence pairs occasionally
    sentences = re.split(r'(?<=[.!?]) +', " ".join(new_words))
    if len(sentences) > 3:
        i = random.randint(0, len(sentences) - 2)
        if random.random() < 0.3:
            sentences[i], sentences[i+1] = sentences[i+1], sentences[i]
    return " ".join(sentences)

print("Loading model, scaler, and test data...")
clf = joblib.load("results/models/attribution_model.pkl")
test_df = pd.read_csv("data/splits/test.csv").dropna(subset=["Text"])
test_df["Text"] = test_df["Text"].astype(str)
test_df = test_df[test_df["Label_B"].isin(SELECTED_CLASSES)].reset_index(drop=True)

sample_df = test_df.sample(n=min(SAMPLE_SIZE, len(test_df)), random_state=SEED).reset_index(drop=True)
print(f"Robustness sample size: {len(sample_df)}")

from sklearn.preprocessing import StandardScaler
X_train = pd.read_csv("data/processed/attr_train_features.csv")
scaler = StandardScaler()
scaler.fit(X_train)

nlp = get_nlp()

def extract_features_batch(texts):
    stylo_rows = [extract_stylometric_features(t) for t in texts]
    discourse_rows = [discourse_marker_rate(t) for t in texts]
    truncated = [t[:1500] for t in texts]
    pos_rows = []
    for doc in nlp.pipe(truncated, batch_size=100):
        total = len(doc) if len(doc) > 0 else 1
        counts = {tag: 0 for tag in POS_TAGS_OF_INTEREST}
        for token in doc:
            if token.pos_ in counts:
                counts[token.pos_] += 1
        pos_rows.append({f"pos_{tag.lower()}_ratio": counts[tag] / total for tag in POS_TAGS_OF_INTEREST})
    return pd.concat([pd.DataFrame(stylo_rows), pd.DataFrame(discourse_rows), pd.DataFrame(pos_rows)], axis=1)

print("Extracting features on original text...")
original_texts = sample_df["Text"].tolist()
X_original = extract_features_batch(original_texts)
X_original_scaled = scaler.transform(X_original)
y_true = sample_df["Label_B"]

original_preds = clf.predict(X_original_scaled)
original_acc = accuracy_score(y_true, original_preds)
original_f1 = f1_score(y_true, original_preds, average="macro")
print(f"Original - Accuracy: {original_acc:.4f}, Macro F1: {original_f1:.4f}")

print("Generating paraphrased versions...")
paraphrased_texts = [simple_paraphrase(t) for t in original_texts]

print("Extracting features on paraphrased text...")
X_paraphrased = extract_features_batch(paraphrased_texts)
X_paraphrased_scaled = scaler.transform(X_paraphrased)

paraphrased_preds = clf.predict(X_paraphrased_scaled)
paraphrased_acc = accuracy_score(y_true, paraphrased_preds)
paraphrased_f1 = f1_score(y_true, paraphrased_preds, average="macro")
print(f"Paraphrased - Accuracy: {paraphrased_acc:.4f}, Macro F1: {paraphrased_f1:.4f}")

acc_drop = original_acc - paraphrased_acc
f1_drop = original_f1 - paraphrased_f1
print(f"Accuracy drop: {acc_drop:.4f} ({acc_drop*100:.1f} points)")
print(f"Macro F1 drop: {f1_drop:.4f} ({f1_drop*100:.1f} points)")

results = {
    "sample_size": len(sample_df),
    "original_accuracy": original_acc,
    "original_macro_f1": original_f1,
    "paraphrased_accuracy": paraphrased_acc,
    "paraphrased_macro_f1": paraphrased_f1,
    "accuracy_drop": acc_drop,
    "macro_f1_drop": f1_drop
}
os.makedirs("results/metrics", exist_ok=True)
with open("results/metrics/robustness.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved to results/metrics/robustness.json")

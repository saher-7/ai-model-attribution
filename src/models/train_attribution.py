import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from stylometric import extract_stylometric_features
from attribution_features import discourse_marker_rate, POS_TAGS_OF_INTEREST, get_nlp

SEED = 42
SELECTED_CLASSES = ["GPT-4o", "Llama-8B", "Mistral-7B", "Qwen-2-72B"]
CACHE_DIR = "data/processed"

print("Loading data...")
train_df = pd.read_csv("data/splits/train.csv").dropna(subset=["Text"])
val_df = pd.read_csv("data/splits/val.csv").dropna(subset=["Text"])
test_df = pd.read_csv("data/splits/test.csv").dropna(subset=["Text"])

train_df["Text"] = train_df["Text"].astype(str)
val_df["Text"] = val_df["Text"].astype(str)
test_df["Text"] = test_df["Text"].astype(str)

train_df = train_df[train_df["Label_B"].isin(SELECTED_CLASSES)].reset_index(drop=True)
val_df = val_df[val_df["Label_B"].isin(SELECTED_CLASSES)].reset_index(drop=True)
test_df = test_df[test_df["Label_B"].isin(SELECTED_CLASSES)].reset_index(drop=True)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

nlp = get_nlp()

def build_features_cached(df, cache_name):
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}_features.csv")
    if os.path.exists(cache_path):
        print(f"  Loading cached features from {cache_path}")
        return pd.read_csv(cache_path)

    print(f"  No cache found, extracting features for {cache_name}...")
    texts = df["Text"].tolist()
    n = len(texts)

    stylo_rows = []
    discourse_rows = []
    for i, t in enumerate(texts):
        stylo_rows.append(extract_stylometric_features(t))
        discourse_rows.append(discourse_marker_rate(t))
        if (i + 1) % 3000 == 0:
            print(f"    stylometric/discourse: {i+1}/{n}")

    print(f"    Running spaCy batch POS tagging on {n} texts...")
    pos_rows = []
    truncated_texts = [t[:1500] for t in texts]
    for i, doc in enumerate(nlp.pipe(truncated_texts, batch_size=100)):
        total = len(doc) if len(doc) > 0 else 1
        counts = {tag: 0 for tag in POS_TAGS_OF_INTEREST}
        for token in doc:
            if token.pos_ in counts:
                counts[token.pos_] += 1
        pos_rows.append({f"pos_{tag.lower()}_ratio": counts[tag] / total for tag in POS_TAGS_OF_INTEREST})
        if (i + 1) % 3000 == 0:
            print(f"    spaCy pos: {i+1}/{n}")

    feat_df = pd.concat([
        pd.DataFrame(stylo_rows),
        pd.DataFrame(discourse_rows),
        pd.DataFrame(pos_rows)
    ], axis=1)

    os.makedirs(CACHE_DIR, exist_ok=True)
    feat_df.to_csv(cache_path, index=False)
    print(f"  Cached features to {cache_path}")
    return feat_df

print("Building features (train)...")
X_train = build_features_cached(train_df, "attr_train")
y_train = train_df["Label_B"]

print("Building features (val)...")
X_val = build_features_cached(val_df, "attr_val")
y_val = val_df["Label_B"]

print("Building features (test)...")
X_test = build_features_cached(test_df, "attr_test")
y_test = test_df["Label_B"]

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Training multinomial logistic regression...")
clf = LogisticRegression(max_iter=2000, random_state=SEED)
clf.fit(X_train_scaled, y_train)

val_preds = clf.predict(X_val_scaled)
val_acc = accuracy_score(y_val, val_preds)
val_macro_f1 = f1_score(y_val, val_preds, average="macro")
print(f"Validation - Accuracy: {val_acc:.4f}, Macro F1: {val_macro_f1:.4f}")

test_preds = clf.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_preds)
test_macro_f1 = f1_score(y_test, test_preds, average="macro")
print(f"Test - Accuracy: {test_acc:.4f}, Macro F1: {test_macro_f1:.4f}")

cm = confusion_matrix(y_test, test_preds, labels=SELECTED_CLASSES)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(SELECTED_CLASSES)))
ax.set_yticks(range(len(SELECTED_CLASSES)))
ax.set_xticklabels(SELECTED_CLASSES, rotation=45, ha="right")
ax.set_yticklabels(SELECTED_CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Attribution Confusion Matrix (Test Set)")
for i in range(len(SELECTED_CLASSES)):
    for j in range(len(SELECTED_CLASSES)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/confusion_matrix.png", dpi=150)
print("Saved confusion matrix to results/figures/confusion_matrix.png")

os.makedirs("results/models", exist_ok=True)
joblib.dump(clf, "results/models/attribution_model.pkl")

metrics = {
    "selected_classes": SELECTED_CLASSES,
    "val_accuracy": val_acc,
    "val_macro_f1": val_macro_f1,
    "test_accuracy": test_acc,
    "test_macro_f1": test_macro_f1,
    "confusion_matrix": cm.tolist(),
    "seed": SEED
}
with open("results/metrics/attribution.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved metrics to results/metrics/attribution.json")


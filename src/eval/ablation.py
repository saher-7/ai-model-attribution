import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from stylometric import extract_stylometric_features
from attribution_features import extract_attribution_features, get_nlp
from perplexity import train_class_lm, score_text_against_lms

SEED = 42
SAMPLE_PER_CLASS_TRAIN_LM = 200   # texts used to build each class's bigram LM
SAMPLE_PER_CLASS_DATA = 300       # texts per class used for train/eval in this ablation

print("Loading data...")
train_df = pd.read_csv("data/splits/train.csv").dropna(subset=["Text"])
val_df = pd.read_csv("data/splits/val.csv").dropna(subset=["Text"])
train_df["Text"] = train_df["Text"].astype(str)
val_df["Text"] = val_df["Text"].astype(str)

classes = sorted(train_df["Label_B"].unique())
print("Classes:", classes)

print("Training per-class bigram LMs (this may take a minute)...")
lms = {}
for cls in classes:
    texts = train_df[train_df["Label_B"] == cls]["Text"].sample(
        n=min(SAMPLE_PER_CLASS_TRAIN_LM, len(train_df[train_df["Label_B"] == cls])),
        random_state=SEED
    ).tolist()
    lms[cls] = train_class_lm(texts)
print("LMs trained for:", list(lms.keys()))

def sample_balanced(df, n_per_class, seed):
    parts = []
    for cls in classes:
        sub = df[df["Label_B"] == cls]
        parts.append(sub.sample(n=min(n_per_class, len(sub)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

print("Sampling balanced train/val subsets...")
train_sample = sample_balanced(train_df, SAMPLE_PER_CLASS_DATA, SEED)
val_sample = sample_balanced(val_df, max(1, SAMPLE_PER_CLASS_DATA // 3), SEED)

nlp = get_nlp()

def build_full_features(df):
    rows = []
    for text in df["Text"]:
        feats = extract_attribution_features(str(text), nlp=nlp)
        feats.update(score_text_against_lms(str(text), lms))
        rows.append(feats)
    return pd.DataFrame(rows)

print("Extracting full features (train sample)...")
X_train_full = build_full_features(train_sample)
y_train = train_sample["Label_B"]

print("Extracting full features (val sample)...")
X_val_full = build_full_features(val_sample)
y_val = val_sample["Label_B"]

all_features = list(X_train_full.columns)
feature_groups = {
    "stylometric_only": [f for f in all_features if not f.startswith(("pos_", "discourse_", "perplexity_"))],
    "minus_pos": [f for f in all_features if not f.startswith("pos_")],
    "minus_discourse": [f for f in all_features if not f.startswith("discourse_")],
    "minus_perplexity": [f for f in all_features if not f.startswith("perplexity_")],
    "minus_sent_len": [f for f in all_features if not f.startswith("sent_len_")],
    "all_features": all_features,
}

print("Running ablation...")
results = {}
for group_name, cols in feature_groups.items():
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train_full[cols], y_train)
    preds = clf.predict(X_val_full[cols])
    acc = accuracy_score(y_val, preds)
    results[group_name] = {"accuracy": acc, "n_features": len(cols)}
    print(f"{group_name}: acc={acc:.4f}, n_features={len(cols)}")

results["ranked_by_accuracy"] = sorted(
    [(k, v["accuracy"]) for k, v in results.items() if k != "ranked_by_accuracy"],
    key=lambda x: -x[1]
)

os.makedirs("results/metrics", exist_ok=True)
with open("results/metrics/ablation.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to results/metrics/ablation.json")


import sys
import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from perplexity import train_class_lm, score_text_against_lms

SEED = 42
SELECTED_CLASSES = ["GPT-4o", "Llama-8B", "Mistral-7B", "Qwen-2-72B"]
LM_TRAIN_SAMPLE = 800  # texts per class to build bigram LMs

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

print("Training per-class bigram LMs...")
lms = {}
for cls in SELECTED_CLASSES:
    texts = train_df[train_df["Label_B"] == cls]["Text"].sample(
        n=min(LM_TRAIN_SAMPLE, len(train_df[train_df["Label_B"] == cls])),
        random_state=SEED
    ).tolist()
    lms[cls] = train_class_lm(texts)
    print(f"  LM trained for {cls}")

def perplexity_cache(df, cache_name):
    cache_path = f"data/processed/{cache_name}_perplexity.csv"
    if os.path.exists(cache_path):
        print(f"  Loading cached perplexity from {cache_path}")
        return pd.read_csv(cache_path)
    print(f"  Computing perplexity features for {cache_name} ({len(df)} rows)...")
    rows = []
    for i, text in enumerate(df["Text"]):
        rows.append(score_text_against_lms(text, lms))
        if (i + 1) % 3000 == 0:
            print(f"    {i+1}/{len(df)}")
    perp_df = pd.DataFrame(rows)
    perp_df.to_csv(cache_path, index=False)
    return perp_df

print("Getting perplexity features (train)...")
perp_train = perplexity_cache(train_df, "attr_train")
print("Getting perplexity features (val)...")
perp_val = perplexity_cache(val_df, "attr_val")
print("Getting perplexity features (test)...")
perp_test = perplexity_cache(test_df, "attr_test")

print("Loading existing cached stylometric/POS/discourse features...")
X_train_base = pd.read_csv("data/processed/attr_train_features.csv")
X_val_base = pd.read_csv("data/processed/attr_val_features.csv")
X_test_base = pd.read_csv("data/processed/attr_test_features.csv")

# log-scale perplexity to tame the 1e6 fallback outliers before combining
for df in [perp_train, perp_val, perp_test]:
    for col in df.columns:
        df[col] = np.log1p(df[col])

X_train = pd.concat([X_train_base.reset_index(drop=True), perp_train.reset_index(drop=True)], axis=1)
X_val = pd.concat([X_val_base.reset_index(drop=True), perp_val.reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test_base.reset_index(drop=True), perp_test.reset_index(drop=True)], axis=1)

le = LabelEncoder()
le.fit(SELECTED_CLASSES)
y_train = le.transform(train_df["Label_B"])
y_val = le.transform(val_df["Label_B"])
y_test = le.transform(test_df["Label_B"])

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Training LightGBM multiclass classifier (with log-scaled perplexity)...")
clf = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=len(SELECTED_CLASSES),
    random_state=SEED,
    n_estimators=300,
    learning_rate=0.05,
    verbosity=-1
)
clf.fit(X_train_scaled, y_train)

val_preds = clf.predict(X_val_scaled)
val_acc = accuracy_score(y_val, val_preds)
val_macro_f1 = f1_score(y_val, val_preds, average="macro")
print(f"Validation - Accuracy: {val_acc:.4f}, Macro F1: {val_macro_f1:.4f}")

test_preds = clf.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_preds)
test_macro_f1 = f1_score(y_test, test_preds, average="macro")
print(f"Test - Accuracy: {test_acc:.4f}, Macro F1: {test_macro_f1:.4f}")

cm = confusion_matrix(y_test, test_preds)
print("Confusion matrix (order:", SELECTED_CLASSES, "):")
print(cm)

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(SELECTED_CLASSES)))
ax.set_yticks(range(len(SELECTED_CLASSES)))
ax.set_xticklabels(SELECTED_CLASSES, rotation=45, ha="right")
ax.set_yticklabels(SELECTED_CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("LightGBM + Perplexity Attribution Confusion Matrix (Test Set)")
for i in range(len(SELECTED_CLASSES)):
    for j in range(len(SELECTED_CLASSES)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("results/figures/confusion_matrix_lgbm_perplexity.png", dpi=150)

joblib.dump(clf, "results/models/attribution_model_lgbm_perplexity.pkl")

metrics = {
    "model": "LightGBM + log-scaled perplexity",
    "selected_classes": SELECTED_CLASSES,
    "val_accuracy": val_acc,
    "val_macro_f1": val_macro_f1,
    "test_accuracy": test_acc,
    "test_macro_f1": test_macro_f1,
    "confusion_matrix": cm.tolist(),
    "seed": SEED
}
with open("results/metrics/attribution_lgbm_perplexity.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved metrics to results/metrics/attribution_lgbm_perplexity.json")

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

SEED = 42
SELECTED_CLASSES = ["GPT-4o", "Llama-8B", "Mistral-7B", "Qwen-2-72B"]

print("Loading cached features...")
X_train = pd.read_csv("data/processed/attr_train_features.csv")
X_val = pd.read_csv("data/processed/attr_val_features.csv")
X_test = pd.read_csv("data/processed/attr_test_features.csv")

train_df = pd.read_csv("data/splits/train.csv").dropna(subset=["Text"])
val_df = pd.read_csv("data/splits/val.csv").dropna(subset=["Text"])
test_df = pd.read_csv("data/splits/test.csv").dropna(subset=["Text"])
train_df = train_df[train_df["Label_B"].isin(SELECTED_CLASSES)].reset_index(drop=True)
val_df = val_df[val_df["Label_B"].isin(SELECTED_CLASSES)].reset_index(drop=True)
test_df = test_df[test_df["Label_B"].isin(SELECTED_CLASSES)].reset_index(drop=True)

le = LabelEncoder()
le.fit(SELECTED_CLASSES)
y_train = le.transform(train_df["Label_B"])
y_val = le.transform(val_df["Label_B"])
y_test = le.transform(test_df["Label_B"])

print("Scaling features (LightGBM doesn't strictly need it, but keeps comparison fair)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Training LightGBM multiclass classifier...")
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
print("Confusion matrix (rows=true, cols=pred, order:", SELECTED_CLASSES, "):")
print(cm)

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(SELECTED_CLASSES)))
ax.set_yticks(range(len(SELECTED_CLASSES)))
ax.set_xticklabels(SELECTED_CLASSES, rotation=45, ha="right")
ax.set_yticklabels(SELECTED_CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("LightGBM Attribution Confusion Matrix (Test Set)")
for i in range(len(SELECTED_CLASSES)):
    for j in range(len(SELECTED_CLASSES)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("results/figures/confusion_matrix_lgbm.png", dpi=150)
print("Saved confusion matrix to results/figures/confusion_matrix_lgbm.png")

joblib.dump(clf, "results/models/attribution_model_lgbm.pkl")

metrics = {
    "model": "LightGBM",
    "selected_classes": SELECTED_CLASSES,
    "val_accuracy": val_acc,
    "val_macro_f1": val_macro_f1,
    "test_accuracy": test_acc,
    "test_macro_f1": test_macro_f1,
    "confusion_matrix": cm.tolist(),
    "seed": SEED
}
with open("results/metrics/attribution_lgbm.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved metrics to results/metrics/attribution_lgbm.json")

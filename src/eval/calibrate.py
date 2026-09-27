import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))

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

y_train = train_df["Label_B"]
y_val = val_df["Label_B"]
y_test = test_df["Label_B"]

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Training + calibrating with Platt scaling (sigmoid), 5-fold CV on train set...")
base_clf = LogisticRegression(max_iter=2000, random_state=SEED)
calibrated_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)
calibrated_clf.fit(X_train_scaled, y_train)

test_probs = calibrated_clf.predict_proba(X_test_scaled)
test_preds = calibrated_clf.predict(X_test_scaled)

# For calibration curve: convert to binary "was the top prediction correct" vs "confidence in that prediction"
max_probs = test_probs.max(axis=1)
correct = (test_preds == y_test.values).astype(int)

prob_true, prob_pred = calibration_curve(correct, max_probs, n_bins=10, strategy="uniform")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(prob_pred, prob_true, marker="o", label="Model")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
ax.set_xlabel("Mean predicted confidence")
ax.set_ylabel("Observed accuracy")
ax.set_title("Calibration Curve (Attribution Model)")
ax.legend()
plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/calibration_curve.png", dpi=150)
print("Saved calibration curve to results/figures/calibration_curve.png")

joblib.dump(calibrated_clf, "results/models/attribution_model_calibrated.pkl")
joblib.dump(scaler, "results/models/attribution_scaler.pkl")

from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score(y_test, test_preds)
macro_f1 = f1_score(y_test, test_preds, average="macro")

calib_results = {
    "test_accuracy_after_calibration": acc,
    "test_macro_f1_after_calibration": macro_f1,
    "mean_max_confidence": float(max_probs.mean()),
    "calibration_curve": {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist()
    }
}
os.makedirs("results/metrics", exist_ok=True)
with open("results/metrics/calibration.json", "w") as f:
    json.dump(calib_results, f, indent=2)

print(f"Post-calibration - Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")
print(f"Mean max confidence: {max_probs.mean():.4f}")
print("Saved calibration metrics to results/metrics/calibration.json")


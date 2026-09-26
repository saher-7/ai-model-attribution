import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from stylometric import extract_stylometric_features

SEED = 42

print("Loading data...")
train_df = pd.read_csv("data/splits/train.csv")
val_df = pd.read_csv("data/splits/val.csv")
test_df = pd.read_csv("data/splits/test.csv")

def build_feature_matrix(df):
    feats = [extract_stylometric_features(str(t)) for t in df["Text"]]
    feat_df = pd.DataFrame(feats)
    return feat_df

print("Extracting features (train)...")
X_train = build_feature_matrix(train_df)
y_train = train_df["Label_A"]

print("Extracting features (val)...")
X_val = build_feature_matrix(val_df)
y_val = val_df["Label_A"]

print("Extracting features (test)...")
X_test = build_feature_matrix(test_df)
y_test = test_df["Label_A"]

print("Training logistic regression...")
clf = LogisticRegression(max_iter=1000, random_state=SEED)
clf.fit(X_train, y_train)

val_preds = clf.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds)
print(f"Validation - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")

test_preds = clf.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)
print(f"Test - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")

os.makedirs("results/models", exist_ok=True)
joblib.dump(clf, "results/models/binary_detector.pkl")

metrics = {
    "val_accuracy": val_acc,
    "val_f1": val_f1,
    "test_accuracy": test_acc,
    "test_f1": test_f1,
    "seed": SEED
}
with open("results/metrics/binary.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model to results/models/binary_detector.pkl")
print("Saved metrics to results/metrics/binary.json")

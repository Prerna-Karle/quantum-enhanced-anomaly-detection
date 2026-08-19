"""
classical_model.py
-------------------
Baseline classical anomaly detector using scikit-learn's Random
Forest classifier, trained on the full 6-feature synthetic network
traffic dataset.

Run directly to train, evaluate, and save the model + metrics:
    python src/classical_model.py
"""

import json
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score,
                              roc_curve)

from preprocessing import scaled_split

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
MODEL_PATH = OUT_DIR / "models" / "classical_rf.joblib"
METRICS_PATH = OUT_DIR / "classical_metrics.json"


def train_and_evaluate(save=True):
    X_train, X_test, y_train, y_test, scaler = scaled_split()

    clf = RandomForestClassifier(
        n_estimators=150, max_depth=6, random_state=42, class_weight="balanced"
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - t0

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    metrics = {
        "model": "RandomForestClassifier",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_time_sec": round(train_time, 4),
        "inference_time_sec": round(inference_time, 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }

    if save:
        OUT_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": clf, "scaler": scaler}, MODEL_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved model -> {MODEL_PATH}")
        print(f"Saved metrics -> {METRICS_PATH}")

    return metrics, clf, (X_test, y_test, y_pred, y_proba)


if __name__ == "__main__":
    metrics, _, _ = train_and_evaluate()
    print(json.dumps({k: v for k, v in metrics.items()
                       if k not in ("roc_curve",)}, indent=2))

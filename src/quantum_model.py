"""
quantum_model.py
-----------------
Lightweight Quantum Machine Learning anomaly detector using Qiskit.

Approach: Quantum Kernel Support Vector Classification (QSVC).
    1. Reduce the 6 classical features to n_components (default 3)
       via PCA, so each feature maps to exactly one qubit.
    2. Encode each sample into a quantum state using a ZZFeatureMap
       (angle encoding + entangling ZZ interactions) -- this is what
       gives the kernel its "quantum" character, letting it capture
       feature correlations classical linear encodings can't reach
       as naturally.
    3. Compute a fidelity-based quantum kernel matrix between samples
       via the Qiskit Aer statevector simulator.
    4. Feed that kernel matrix into a standard scikit-learn-style SVM
       (QSVC wraps this) for classification.

This runs entirely on a classical *simulator* (no real quantum
hardware / QPU access needed), which is why the sample count is
deliberately capped -- kernel evaluation is O(n^2) circuit runs, and
we're optimizing for "runs on a laptop in under a minute", not
"scales to production traffic volumes". This is a research/learning
demonstration of the QML technique, not a production-scale claim.

Run directly to train, evaluate, and save the model + metrics:
    python src/quantum_model.py
"""

import json
import time
from pathlib import Path

import joblib
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score,
                              roc_curve)

from preprocessing import quantum_ready_split

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
MODEL_PATH = OUT_DIR / "models" / "quantum_qsvc.joblib"
METRICS_PATH = OUT_DIR / "quantum_metrics.json"

N_QUBITS = 3  # = n_components in PCA reduction; keep small for simulator speed


def build_quantum_kernel(n_qubits=N_QUBITS):
    feature_map = zz_feature_map(feature_dimension=n_qubits, reps=2, entanglement="linear")
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    return kernel


def train_and_evaluate(save=True):
    X_train, X_test, y_train, y_test, pca = quantum_ready_split(n_components=N_QUBITS)

    kernel = build_quantum_kernel()
    qsvc = QSVC(quantum_kernel=kernel)

    t0 = time.perf_counter()
    qsvc.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = qsvc.predict(X_test)
    inference_time = time.perf_counter() - t0

    # QSVC doesn't expose predict_proba by default; use decision_function
    # for an ROC curve, min-max normalized into a pseudo-probability.
    try:
        scores = qsvc.decision_function(X_test)
        s_min, s_max = scores.min(), scores.max()
        y_proba = (scores - s_min) / (s_max - s_min + 1e-9)
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    except Exception:
        roc_auc = None
        roc_curve_data = None

    metrics = {
        "model": "QSVC (Quantum Kernel SVM, ZZFeatureMap, 3 qubits)",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_time_sec": round(train_time, 4),
        "inference_time_sec": round(inference_time, 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_qubits": N_QUBITS,
        "roc_curve": roc_curve_data,
    }

    if save:
        OUT_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": qsvc, "pca": pca}, MODEL_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved model -> {MODEL_PATH}")
        print(f"Saved metrics -> {METRICS_PATH}")

    return metrics, qsvc, (X_test, y_test, y_pred)


if __name__ == "__main__":
    metrics, _, _ = train_and_evaluate()
    print(json.dumps({k: v for k, v in metrics.items()
                       if k not in ("roc_curve",)}, indent=2))

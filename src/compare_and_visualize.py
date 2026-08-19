"""
compare_and_visualize.py
-------------------------
Runs both the classical (Random Forest) and quantum (QSVC) pipelines,
then produces:
    1. A side-by-side metrics comparison table (JSON + printed).
    2. A bar chart comparing accuracy / precision / recall / F1.
    3. ROC curves for both models on one plot.
    4. A 2D PCA scatter plot of the dataset (normal vs anomalous),
       to visualize class separability.

Run:
    python src/compare_and_visualize.py

Note: this script re-trains both models by importing their
train_and_evaluate() functions, so it can simply be run standalone
after `pip install -r requirements.txt`.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from preprocessing import scaled_split
import classical_model
import quantum_model

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIG_DIR = OUT_DIR / "figures"
COMPARISON_PATH = OUT_DIR / "comparison_summary.json"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
})

CLASSICAL_COLOR = "#2E5EAA"
QUANTUM_COLOR = "#B23A48"


def run_pipelines():
    print("Training classical model (Random Forest)...")
    c_metrics, _, _ = classical_model.train_and_evaluate(save=True)

    print("Training quantum model (QSVC, this runs a simulator "
          "and takes ~20-40s)...")
    q_metrics, _, _ = quantum_model.train_and_evaluate(save=True)

    return c_metrics, q_metrics


def plot_metrics_bar(c_metrics, q_metrics):
    labels = ["accuracy", "precision", "recall", "f1_score"]
    c_vals = [c_metrics[k] for k in labels]
    q_vals = [q_metrics[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, c_vals, width, label="Classical (Random Forest)",
           color=CLASSICAL_COLOR)
    ax.bar(x + width / 2, q_vals, width, label="Quantum (QSVC)",
           color=QUANTUM_COLOR)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.18)
    ax.set_title("Classical vs Quantum Model Performance")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", " ").title() for l in labels])
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, v in enumerate(c_vals):
        ax.text(i - width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(q_vals):
        ax.text(i + width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    path = FIG_DIR / "metrics_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_roc_curves(c_metrics, q_metrics):
    fig, ax = plt.subplots(figsize=(6, 5.5))

    c_roc = c_metrics.get("roc_curve")
    if c_roc:
        ax.plot(c_roc["fpr"], c_roc["tpr"], color=CLASSICAL_COLOR, linewidth=2,
                 label=f"Classical RF (AUC={c_metrics['roc_auc']:.3f})")

    q_roc = q_metrics.get("roc_curve")
    if q_roc:
        ax.plot(q_roc["fpr"], q_roc["tpr"], color=QUANTUM_COLOR, linewidth=2,
                 label=f"Quantum QSVC (AUC={q_metrics['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
             label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Classical vs Quantum Anomaly Detector")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = FIG_DIR / "roc_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_pca_scatter():
    X_train, X_test, y_train, y_test, _ = scaled_split()
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    normal = y == 0
    anomaly = y == 1

    ax.scatter(X2[normal, 0], X2[normal, 1], c=CLASSICAL_COLOR, alpha=0.6,
               s=35, label="Normal traffic", edgecolors="white", linewidths=0.4)
    ax.scatter(X2[anomaly, 0], X2[anomaly, 1], c=QUANTUM_COLOR, alpha=0.75,
               s=35, label="Anomalous traffic", edgecolors="white", linewidths=0.4)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title("Network Traffic: 2D PCA Projection")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = FIG_DIR / "pca_class_separation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def save_summary(c_metrics, q_metrics):
    summary = {
        "classical": {k: v for k, v in c_metrics.items() if k != "roc_curve"},
        "quantum": {k: v for k, v in q_metrics.items() if k != "roc_curve"},
    }
    with open(COMPARISON_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {COMPARISON_PATH}")

    print("\n=== SUMMARY ===")
    print(f"{'Metric':<14}{'Classical':<14}{'Quantum':<14}")
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        cv = c_metrics.get(k)
        qv = q_metrics.get(k)
        cv_s = f"{cv:.3f}" if isinstance(cv, float) else str(cv)
        qv_s = f"{qv:.3f}" if isinstance(qv, float) else str(qv)
        print(f"{k:<14}{cv_s:<14}{qv_s:<14}")
    print(f"{'train_time(s)':<14}{c_metrics['train_time_sec']:<14}"
          f"{q_metrics['train_time_sec']:<14}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    c_metrics, q_metrics = run_pipelines()
    plot_metrics_bar(c_metrics, q_metrics)
    plot_roc_curves(c_metrics, q_metrics)
    plot_pca_scatter()
    save_summary(c_metrics, q_metrics)

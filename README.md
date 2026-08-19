# Quantum-Enhanced Anomaly Detection for Defence Cybersecurity

A lightweight, fully runnable demonstration project comparing a **classical
machine learning model** against a **quantum machine learning model** for
detecting anomalous (potentially malicious) network traffic — the kind of
task that sits at the core of intrusion detection systems (IDS) used to
protect networked infrastructure, including defence and government networks.

Built with **Python, Scikit-learn, and Qiskit**, runnable end-to-end on a
laptop CPU in under a minute using Qiskit's classical simulator (no quantum
hardware access required).

> **Scope note:** This is a research/learning-oriented demonstration, not a
> production intrusion detection system and not a benchmark of quantum
> advantage. See [Honest Limitations](#honest-limitations--scope) below.

---

## Why this matters for defence cybersecurity

Modern defence and critical-infrastructure networks are protected in part by
**anomaly-based intrusion detection**: systems that learn what "normal"
network behaviour looks like and flag deviations (port scans, denial-of-service
bursts, brute-force login attempts, data-exfiltration patterns) for review.
This is one layer of a broader defence-in-depth cybersecurity posture.

**Quantum machine learning (QML)** is an active area of research interest
for defence and national-security organisations (including DRDO) because
quantum kernels can, in principle, capture certain non-linear feature
correlations more naturally than some classical kernels — and because
building in-house QML competency now is a reasonable hedge as quantum
hardware matures. This project is a small, honest first step in that
direction: it does **not** claim quantum speed or accuracy advantage today
(on today's noisy, small-qubit-count hardware and simulators, it typically
doesn't have one for problems this size) — it demonstrates the *methodology*
end-to-end: data → classical baseline → quantum kernel model → fair,
side-by-side comparison.

---

## What this project does

1. **Generates a synthetic network-traffic dataset** (350 records, 6
   features: `duration`, `src_bytes`, `dst_bytes`, `error_rate`,
   `conn_count`, `packet_size`) with two classes — normal traffic and
   attack-like traffic (DoS-burst and scan/brute-force patterns) — with
   deliberately injected overlap so the task isn't trivially easy.
2. **Trains a classical baseline**: a Scikit-learn `RandomForestClassifier`
   on all 6 features.
3. **Trains a quantum model**: a Qiskit `QSVC` (Quantum Support Vector
   Classifier) using a `ZZFeatureMap` fidelity quantum kernel, on a
   PCA-reduced 3-feature (3-qubit) version of the data, run on Qiskit's
   classical statevector simulator.
4. **Compares both models fairly** on the same held-out test split
   (accuracy, precision, recall, F1, ROC-AUC, train/inference time).
5. **Generates 3 visualizations**: a metrics comparison bar chart, ROC
   curves for both models, and a 2D PCA scatter plot showing class
   separability.

---

## Project structure

```
quantum-enhanced-anomaly-detection/
├── main.py                      # One-command end-to-end pipeline runner
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   ├── generate_dataset.py      # Synthetic dataset generator
│   └── network_traffic.csv      # Generated dataset (350 records)
├── src/
│   ├── preprocessing.py         # Shared data loading / scaling / PCA split
│   ├── classical_model.py       # Random Forest baseline
│   ├── quantum_model.py         # Qiskit QSVC quantum kernel model
│   └── compare_and_visualize.py # Runs both models + builds charts
└── outputs/
    ├── classical_metrics.json
    ├── quantum_metrics.json
    ├── comparison_summary.json
    ├── models/
    │   ├── classical_rf.joblib
    │   └── quantum_qsvc.joblib
    └── figures/
        ├── metrics_comparison.png
        ├── roc_comparison.png
        └── pca_class_separation.png
```

---

## Setup & run instructions

### 1. Clone / extract and open in VS Code

```bash
cd quantum-enhanced-anomaly-detection
code .
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python main.py
```

This will (re)generate the dataset, train both models, print a metrics
summary to the console, and save all outputs (models, metrics, charts) into
`outputs/`. Total runtime is roughly **30–60 seconds** on a normal laptop
CPU — the quantum kernel step is the slowest part by design (see
[Honest Limitations](#honest-limitations--scope)).

### Running individual pieces

You can also run each stage independently from inside `src/`:

```bash
cd data && python generate_dataset.py      # regenerate the dataset only
cd src  && python classical_model.py       # classical model only
cd src  && python quantum_model.py         # quantum model only (~20-40s)
cd src  && python compare_and_visualize.py # both models + all charts
```

---

## Results (example run)

Exact numbers vary slightly by random seed / environment, but a typical run
looks like this:

| Metric          | Classical (Random Forest) | Quantum (QSVC, 3-qubit) |
|------------------|:--------------------------:|:-------------------------:|
| Accuracy         | 0.971                      | 0.867                     |
| Precision        | 1.000                      | 0.714                     |
| Recall           | 0.889                      | 0.714                     |
| F1 Score         | 0.941                      | 0.714                     |
| ROC-AUC          | 0.997                      | 0.932                     |
| Train time (s)   | ~0.2                       | ~7                        |
| Test set size    | 105 samples                | 30 samples (capped, see below) |

**Interpretation:** the classical Random Forest outperforms the quantum
kernel model here — which is expected and honestly reported. Classical
ensemble methods are extremely strong on small, structured tabular data
like this. The value of this project isn't "quantum wins" — it's a working,
reproducible pipeline that shows *how* a quantum ML model is built, trained,
and fairly benchmarked against a classical one, which is the actual skill
being demonstrated.

### Visualizations

- `outputs/figures/metrics_comparison.png` — accuracy/precision/recall/F1
  bar chart, classical vs quantum.
- `outputs/figures/roc_comparison.png` — ROC curves for both models.
- `outputs/figures/pca_class_separation.png` — 2D PCA projection of the
  dataset showing how separable normal vs anomalous traffic is.

---

## How the quantum model works (brief)

1. The 6 classical features are reduced to **3 principal components** via
   PCA (so each feature maps to exactly one qubit).
2. Each sample is encoded into a 3-qubit quantum state using a
   **`ZZFeatureMap`** — this applies single-qubit rotations proportional to
   each feature value, plus entangling `ZZ` interactions between qubits,
   so the resulting quantum state depends on feature *correlations*, not
   just individual values.
3. A **fidelity quantum kernel** measures the "overlap" (similarity)
   between every pair of encoded quantum states, computed via
   `FidelityQuantumKernel` on Qiskit's statevector simulator.
4. That kernel matrix is fed into a standard **Support Vector Classifier**
   (`QSVC`), exactly like a classical SVM would use an RBF or polynomial
   kernel — except this kernel is computed quantum-mechanically.

This is the standard "quantum kernel method" approach to QML — one of the
more mature and NISQ-era-realistic techniques, as opposed to deeper
variational quantum circuits which tend to be harder to train.

---

## Honest limitations & scope

This project is intentionally scoped to be **lightweight, fast, and
runnable on a laptop simulator** — which comes with real trade-offs that
are important to state plainly:

- **Synthetic data, not real traffic.** The dataset is generated to
  *resemble* the statistical shape of network intrusion features (as seen
  in public IDS datasets like NSL-KDD), but it is not captured real-world
  traffic and should not be treated as such.
- **Small sample size for the quantum model.** Quantum kernel evaluation
  on a simulator costs roughly O(n²) circuit executions. To keep runtime
  under a minute, the quantum model is trained/tested on a capped
  60-train / 30-test subsample rather than the full dataset. This is a
  deliberate scope choice for a fast demo, not a claim about how the
  method would scale.
- **No claim of quantum advantage.** On problems this small, classical
  models are expected to perform as well as or better than quantum kernel
  methods — that's consistent with the current state of QML research, not
  a flaw in this implementation. The project's purpose is to demonstrate
  the pipeline and methodology, not to claim a speed or accuracy edge.
- **Simulator, not real quantum hardware.** All quantum circuits run on
  Qiskit's classical statevector simulator. Real NISQ hardware would
  introduce additional noise and would very likely change these results.
- **Not a production IDS.** There's no real-time packet capture, no
  streaming pipeline, and no adversarial-robustness testing here — this is
  a model-comparison research demo, not a deployable detection system.

---

## Possible extensions

- Swap in a real public IDS dataset (e.g. a subsampled NSL-KDD or
  CICIDS2017) once feature engineering is aligned.
- Try a Variational Quantum Classifier (VQC) as an additional quantum
  baseline alongside QSVC.
- Add cross-validation and hyperparameter search for the classical model.
- Explore quantum autoencoders for unsupervised anomaly scoring instead of
  supervised classification.

---

## Tech stack

- **Python 3.10+**
- **Scikit-learn** — Random Forest classifier, PCA, scaling, metrics
- **Qiskit** + **Qiskit Machine Learning** — ZZFeatureMap, FidelityQuantumKernel, QSVC
- **Pandas / NumPy** — data handling
- **Matplotlib** — visualizations

Tested with: `qiskit 2.5.2`, `qiskit-machine-learning 0.9.0`,
`scikit-learn 1.8.0`, Python 3.12.

---

## License

MIT — see [LICENSE](LICENSE).

## Author

Prerna — B.Tech CSE-AI, G H Raisoni College of Engineering and Management, Pune.
Built as part of an internship application demonstrating applied ML +
quantum computing fundamentals for cybersecurity use cases.

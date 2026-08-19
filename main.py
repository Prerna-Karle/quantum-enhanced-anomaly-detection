"""
main.py
--------
End-to-end pipeline runner for the Quantum-Enhanced Anomaly Detection
for Defence Cybersecurity project.

Running this single script will:
    1. (Re)generate the synthetic network-traffic dataset.
    2. Train + evaluate the classical Random Forest baseline.
    3. Train + evaluate the quantum QSVC model (Qiskit simulator).
    4. Generate all comparison visualizations.
    5. Print a final summary to the console.

Usage:
    python main.py

Expect total runtime of roughly 30-60 seconds on a normal laptop CPU
(the quantum kernel simulation is the slowest step, by design capped
to a small sample count -- see src/preprocessing.py for why).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))


def main():
    print("=" * 60)
    print("Quantum-Enhanced Anomaly Detection for Defence Cybersecurity")
    print("=" * 60)

    print("\n[1/4] Generating synthetic dataset...")
    import generate_dataset
    df = generate_dataset.build_dataframe()
    out_path = ROOT / "data" / "network_traffic.csv"
    df.to_csv(out_path, index=False)
    print(f"      -> {len(df)} records saved to {out_path}")

    print("\n[2/4] Running classical + quantum pipelines "
          "and generating visualizations...")
    import compare_and_visualize
    compare_and_visualize.FIG_DIR.mkdir(parents=True, exist_ok=True)
    c_metrics, q_metrics = compare_and_visualize.run_pipelines()

    print("\n[3/4] Building comparison charts...")
    compare_and_visualize.plot_metrics_bar(c_metrics, q_metrics)
    compare_and_visualize.plot_roc_curves(c_metrics, q_metrics)
    compare_and_visualize.plot_pca_scatter()

    print("\n[4/4] Saving summary...")
    compare_and_visualize.save_summary(c_metrics, q_metrics)

    print("\nDone. See outputs/figures/ for charts and outputs/*.json for metrics.")


if __name__ == "__main__":
    main()

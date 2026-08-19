"""
generate_dataset.py
--------------------
Generates a small, synthetic network-traffic dataset for anomaly
(intrusion) detection, loosely inspired by the feature style of
public IDS datasets such as NSL-KDD / CICIDS (duration, byte counts,
error rates, connection counts, etc.) but produced synthetically so
the project is fully self-contained and license-free.

The dataset simulates two classes of network connection records:
    0 -> normal traffic
    1 -> anomalous / attack-like traffic (e.g. port scan, DoS burst,
         brute-force login pattern)

This is NOT real captured traffic and NOT a claim of operational
accuracy — it exists purely to demonstrate a classical-vs-quantum
anomaly detection pipeline end to end.

Run:
    python generate_dataset.py
Produces:
    network_traffic.csv  (in this same folder)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_NORMAL = 260
N_ANOMALY = 90  # anomalies are the minority class, as in real traffic
NOISE_FRACTION = 0.12  # fraction of records perturbed to overlap the other class,
                        # so the dataset isn't trivially/perfectly separable

rng = np.random.default_rng(RANDOM_SEED)


def make_normal_traffic(n):
    """Well-behaved connections: short-moderate duration, balanced
    byte counts, low error rate, few connections per source."""
    duration = rng.gamma(shape=2.0, scale=1.5, size=n)                 # seconds
    src_bytes = rng.normal(loc=500, scale=120, size=n).clip(min=20)    # bytes sent
    dst_bytes = rng.normal(loc=450, scale=110, size=n).clip(min=20)    # bytes received
    error_rate = rng.beta(a=1.5, b=20, size=n)                        # ~low
    conn_count = rng.poisson(lam=3, size=n).clip(min=1)               # conns/host in window
    packet_size = rng.normal(loc=350, scale=60, size=n).clip(min=40)
    return duration, src_bytes, dst_bytes, error_rate, conn_count, packet_size


def make_anomalous_traffic(n):
    """Attack-like connections: mix of DoS-burst (very short duration,
    huge connection counts) and scan/brute-force (high error rate,
    tiny payloads)."""
    # split into two attack sub-patterns for realism
    n_dos = n // 2
    n_scan = n - n_dos

    # DoS-burst pattern
    duration_dos = rng.exponential(scale=0.3, size=n_dos)
    src_bytes_dos = rng.normal(loc=60, scale=25, size=n_dos).clip(min=1)
    dst_bytes_dos = rng.normal(loc=40, scale=20, size=n_dos).clip(min=1)
    error_dos = rng.beta(a=2, b=6, size=n_dos)
    conn_dos = rng.poisson(lam=45, size=n_dos).clip(min=10)
    pkt_dos = rng.normal(loc=90, scale=30, size=n_dos).clip(min=10)

    # Scan / brute-force pattern
    duration_scan = rng.exponential(scale=0.15, size=n_scan)
    src_bytes_scan = rng.normal(loc=25, scale=10, size=n_scan).clip(min=1)
    dst_bytes_scan = rng.normal(loc=15, scale=8, size=n_scan).clip(min=1)
    error_scan = rng.beta(a=6, b=3, size=n_scan)  # high error/rejection rate
    conn_scan = rng.poisson(lam=20, size=n_scan).clip(min=5)
    pkt_scan = rng.normal(loc=55, scale=20, size=n_scan).clip(min=5)

    duration = np.concatenate([duration_dos, duration_scan])
    src_bytes = np.concatenate([src_bytes_dos, src_bytes_scan])
    dst_bytes = np.concatenate([dst_bytes_dos, dst_bytes_scan])
    error_rate = np.concatenate([error_dos, error_scan])
    conn_count = np.concatenate([conn_dos, conn_scan])
    packet_size = np.concatenate([pkt_dos, pkt_scan])
    return duration, src_bytes, dst_bytes, error_rate, conn_count, packet_size


def build_dataframe():
    d0 = make_normal_traffic(N_NORMAL)
    d1 = make_anomalous_traffic(N_ANOMALY)

    cols = ["duration", "src_bytes", "dst_bytes", "error_rate",
            "conn_count", "packet_size"]

    df0 = pd.DataFrame(dict(zip(cols, d0)))
    df0["label"] = 0

    df1 = pd.DataFrame(dict(zip(cols, d1)))
    df1["label"] = 1

    df = pd.concat([df0, df1], ignore_index=True)
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    df[cols] = df[cols].astype(float)

    # Inject realistic overlap: nudge a fraction of each class partway
    # toward the "typical" feature values of the opposite class. Real
    # traffic is noisy and borderline cases exist (e.g. a slow scan
    # that looks almost normal) — this keeps the task non-trivial
    # instead of a perfectly separable toy problem.
    n_noisy = int(len(df) * NOISE_FRACTION)
    noisy_idx = rng.choice(df.index, size=n_noisy, replace=False)

    normal_means = df.loc[df.label == 0, cols].mean()
    anomaly_means = df.loc[df.label == 1, cols].mean()

    for idx in noisy_idx:
        row_label = df.loc[idx, "label"]
        target_means = anomaly_means if row_label == 0 else normal_means
        blend = rng.uniform(0.35, 0.6)  # how far to pull toward the other class
        for c in cols:
            df.loc[idx, c] = (1 - blend) * df.loc[idx, c] + blend * target_means[c]

    # round for readability
    for c in ["duration", "error_rate"]:
        df[c] = df[c].round(4)
    for c in ["src_bytes", "dst_bytes", "packet_size"]:
        df[c] = df[c].round(1)

    return df


if __name__ == "__main__":
    df = build_dataframe()
    out_path = Path(__file__).parent / "network_traffic.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} records ({(df.label==0).sum()} normal, "
          f"{(df.label==1).sum()} anomalous) -> {out_path}")

"""
preprocessing.py
-----------------
Shared data loading, splitting, and scaling utilities used by both
the classical and quantum pipelines, so that both models are always
trained/evaluated on an identical, fair split of the data.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "network_traffic.csv"

FEATURE_COLS = ["duration", "src_bytes", "dst_bytes",
                 "error_rate", "conn_count", "packet_size"]
LABEL_COL = "label"

RANDOM_SEED = 42


def load_raw():
    df = pd.read_csv(DATA_PATH)
    return df


def load_split(test_size=0.3, random_state=RANDOM_SEED):
    """Returns X_train, X_test, y_train, y_test as numpy arrays
    (unscaled, full 6-feature set)."""
    df = load_raw()
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scaled_split(test_size=0.3, random_state=RANDOM_SEED):
    """Full 6-feature set, scaled to [0, 1] (fit on train only)."""
    X_train, X_test, y_train, y_test = load_split(test_size, random_state)
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler


def quantum_ready_split(n_components=3, test_size=0.3, random_state=RANDOM_SEED,
                         n_train_cap=60, n_test_cap=30):
    """
    Produces a reduced-dimensionality, scaled dataset suitable for a
    quantum kernel running on a classical simulator.

    Quantum kernel evaluation costs scale roughly with O(n^2) circuit
    executions on a simulator, so the *sample count* is deliberately
    capped (n_train_cap / n_test_cap) to keep runtime lightweight for
    a laptop/simulator demo. The *feature count* is reduced via PCA
    to n_components, since each feature maps to one qubit in the
    feature map used here (see quantum_model.py) — keeping qubit
    count low keeps simulation fast.

    This subsampling is a deliberate scope choice for a lightweight
    demo, not a claim about production-scale quantum ML.
    """
    X_train_s, X_test_s, y_train, y_test, scaler = scaled_split(test_size, random_state)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)

    # re-scale PCA output to [0, 2*pi) range, friendly for angle encoding
    qscaler = MinMaxScaler(feature_range=(0, 2 * 3.14159265))
    X_train_q = qscaler.fit_transform(X_train_p)
    X_test_q = qscaler.transform(X_test_p)

    rng_state = random_state
    import numpy as np
    rng = np.random.default_rng(rng_state)

    if len(X_train_q) > n_train_cap:
        idx = rng.choice(len(X_train_q), size=n_train_cap, replace=False)
        X_train_q, y_train_q = X_train_q[idx], y_train[idx]
    else:
        y_train_q = y_train

    if len(X_test_q) > n_test_cap:
        idx = rng.choice(len(X_test_q), size=n_test_cap, replace=False)
        X_test_q, y_test_q = X_test_q[idx], y_test[idx]
    else:
        y_test_q = y_test

    return X_train_q, X_test_q, y_train_q, y_test_q, pca

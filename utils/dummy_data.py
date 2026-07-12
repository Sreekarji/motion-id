"""
Synthetic data for pipeline shape testing.
Run from D:\motionid\: python utils/dummy_data.py

Expected output:
  MPI dummy — X: (200, 18, 150), y: (200,), user_ids: (200,)
    Positive rate: 0.30
  UV dummy  — X: (101, 300, 22, 4, 50), cluster_ids: (101, 300)
    Unique clusters per user: [0 1 2 3 4 5]
"""

import numpy as np
import os

DUMMY_DIR = "data/dummy"


def generate_mpi_dummy(n_samples=200, n_channels=18, T=150, seed=42):
    rng      = np.random.default_rng(seed)
    X        = rng.standard_normal((n_samples, n_channels, T)).astype(np.float32)
    y        = (rng.random(n_samples) < 0.3).astype(np.int64)
    user_ids = rng.integers(0, 6, size=n_samples).astype(int)
    return X, y, user_ids


def generate_uv_dummy(n_users=101, n_trials=300, n_features=22,
                      n_channels=4, T=50, seed=42):
    rng         = np.random.default_rng(seed)
    X           = rng.standard_normal(
                    (n_users, n_trials, n_features, n_channels, T)
                  ).astype(np.float32)
    cluster_ids = np.tile(np.repeat(np.arange(6), 50), (n_users, 1)).astype(int)
    return X, cluster_ids


def save_dummy_data():
    os.makedirs(DUMMY_DIR, exist_ok=True)
    X_mpi, y_mpi, uid_mpi = generate_mpi_dummy()
    np.save(os.path.join(DUMMY_DIR, "mpi_X.npy"),        X_mpi)
    np.save(os.path.join(DUMMY_DIR, "mpi_y.npy"),        y_mpi)
    np.save(os.path.join(DUMMY_DIR, "mpi_user_ids.npy"), uid_mpi)
    print(f"MPI dummy — X: {X_mpi.shape}, y: {y_mpi.shape}, "
          f"user_ids: {uid_mpi.shape}")
    print(f"  Positive rate: {y_mpi.mean():.2f}")
    X_uv, cl_uv = generate_uv_dummy()
    np.save(os.path.join(DUMMY_DIR, "uv_X.npy"),           X_uv)
    np.save(os.path.join(DUMMY_DIR, "uv_cluster_ids.npy"), cl_uv)
    print(f"UV dummy  — X: {X_uv.shape}, cluster_ids: {cl_uv.shape}")
    print(f"  Unique clusters per user: {np.unique(cl_uv[0])}")


def load_mpi_dummy():
    return (np.load(os.path.join(DUMMY_DIR, "mpi_X.npy")),
            np.load(os.path.join(DUMMY_DIR, "mpi_y.npy")),
            np.load(os.path.join(DUMMY_DIR, "mpi_user_ids.npy")))


def load_uv_dummy():
    return (np.load(os.path.join(DUMMY_DIR, "uv_X.npy")),
            np.load(os.path.join(DUMMY_DIR, "uv_cluster_ids.npy")))


if __name__ == "__main__":
    save_dummy_data()

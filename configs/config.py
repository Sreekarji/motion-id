from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── MPI (paper Section 4.1.1 + 4.3.1) ───────────────────────────────
    mpi_sampling_rate: int = 50
    mpi_window_sec: float = 3.0          # paper: "gathered data over 3 seconds"
    mpi_min_readings: int = 100          # paper: "at least 100 readings for each sensor"
    mpi_n_channels: int = 18            # 6 sensors × 3 axes
    mpi_epochs: int = 30
    stationary_threshold: float = 0.01  # near-zero lin_acc (m/s²)

    # ── UV (paper Section 4.1.2 + 4.3.2) ────────────────────────────────
    uv_sampling_rate: int = 50
    uv_window_sec: float = 1.0           # paper: "only interested in last second"
    uv_augment_window_sec: float = 1.5  # paper: "1.5 seconds randomly cut to 1s"
    uv_n_features: int = 22             # paper: "total number of feature vectors was 22"
    uv_n_channels_per_feature: int = 4  # 3-ch features zero-padded to 4
    uv_n_trials: int = 300              # paper: 300 lifts per user
    uv_n_clusters: int = 6             # paper: 6 locations
    uv_total_users: int = 101
    uv_test_users: int = 11

    # ── Training (paper Section 6.5) ─────────────────────────────────────
    uv_baseline_n: int = 75
    uv_n_splits: List[int] = field(
        default_factory=lambda: [60, 65, 70, 75, 80, 85])
    baseline_lr: float = 1e-3           # not in paper
    finetune_lr: float = 1e-4           # not in paper
    baseline_epochs: int = 50           # not in paper
    finetune_epochs: int = 10           # not in paper
    batch_size: int = 64                # not in paper
    alpha_tm: float = 1.0               # not in paper
    supcon_temperature: float = 0.07    # Khosla et al. 2020 default
    n_training_runs: int = 5            # paper trains 5 times, reports mean±std

    # ── Evaluation (paper Section 4.2 + 7) ───────────────────────────────
    tar_threshold: float = 0.90         # paper: TAR(@FAR=1/50000)=90%
    target_far: float = 1 / 50000       # paper: Android CDD Class 3
    bootstrap_repeats: int = 5000       # paper: 5000 repeats
    far_sweep_steps: int = 100_000


cfg = Config()

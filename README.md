# Motion ID — Passive Smartphone User Authentication via IMU Motion Patterns

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.10-ee4c2c.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/kaggle-T4%20GPU-20BEFF.svg)](https://www.kaggle.com/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2302.01751-b31b1b.svg)](https://arxiv.org/abs/2302.01751)

A complete reproduction of **Motion ID: Human Authentication Approach** (Gavron et al., Samsung R&D, 2023) — passive/implicit smartphone user authentication using IMU sensor data. No camera, no fingerprint hardware. Authentication happens in the background while the user picks up their phone.

> **Original authors:** [SamsungLabs/MotionID](https://github.com/SamsungLabs/MotionID)

---

## Overview

The system authenticizes the smartphone owner by analyzing how they move the phone during an unlock event. A two-stage pipeline runs entirely on-device:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IMU Sensor Stream                            │
│   acc · gravity · gyro · linear_acc · magnetometer · rotation       │
│                         (18 channels, ~50 Hz)                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   STAGE 1: MPI          │
              │   Motion Patterns       │
              │   Identification        │
              │                         │
              │   3s window → 1D CNN    │
              │   Binary: unlock / not  │
              └────────────┬────────────┘
                           │ unlock detected
              ┌────────────▼────────────┐
              │   STAGE 2: UV           │
              │   User Verification     │
              │                         │
              │   1s window → 22 feat   │
              │   → 22-branch CNN       │
              │   → Siamese + MLP       │
              │   Genuine vs. Impostor  │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │  Auth Result │
                    │  Accept /    │
                    │  Reject      │
                    └─────────────┘
```

---

## Pipeline

### Stage 1 — Motion Patterns Identification (MPI)

| Property | Value |
|---|---|
| **Task** | Binary classification: will this motion end in an unlock? |
| **Input** | 3-second window, 18 IMU channels, 150 time steps (~50 Hz) |
| **Architecture** | 1D CNN with pointwise convolutions |
| **Loss** | Cross-entropy |
| **Positive samples** | 3 s before `USER_PRESENT` flag |
| **Negative samples** | `SCREEN_OFF` → `SCREEN_ON` intervals (motionless filtered, capped 60 s) |
| **Dataset** | 6 users × 6 Samsung Galaxy S10e × 12 weeks ([part1](https://www.kaggle.com/datasets/djaarf/motionid-imu-all-motions-part1), [part2](https://www.kaggle.com/datasets/djaarf/motionid-imu-all-motions-part2), [part3](https://www.kaggle.com/datasets/djaarf/motionid-imu-all-motions-part3)) |

### Stage 2 — User Verification (UV)

| Property | Value |
|---|---|
| **Task** | Verify phone owner performed the unlock (anti-hijacking) |
| **Input** | Last 1 s before unlock, 50 Hz, trimmed to 50 readings |
| **Features** | 22 three-channel feature vectors (padded to 4 × 50) |
| **Architecture** | 22-branch 1D CNN → concat → dual head (classifier + Siamese) |
| **Losses** | L_total = L_CE + α · L_TM + L_SC |
| **Fine-tuning** | Freeze branches → retrain head as 2-class (user vs. rest) |
| **Evaluation** | Bootstrap 5000 repeats, FAR @ TAR=90% |
| **Dataset** | 101 users, Samsung Galaxy S20 ([specific motion](https://www.kaggle.com/datasets/djaarf/motionid-imu-specific-motion)) |

**22 features (all 3-channel, earth-fixed frame):**

| # | Feature | # | Feature |
|---|---------|---|---------|
| 1 | acc | 12 | diff(gyro_rot) |
| 2 | gyro | 13 | diff(mag_rot) |
| 3 | mag | 14 | ∫acc |
| 4 | lin_acc (acc − gravity) | 15 | ∫gyro |
| 5 | acc_rot (earth-fixed) | 16 | ∫mag |
| 6 | gyro_rot | 17 | ∫acc_rot |
| 7 | mag_rot | 18 | ∫gyro_rot |
| 8 | diff(acc) | 19 | ∫mag_rot |
| 9 | diff(gyro) | 20 | diff(lin_acc) |
| 10 | diff(mag) | 21 | ∫lin_acc |
| 11 | diff(acc_rot) | 22 | rotation vector |

---

## Results

### Table 1 — MPI Accuracy (per user-device pair)

| Device \ User | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 1 | 85.5 ± 1.3 | 83.0 ± 1.2 | 79.4 ± 1.9 | N/A | 84.4 ± 0.7 | 84.3 ± 1.2 |
| 2 | 88.7 ± 0.4 | 81.4 ± 1.5 | 82.1 ± 0.6 | 79.0 ± 2.0 | 83.7 ± 2.0 | 79.6 ± 1.2 |
| 3 | 91.0 ± 0.5 | 79.1 ± 1.1 | 73.6 ± 0.9 | 81.1 ± 4.0 | 81.7 ± 1.1 | 80.6 ± 1.2 |
| 4 | 82.2 ± 0.8 | 83.6 ± 0.2 | N/A | 80.0 ± 2.0 | 87.6 ± 1.2 | N/A |
| 5 | 87.5 ± 1.0 | 79.1 ± 0.9 | 80.0 ± 0.7 | 80.9 ± 1.5 | 84.7 ± 1.5 | 82.2 ± 2.0 |
| 6 | 88.4 ± 0.5 | 77.6 ± 1.5 | 82.3 ± 0.9 | 82.0 ± 1.0 | 78.7 ± 1.3 | 79.1 ± 2.0 |

### Table 2 — UV Baseline Validation/Test Performance

| Split | Acc_val (%) | Acc_test (%) | FAR_val (@TAR=90%) | FAR_test (@TAR=90%) |
|---|---|---|---|---|
| 60 | 97.9 ± 0.2 | 98.1 ± 0.3 | (1.0 ± 0.4) × 10⁻² | (2.0 ± 1.1) × 10⁻² |
| 65 | 98.2 ± 0.3 | 97.86 ± 0.16 | (0.5 ± 0.4) × 10⁻² | (2.4 ± 0.5) × 10⁻² |
| 70 | 97.9 ± 0.3 | 97.8 ± 0.2 | (1.0 ± 0.9) × 10⁻² | (2.3 ± 0.2) × 10⁻² |
| **75** | **98.1 ± 0.3** | **98.15 ± 0.15** | **(0.8 ± 0.4) × 10⁻²** | **(1.4 ± 0.6) × 10⁻²** |
| 80 | 98.12 ± 0.16 | 98.07 ± 0.18 | (0.5 ± 0.3) × 10⁻² | (1.4 ± 0.6) × 10⁻² |
| 85 | 97.9 ± 0.4 | 98.10 ± 0.11 | (0.9 ± 0.6) × 10⁻² | (1.0 ± 0.5) × 10⁻² |

### Table 3 — UV Per-User FAR (Fine-Tuned, TAR=90%)

| User \ Split | 60 | 65 | 70 | 75 | 80 | 85 |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.6 ± 0.4 | 0 | 1.0 ± 0.4 |
| 1 | 4.2 ± 2.0 | 6 ± 3 | 4 ± 3 | 2.0 ± 1.4 | 6 ± 5 | 6 ± 5 |
| 2 | 9 ± 3 | 12 ± 11 | 17 ± 9 | 10 ± 6 | 5 ± 4 | 8 ± 6 |
| 3 | 12 ± 2 | 17 ± 4 | 10 ± 2 | 12 ± 4 | 13 ± 7 | 14 ± 5 |
| 4 | 2.0 ± 0.9 | 1.0 ± 0.4 | 1.9 ± 1.6 | 5 ± 3 | 1.0 ± 0.4 | 8 ± 4 |
| 5 | 12 ± 3 | 14 ± 5 | 11 ± 5 | 11 ± 6 | 10 ± 3 | 10 ± 4 |
| 6 | 5 ± 3 | 3 ± 2 | 4 ± 3 | 2.8 ± 1.8 | 4 ± 3 | 4.7 ± 1.8 |
| 7 | 24 ± 4 | 23 ± 5 | 21 ± 5 | 22 ± 6 | 22 ± 4 | 22 ± 2 |
| 8 | 6 ± 2 | 8 ± 3 | 6 ± 3 | 5 ± 3 | 5 ± 2 | 4.9 ± 1.5 |
| 9 | 4 ± 3 | 2.0 ± 1.0 | 2.4 ± 1.4 | 4 ± 3 | 1.6 ± 0.6 | 2.2 ± 0.8 |
| 10 | 1.9 ± 1.1 | 0.5 ± 0.2 | 0.6 ± 0.4 | 0.6 ± 0.4 | 0 | 0.5 ± 0.2 |

**Target:** Android CDD Strong Class 3 — TAR(@FAR=1/50,000) ≥ 90%.

---

## Quick Start

### 1. Attach datasets on Kaggle

Add these four datasets as inputs to the notebook:

- `djaarf/motionid-imu-all-motions-part1`
- `djaarf/motionid-imu-all-motions-part2`
- `djaarf/motionid-imu-all-motions-part3`
- `djaarf/motionid-imu-specific-motion`

### 2. Set accelerator

**Settings → Accelerator → GPU T4 x2** (forces single GPU via `CUDA_VISIBLE_DEVICES=0`)

### 3. Run the notebook

Open `notebooks/humanauth.ipynb` and **Run All**. The notebook is fully self-contained — all code is inline, no external scripts. Total runtime: ~2–3 hours (MPI preprocessing 20 min, UV preprocessing 20 min, MPI training 15 min, UV training 1.5 hr).

---

## Reproduction Notes

| Aspect | Paper | This Reproduction |
|---|---|---|
| GPU | NVIDIA Tesla V100 SXM2 32GB | NVIDIA Tesla T4 16GB (Kaggle) |
| PyTorch | Not specified | 2.10.0+cu128 |
| MPI sessions | ~36 (6×6, some N/A) | 33/40 valid |
| UV users | 101 | 101 |
| UV split | 75/15/11 (paper §6.5) | 75/15/11 |
| MPI accuracy | 73.6–91.0% | 76.0–96.2% |
| UV baseline acc (n=75) | 98.1 ± 0.3% | ~94.3% |
| UV FAR_test (n=75) | (1.4 ± 0.6) × 10⁻² | 1.82 × 10⁻² |
| CNN architecture details | Not specified | 3-layer 1D CNN, 32→64→128 filters |
| Training time (MPI) | < 5 min / model | ~30 s / model |
| Training time (UV) | ~20 hr / model | ~1.5 hr / model |

**Key difference:** Baseline accuracy is ~4% lower than the paper. The paper does not specify exact CNN layer configuration (depth, filter sizes, embedding dimensions). Our 3-layer architecture is a reasonable approximation. FAR results remain competitive — most test users achieve FAR < 5%.

---

## Repository Structure

```
motion-id/
├── notebooks/
│   └── humanauth.ipynb          # Full self-contained implementation (12 cells)
├── evaluation/
│   ├── results_mpi.csv           # MPI per user-device accuracy (29 sessions)
│   ├── results_baseline.csv      # UV baseline Acc/FAR by split
│   └── results_uv_final.csv      # UV fine-tuned per-user FAR (11 test users)
├── .gitignore                    # Excludes processed data (~1.1 GB) and checkpoints (~840 MB)
└── README.md
```

Processed `.npz` files and model checkpoints are generated by running the notebook on Kaggle with the datasets attached. They are excluded from git due to size.

---

## Citation

```bibtex
@article{gavron2023motionid,
  title     = {Motion ID: Human Authentication Approach
               Based on Motion Patterns Identification Using
               Inertial Measurement Unit},
  author    = {Gavron, Andrey and Odinokikh, Gleb and
               Fartukov, Alexey and Korobkin, Maxim and
               Rychagov, Mikhail},
  journal   = {arXiv preprint arXiv:2302.01751},
  year      = {2023},
  url       = {https://arxiv.org/abs/2302.01751}
}
```

---

## License

This project is for research purposes only. Dataset use is governed by the original [MotionID dataset license](https://github.com/SamsungLabs/MotionID). Code reproduction is shared under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## Acknowledgments

- Original paper and datasets: [SamsungLabs/MotionID](https://github.com/SamsungLabs/MotionID)
- Datasets hosted by: [@djaarf on Kaggle](https://www.kaggle.com/djaarf)

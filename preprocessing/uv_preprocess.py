"""
UV dataset preprocessing.
Paper: Sections 4.1.2 and 4.3.2 of arXiv:2302.01751

Input: Kaggle dataset mounted at:
  /kaggle/input/motionid-imu-specific-motion/IMU_specific_motion/train_val_test/

Directory structure:
  {user_id}/s20/{user_id}_20000/
    accel.bin, gravity.bin, gyro.bin, linAccel.bin, MagneticField.bin,
    Rotation.bin, screen.txt

Run: python preprocessing/uv_preprocess.py --use-dummy
"""

import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import cfg
from utils.bin_reader import load_session, FLAG_USER_PRESENT
from preprocessing.feature_inventory import FEATURE_LIST

UV_INPUT_DIR  = ("/kaggle/input/datasets/djaarf/motionid-imu-specific-motion"
                 "/IMU_specific_motion/train_val_test")
PROCESSED_DIR = "data/uv/processed"


def rotate_to_earth(v: np.ndarray, rot_vec: np.ndarray) -> np.ndarray:
    """
    Paper Section 6.1: convert device-frame readings to Earth-fixed frame.
    v:       (T, 3) — sensor readings in device frame
    rot_vec: (T, 3) — rotation vector (azimuth, pitch, roll) from Rotation sensor
    Returns: (T, 3) in Earth-fixed frame
    Uses scipy R.from_rotvec() — handles small/zero norms automatically.
    """
    return R.from_rotvec(rot_vec).apply(v).astype(np.float32)


def extract_window(merged_df: pd.DataFrame, unlock_ts: int):
    """
    Paper 4.3.2: last 1 second of data before unlock.
    Returns DataFrame of exactly uv_sampling_rate rows, or None.
    """
    n         = int(cfg.uv_window_sec * cfg.uv_sampling_rate)
    window_ms = int(cfg.uv_window_sec * 1000)
    sub       = (merged_df[
                    (merged_df["timestamp"] <= unlock_ts) &
                    (merged_df["timestamp"] >  unlock_ts - window_ms)]
                 .sort_values("timestamp")
                 .tail(n))
    return sub.reset_index(drop=True) if len(sub) == n else None


def pad4(x: np.ndarray, T: int) -> np.ndarray:
    """(T, 3) → (4, T): zero-pad 4th channel."""
    out        = np.zeros((4, T), dtype=np.float32)
    out[:3, :] = x.T
    return out


def diff(x: np.ndarray) -> np.ndarray:
    """Consecutive differences, preserving length T."""
    return np.concatenate([x[:1], np.diff(x, axis=0)], axis=0)


def compute_features(window: pd.DataFrame) -> np.ndarray:
    """
    Paper Section 6.1: 22 feature vectors.
    Returns shape (22, 4, T).
    All features are 3-channel, zero-padded to 4 for uniform branch input.
    Feature order matches FEATURE_LIST exactly.

    Earth-fixed frame uses R.from_rotvec(rot) where rot is the
    rotation vector (azimuth, pitch, roll) from the Android Rotation sensor.
    """
    T    = len(window)
    acc  = window[["acc_X",  "acc_Y",  "acc_Z"]].values.astype(np.float32)
    grav = window[["grav_X", "grav_Y", "grav_Z"]].values.astype(np.float32)
    gyro = window[["gyro_X", "gyro_Y", "gyro_Z"]].values.astype(np.float32)
    lin  = window[["lin_X",  "lin_Y",  "lin_Z"]].values.astype(np.float32)
    mag  = window[["mag_X",  "mag_Y",  "mag_Z"]].values.astype(np.float32)
    rot  = window[["rot_X",  "rot_Y",  "rot_Z"]].values.astype(np.float32)

    lin_acc  = acc - grav                       # paper Section 6.1 formula
    acc_rot  = rotate_to_earth(acc,  rot)
    gyro_rot = rotate_to_earth(gyro, rot)
    mag_rot  = rotate_to_earth(mag,  rot)

    return np.stack([
        pad4(acc,                        T),  #  1  acc_raw
        pad4(gyro,                       T),  #  2  gyro_raw
        pad4(mag,                        T),  #  3  mag_raw
        pad4(lin_acc,                    T),  #  4  lin_acc_manual
        pad4(acc_rot,                    T),  #  5  acc_rot
        pad4(gyro_rot,                   T),  #  6  gyro_rot
        pad4(mag_rot,                    T),  #  7  mag_rot
        pad4(diff(acc),                  T),  #  8  acc_delta
        pad4(diff(gyro),                 T),  #  9  gyro_delta
        pad4(diff(mag),                  T),  # 10  mag_delta
        pad4(diff(acc_rot),              T),  # 11  acc_rot_delta
        pad4(diff(gyro_rot),             T),  # 12  gyro_rot_delta
        pad4(diff(mag_rot),              T),  # 13  mag_rot_delta
        pad4(np.cumsum(acc,      axis=0), T), # 14  acc_integral
        pad4(np.cumsum(gyro,     axis=0), T), # 15  gyro_integral
        pad4(np.cumsum(mag,      axis=0), T), # 16  mag_integral
        pad4(np.cumsum(acc_rot,  axis=0), T), # 17  acc_rot_integral
        pad4(np.cumsum(gyro_rot, axis=0), T), # 18  gyro_rot_integral
        pad4(np.cumsum(mag_rot,  axis=0), T), # 19  mag_rot_integral
        pad4(diff(lin_acc),              T),  # 20  lin_acc_delta
        pad4(np.cumsum(lin_acc,  axis=0), T), # 21  lin_acc_integral
        pad4(rot,                        T),  # 22  rot_raw
    ], axis=0)   # → (22, 4, T)


def preprocess_user(uid: str, session_dir: str):
    """
    Process all unlock events for one user.
    Returns (features, cluster_ids) or (None, None).
      features:    (N_valid, 22, 4, T)
      cluster_ids: (N_valid,)  values 0-5
    """
    merged, screen = load_session(session_dir)
    if merged is None or screen is None:
        return None, None

    unlock_rows = (screen[screen["event"] == FLAG_USER_PRESENT]
                   .sort_values("timestamp"))
    feats, clusters, n_skip = [], [], 0
    for trial_idx, (_, row) in enumerate(unlock_rows.iterrows()):
        w = extract_window(merged, int(row["timestamp"]))
        if w is None:
            n_skip += 1; continue
        feats.append(compute_features(w))
        clusters.append(trial_idx // 50)        # 6 clusters × 50 trials

    if n_skip:
        print(f"  user={uid}: skipped {n_skip} trials")
    if not feats:
        return None, None
    return (np.stack(feats, axis=0).astype(np.float32),
            np.array(clusters, dtype=np.int64))


def run_on_dummy():
    from utils.dummy_data import load_uv_dummy
    X, cl = load_uv_dummy()
    assert X.shape == (101, 300, 22, 4, 50), f"Wrong shape: {X.shape}"
    print(f"Dummy UV: X={X.shape}  OK")

    # Test compute_features on synthetic window
    T = 50; np.random.seed(0)
    fake_window = pd.DataFrame({
        "acc_X":  np.random.randn(T), "acc_Y":  np.random.randn(T),
        "acc_Z":  np.random.randn(T) + 9.8,
        "grav_X": np.zeros(T), "grav_Y": np.zeros(T),
        "grav_Z": np.full(T, 9.8),
        "gyro_X": np.random.randn(T)*0.1, "gyro_Y": np.random.randn(T)*0.1,
        "gyro_Z": np.random.randn(T)*0.1,
        "lin_X":  np.random.randn(T)*0.1, "lin_Y":  np.random.randn(T)*0.1,
        "lin_Z":  np.random.randn(T)*0.1,
        "mag_X":  np.random.randn(T)*10,  "mag_Y":  np.random.randn(T)*10,
        "mag_Z":  np.random.randn(T)*10,
        "rot_X":  np.random.randn(T)*0.1, "rot_Y":  np.random.randn(T)*0.1,
        "rot_Z":  np.random.randn(T)*0.1,
    })
    feats = compute_features(fake_window)
    assert feats.shape == (22, 4, T), f"Expected (22,4,{T}), got {feats.shape}"
    assert np.all(feats[:, 3, :] == 0.0), "4th channel must be zero (padding)"
    assert np.any(feats[:, 0, :] != 0.0), "Feature data must be non-zero"
    print(f"  compute_features: {feats.shape}  OK")
    print(f"  4th channel padding: confirmed all-zero")
    print("Dummy UV test passed.")


def main(use_dummy=False):
    if use_dummy:
        run_on_dummy(); return
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if not os.path.exists(UV_INPUT_DIR):
        print(f"UV input not found: {UV_INPUT_DIR}"); return
    user_dirs = sorted(
        d for d in os.listdir(UV_INPUT_DIR)
        if os.path.isdir(os.path.join(UV_INPUT_DIR, d)))
    print(f"Found {len(user_dirs)} users in UV dataset.")
    for uid in user_dirs:
        # Structure: {user_id}/s20/{user_id}_20000/
        base      = os.path.join(UV_INPUT_DIR, uid, "s20")
        if not os.path.exists(base):
            print(f"  user={uid}: no s20 dir, skipping"); continue
        sessions  = sorted(os.listdir(base))
        if not sessions:
            continue
        sdir      = os.path.join(base, sessions[0])
        feats, cl = preprocess_user(uid, sdir)
        if feats is None:
            print(f"  user={uid}: no valid trials"); continue
        np.savez(os.path.join(PROCESSED_DIR, f"{uid}.npz"),
                 features=feats, cluster_ids=cl, user_id=uid)
        print(f"  user={uid}: {feats.shape}")
    print("UV preprocessing done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-dummy", action="store_true")
    args = parser.parse_args()
    main(use_dummy=args.use_dummy)

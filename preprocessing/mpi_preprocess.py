"""
MPI dataset preprocessing.
Paper: Sections 4.1.1 and 4.3.1 of arXiv:2302.01751

EFFICIENT APPROACH: reads only small time windows around screen events,
NOT the entire 12-week session (which would be ~25M rows per sensor).

Run: python preprocessing/mpi_preprocess.py --use-dummy
"""

import os, sys, argparse
import numpy as np
import pandas as pd
from scipy import interpolate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import cfg
from utils.bin_reader import (read_sensor_bin, read_screen,
                               SENSOR_FILES, SENSOR_COLS, ALL_SENSOR_COLS,
                               FLAG_USER_PRESENT, FLAG_SCREEN_OFF, FLAG_SCREEN_ON)

MPI_INPUT_DIRS = [
    "/kaggle/input/datasets/djaarf/motionid-imu-all-motions-part1/IMU_all_motions_part1",
    "/kaggle/input/datasets/djaarf/motionid-imu-all-motions-part2/IMU_all_motions_part2",
    "/kaggle/input/datasets/djaarf/motionid-imu-all-motions-part3/IMU_all_motions_part3",
]
PROCESSED_DIR = "data/mpi/processed"
LIN_COLS      = ["lin_X", "lin_Y", "lin_Z"]


def read_sensors_window(session_dir: str, t_start: int, t_end: int) -> pd.DataFrame:
    """Read all 6 sensors in a tight time window and merge."""
    sensor_dfs = {}
    for name, fname in SENSOR_FILES.items():
        fpath = os.path.join(session_dir, fname)
        if not os.path.exists(fpath):
            return None
        df = read_sensor_bin(fpath, t_start=t_start, t_end=t_end)
        if len(df) == 0:
            return None
        cols = SENSOR_COLS[name]
        df = df.rename(columns={"X": cols[0], "Y": cols[1], "Z": cols[2]})
        sensor_dfs[name] = df.sort_values("timestamp").reset_index(drop=True)

    merged = sensor_dfs["acc"]
    for name in ["grav", "gyro", "lin", "mag", "rot"]:
        merged = pd.merge_asof(
            merged, sensor_dfs[name],
            on="timestamp", direction="nearest", tolerance=100)
    return merged.dropna().reset_index(drop=True)


def extract_positives(session_dir: str, screen_df: pd.DataFrame) -> list:
    """Paper 4.3.1: 3 sec before each USER_PRESENT, >=100 readings."""
    samples = []
    window_ms = int(cfg.mpi_window_sec * 1000)
    for _, row in screen_df[screen_df["event"] == FLAG_USER_PRESENT].iterrows():
        ts = int(row["timestamp"])
        merged = read_sensors_window(session_dir, ts - window_ms, ts)
        if merged is not None and len(merged) >= cfg.mpi_min_readings:
            samples.append(merged[ALL_SENSOR_COLS].values.astype(np.float32))
    return samples


def extract_negatives(session_dir: str, screen_df: pd.DataFrame) -> list:
    """Paper 4.3.1: SCREEN_OFF to next SCREEN_ON/USER_PRESENT, exclude last 3s.
    MEMORY FIX: only read last 60s of each interval (not hours of idle data)."""
    samples = []
    window_ms = int(cfg.mpi_window_sec * 1000)
    max_read_sec = 60  # cap: don't read hours of idle phone data
    max_read_ms = max_read_sec * 1000
    off_events = screen_df[screen_df["event"] == FLAG_SCREEN_OFF]

    for _, row in off_events.iterrows():
        off_ts = int(row["timestamp"])
        later = screen_df[
            (screen_df["timestamp"] > off_ts) &
            (screen_df["event"].isin([FLAG_SCREEN_ON, FLAG_USER_PRESENT]))]
        if later.empty:
            continue
        end_ts = int(later.iloc[0]["timestamp"])
        effective_end = end_ts - window_ms
        if effective_end <= off_ts:
            continue
        # MEMORY FIX: only read last 60s, not entire gap
        read_start = max(off_ts, effective_end - max_read_ms)
        if effective_end - read_start < window_ms:
            continue

        merged = read_sensors_window(session_dir, read_start, effective_end)
        if merged is None or len(merged) < cfg.mpi_min_readings:
            continue
        # Check not stationary
        lin = merged[LIN_COLS].values
        if np.all(np.abs(lin) < cfg.stationary_threshold):
            continue
        samples.append(merged[ALL_SENSOR_COLS].values.astype(np.float32))
    return samples


def normalize_length(sample: np.ndarray, target_len: int = 150) -> np.ndarray:
    """(n, 18) -> (18, target_len) channel-first."""
    n, c = sample.shape
    if n == target_len:
        return sample.T.astype(np.float32)
    if n < target_len:
        x_old, x_new = np.linspace(0, 1, n), np.linspace(0, 1, target_len)
        out = np.zeros((target_len, c), dtype=np.float32)
        for i in range(c):
            out[:, i] = interpolate.interp1d(
                x_old, sample[:, i], kind="linear")(x_new)
        return out.T
    return sample[-target_len:].T.astype(np.float32)


def discover_sessions(input_dirs: list) -> list:
    """Returns list of (user_id, device_id, session_dir)."""
    sessions = []
    for root in input_dirs:
        if not os.path.exists(root):
            continue
        for user in sorted(os.listdir(root)):
            user_path = os.path.join(root, user)
            if not os.path.isdir(user_path):
                continue
            for phone in sorted(os.listdir(user_path)):
                phone_path = os.path.join(user_path, phone)
                if not os.path.isdir(phone_path):
                    continue
                for session in sorted(os.listdir(phone_path)):
                    session_path = os.path.join(phone_path, session)
                    if os.path.isdir(session_path):
                        sessions.append((user, phone, session_path))
    return sessions


def process_session(uid: str, did: str, session_dir: str):
    target_len = int(cfg.mpi_sampling_rate * cfg.mpi_window_sec)

    # Read screen events (small text file)
    screen_path = os.path.join(session_dir, "screen.txt")
    if not os.path.exists(screen_path):
        return None, None
    screen_df = read_screen(screen_path)
    if len(screen_df) == 0:
        return None, None

    # Extract windows around events (reads only small chunks)
    pos = extract_positives(session_dir, screen_df)
    neg = extract_negatives(session_dir, screen_df)

    if len(pos) < 10 or len(neg) < 10:
        return None, None

    X = np.stack([normalize_length(s, target_len) for s in pos + neg])
    y = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int64)
    return X, y


def run_on_dummy():
    from utils.dummy_data import load_mpi_dummy
    X, y, _ = load_mpi_dummy()
    print(f"Dummy MPI: X={X.shape}, y={y.shape}")
    for n_in in [120, 200, 150]:
        out = normalize_length(np.random.randn(n_in, 18).astype(np.float32), 150)
        assert out.shape == (18, 150), f"Got {out.shape}"
        print(f"  normalize_length({n_in}, 18) -> {out.shape}  OK")
    print("Dummy MPI test passed.")


def main(use_dummy=False):
    if use_dummy:
        run_on_dummy(); return
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    sessions = discover_sessions(MPI_INPUT_DIRS)
    print(f"Found {len(sessions)} sessions across 3 MPI datasets.")
    rows = []
    for uid, did, sdir in sessions:
        print(f"  user={uid} device={did}...", end=" ", flush=True)
        X, y = process_session(uid, did, sdir)
        key  = f"{uid}_{did}"
        if X is None:
            print("N/A")
            rows.append({"user_id": uid, "device_id": did,
                         "n_pos": 0, "n_neg": 0, "status": "N/A"})
        else:
            np.savez(os.path.join(PROCESSED_DIR, f"{key}.npz"), X=X, y=y)
            n_pos, n_neg = int((y==1).sum()), int((y==0).sum())
            print(f"X={X.shape}, pos={n_pos}, neg={n_neg}")
            rows.append({"user_id": uid, "device_id": did,
                         "n_pos": n_pos, "n_neg": n_neg, "status": "OK"})
    mf = pd.DataFrame(rows)
    mf.to_csv(os.path.join(PROCESSED_DIR, "manifest.csv"), index=False)
    print(f"Done. {(mf.status=='OK').sum()} valid sessions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-dummy", action="store_true")
    args = parser.parse_args()
    main(use_dummy=args.use_dummy)

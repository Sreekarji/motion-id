"""
Reads MotionID sensor files.

The Kaggle dataset stores sensors as files with .txt extension but BINARY content:
  accel.txt, gravity.txt, gyro.txt, linAccel.txt, MagneticField.txt, Rotation.txt
  Format: 20 bytes per record
    bytes 0-7:  timestamp ms, big-endian int64   (struct '>q')
    bytes 8-19: X Y Z values, little-endian float32 (struct '<fff')

screen.txt is actual text:
  Format: "{timestamp_ms} {android.intent.action.*}"

EFFICIENT LOADING: binary files support seeking to a time offset,
so we only read records within the needed time window.
"""

import os
import struct
import numpy as np
import pandas as pd

FLAG_USER_PRESENT = "android.intent.action.USER_PRESENT"
FLAG_SCREEN_OFF   = "android.intent.action.SCREEN_OFF"
FLAG_SCREEN_ON    = "android.intent.action.SCREEN_ON"

RECORD_SIZE = 20  # bytes per sensor record

SENSOR_FILES = {
    "acc":  "accel.txt",
    "grav": "gravity.txt",
    "gyro": "gyro.txt",
    "lin":  "linAccel.txt",
    "mag":  "MagneticField.txt",
    "rot":  "Rotation.txt",
}
SCREEN_FILE = "screen.txt"

SENSOR_COLS = {
    name: [f"{name}_X", f"{name}_Y", f"{name}_Z"]
    for name in SENSOR_FILES
}

ALL_SENSOR_COLS = (
    SENSOR_COLS["acc"]  + SENSOR_COLS["grav"] + SENSOR_COLS["gyro"] +
    SENSOR_COLS["lin"]  + SENSOR_COLS["mag"]  + SENSOR_COLS["rot"]
)


def read_sensor_bin(path: str,
                    t_start: int = None,
                    t_end:   int = None) -> pd.DataFrame:
    """
    Read a binary sensor file (20 bytes/record).
    If t_start/t_end given, uses binary search to skip to the right offset.
    File must be sorted by timestamp (it is).
    """
    file_size = os.path.getsize(path)
    if file_size == 0 or file_size % RECORD_SIZE != 0:
        return pd.DataFrame(columns=["timestamp", "X", "Y", "Z"])

    n_records = file_size // RECORD_SIZE

    with open(path, "rb") as f:
        # If no time filter, read everything
        if t_start is None and t_end is None:
            data = f.read()
            return _parse_binary_block(data)

        # Binary search for start offset
        start_idx = _binary_search_ts(f, n_records, t_start or 0)
        # Read from start_idx to end (or until t_end)
        f.seek(start_idx * RECORD_SIZE)
        # Read a chunk, then filter by timestamp
        remaining = (n_records - start_idx) * RECORD_SIZE
        chunk_size = min(remaining, 10_000_000)  # 10MB chunks
        data = f.read(chunk_size)
        df = _parse_binary_block(data)
        # Filter by time range
        if t_start is not None:
            df = df[df["timestamp"] >= t_start]
        if t_end is not None:
            df = df[df["timestamp"] <= t_end]
        return df.reset_index(drop=True)


def _parse_binary_block(data: bytes) -> pd.DataFrame:
    """Parse raw bytes into DataFrame using numpy (fast)."""
    n = len(data) // RECORD_SIZE
    if n == 0:
        return pd.DataFrame(columns=["timestamp", "X", "Y", "Z"])
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n, 20)
    # Timestamp: bytes 0-7, big-endian int64
    ts = np.zeros(n, dtype=np.int64)
    for i in range(8):
        ts = ts | (arr[:, i].astype(np.int64) << (56 - 8 * i))
    # XYZ: bytes 8-19, little-endian float32
    xyz = arr[:, 8:20].view(np.float32).reshape(n, 3)
    return pd.DataFrame({"timestamp": ts, "X": xyz[:, 0],
                         "Y": xyz[:, 1], "Z": xyz[:, 2]})


def _binary_search_ts(f, n_records: int, target_ts: int) -> int:
    """Binary search for the first record with timestamp >= target_ts."""
    lo, hi = 0, n_records - 1
    while lo < hi:
        mid = (lo + hi) // 2
        f.seek(mid * RECORD_SIZE)
        ts = struct.unpack(">q", f.read(8))[0]
        if ts < target_ts:
            lo = mid + 1
        else:
            hi = mid
    return lo


def read_screen(path: str) -> pd.DataFrame:
    """
    Read screen.txt. Handles both formats:
      "timestamp event"              (2 fields — Kaggle dataset format)
      "date time event"              (3 fields — older format)
    Returns DataFrame with columns [timestamp, event].
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    rows.append({"timestamp": int(parts[0]), "event": parts[1]})
                except ValueError:
                    continue
            elif len(parts) == 3:
                pass  # older format, skip
    return pd.DataFrame(rows)


def load_session(session_dir: str,
                 window_sec_before: float = 60.0) -> tuple:
    """
    Load all sensor files and screen events for one session.
    Uses binary search to only read records in the needed time window.

    Returns (merged_df, screen_df) or (None, None) on failure.
    """
    # Check all files exist
    for name, fname in SENSOR_FILES.items():
        if not os.path.exists(os.path.join(session_dir, fname)):
            return None, None
    screen_path = os.path.join(session_dir, SCREEN_FILE)
    if not os.path.exists(screen_path):
        return None, None

    # Read screen events first (small text file)
    screen_df = read_screen(screen_path)
    if len(screen_df) == 0:
        return None, None

    # Time window from screen events
    t_min = int(screen_df["timestamp"].min()) - int(window_sec_before * 1000)
    t_max = int(screen_df["timestamp"].max()) + 5000

    # Load each sensor with time filtering via binary search
    sensor_dfs = {}
    for name, fname in SENSOR_FILES.items():
        fpath = os.path.join(session_dir, fname)
        df    = read_sensor_bin(fpath, t_start=t_min, t_end=t_max)
        if len(df) == 0:
            return None, None
        cols = SENSOR_COLS[name]
        df   = df.rename(columns={"X": cols[0], "Y": cols[1], "Z": cols[2]})
        sensor_dfs[name] = df.sort_values("timestamp").reset_index(drop=True)

    # Merge all sensors on timestamp
    merged = sensor_dfs["acc"]
    for name in ["grav", "gyro", "lin", "mag", "rot"]:
        merged = pd.merge_asof(
            merged, sensor_dfs[name],
            on="timestamp", direction="nearest", tolerance=100)
    merged = merged.dropna().reset_index(drop=True)

    return merged, screen_df

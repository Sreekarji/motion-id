"""
Reads MotionID sensor text files.

The Kaggle dataset stores sensors as space-separated text files:
  accel.txt, gravity.txt, gyro.txt, linAccel.txt, MagneticField.txt, Rotation.txt
  Format per line: "{timestamp_ms} {X} {Y} {Z}"

screen.txt stores Android broadcast events:
  Format: "{timestamp_ms} {android.intent.action.*}"

EFFICIENT LOADING: for large files (12 weeks = ~25M rows),
we filter rows to a time window before loading into memory.
"""

import os
import numpy as np
import pandas as pd

FLAG_USER_PRESENT = "android.intent.action.USER_PRESENT"
FLAG_SCREEN_OFF   = "android.intent.action.SCREEN_OFF"
FLAG_SCREEN_ON    = "android.intent.action.SCREEN_ON"

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


def read_sensor_txt(path: str,
                    t_start: int = None,
                    t_end:   int = None) -> pd.DataFrame:
    """
    Read a sensor text file. Format: "timestamp X Y Z" (space-separated, no header).
    If t_start/t_end given, only loads rows within that time window.
    This avoids loading 25M-row files into memory when we only need a small window.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                ts = int(parts[0])
            except ValueError:
                continue
            if t_start is not None and ts < t_start:
                continue
            if t_end is not None and ts > t_end:
                break          # file is sorted by time — safe to break
            try:
                rows.append((ts, float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["timestamp", "X", "Y", "Z"])


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
                # "date time event" — skip, Kaggle already has 2-field format
                pass
    return pd.DataFrame(rows)


def load_session(session_dir: str,
                 window_sec_before: float = 60.0) -> tuple:
    """
    Load all sensor files and screen events for one session.

    For MPI: loads full session (many hours), so uses time windowing.
    For UV: 300 lifts x 1 second each ~ short session, loads all.

    Returns (merged_df, screen_df) or (None, None) on failure.
    merged_df columns: [timestamp, acc_X, acc_Y, acc_Z, grav_X, ...]
    screen_df columns: [timestamp, event]
    """
    # Check all files exist
    for name, fname in SENSOR_FILES.items():
        if not os.path.exists(os.path.join(session_dir, fname)):
            return None, None
    screen_path = os.path.join(session_dir, SCREEN_FILE)
    if not os.path.exists(screen_path):
        return None, None

    # Read screen events first (small file)
    screen_df = read_screen(screen_path)
    if len(screen_df) == 0:
        return None, None

    # Find time range needed: from first SCREEN_OFF to last USER_PRESENT
    # Add buffer so we have data for all windows
    t_min = int(screen_df["timestamp"].min()) - int(window_sec_before * 1000)
    t_max = int(screen_df["timestamp"].max()) + 5000

    # Load each sensor with time filtering
    sensor_dfs = {}
    for name, fname in SENSOR_FILES.items():
        fpath = os.path.join(session_dir, fname)
        df    = read_sensor_txt(fpath, t_start=t_min, t_end=t_max)
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

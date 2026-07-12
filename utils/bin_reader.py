"""
Reads the MotionID binary sensor files.
Format: 20 bytes per record
  bytes 0-7:   timestamp ms, big-endian int64  (struct '>q')
  bytes 8-19:  X Y Z values, little-endian float32  (struct '<fff')

Screen events (screen.txt) are plain text:
  "{timestamp_ms} {android_intent_string}"
  KEY FLAGS:
    android.intent.action.USER_PRESENT  ← unlock event
    android.intent.action.SCREEN_OFF
    android.intent.action.SCREEN_ON

Run: python utils/bin_reader.py
Expected: All bin_reader unit tests passed.
"""

import struct
import numpy as np
import pandas as pd
import os

# ── Constants ─────────────────────────────────────────────────────────────
FLAG_USER_PRESENT = "android.intent.action.USER_PRESENT"
FLAG_SCREEN_OFF   = "android.intent.action.SCREEN_OFF"
FLAG_SCREEN_ON    = "android.intent.action.SCREEN_ON"
RECORD_SIZE       = 20

# Sensor file names inside each session folder
SENSOR_FILES = {
    "acc":  "accel.bin",
    "grav": "gravity.bin",
    "gyro": "gyro.bin",
    "lin":  "linAccel.bin",
    "mag":  "MagneticField.bin",
    "rot":  "Rotation.bin",
}
SCREEN_FILE = "screen.txt"

# Column names after loading (prefix_X, prefix_Y, prefix_Z)
SENSOR_COLS = {
    name: [f"{name}_X", f"{name}_Y", f"{name}_Z"]
    for name in SENSOR_FILES
}

# All 18 sensor columns in fixed order
ALL_SENSOR_COLS = (
    SENSOR_COLS["acc"] + SENSOR_COLS["grav"] + SENSOR_COLS["gyro"] +
    SENSOR_COLS["lin"] + SENSOR_COLS["mag"]  + SENSOR_COLS["rot"]
)


def read_bin(path: str) -> pd.DataFrame:
    """
    Read one .bin sensor file.
    Returns DataFrame with columns [timestamp, X, Y, Z].
    Raises AssertionError if file size is not divisible by 20.
    """
    with open(path, "rb") as f:
        data = f.read()
    assert len(data) % RECORD_SIZE == 0, (
        f"{path}: size {len(data)} not divisible by {RECORD_SIZE}")
    n   = len(data) // RECORD_SIZE
    ts  = np.zeros(n, dtype=np.int64)
    xyz = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        off    = i * RECORD_SIZE
        ts[i]  = struct.unpack(">q",   data[off:off+8])[0]
        xyz[i] = struct.unpack("<fff", data[off+8:off+20])
    return pd.DataFrame({"timestamp": ts, "X": xyz[:,0],
                         "Y": xyz[:,1],   "Z": xyz[:,2]})


def read_screen(path: str) -> pd.DataFrame:
    """
    Read screen.txt event file.
    Format: "{timestamp_ms} {android.intent.action.*}"
    Returns DataFrame with columns [timestamp, event].
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    rows.append({"timestamp": int(parts[0]),
                                 "event":     parts[1]})
                except ValueError:
                    continue
            elif len(parts) == 3:
                # Old format: "date time event" — convert to ms
                # Skip; Kaggle dataset already uses 2-field format
                pass
    return pd.DataFrame(rows)


def load_session(session_dir: str) -> tuple:
    """
    Load all sensor .bin files and screen.txt from one session directory.
    Returns (merged_df, screen_df) where:
      merged_df has columns: [timestamp, acc_X, acc_Y, acc_Z, grav_X, ...]
      screen_df has columns: [timestamp, event]
    Returns (None, None) if any sensor file is missing or empty.
    """
    # Load each sensor
    sensor_dfs = {}
    for name, fname in SENSOR_FILES.items():
        fpath = os.path.join(session_dir, fname)
        if not os.path.exists(fpath):
            return None, None
        df = read_bin(fpath)
        if len(df) == 0:
            return None, None
        cols = SENSOR_COLS[name]
        df   = df.rename(columns={"X": cols[0], "Y": cols[1], "Z": cols[2]})
        sensor_dfs[name] = df.sort_values("timestamp").reset_index(drop=True)

    # Load screen events
    screen_path = os.path.join(session_dir, SCREEN_FILE)
    if not os.path.exists(screen_path):
        return None, None
    screen_df = read_screen(screen_path)
    if len(screen_df) == 0:
        return None, None

    # Merge all sensors on timestamp using merge_asof (nearest, 100ms tolerance)
    merged = sensor_dfs["acc"]
    for name in ["grav", "gyro", "lin", "mag", "rot"]:
        merged = pd.merge_asof(
            merged,
            sensor_dfs[name],
            on="timestamp",
            direction="nearest",
            tolerance=100)
    merged = merged.dropna().reset_index(drop=True)

    return merged, screen_df


if __name__ == "__main__":
    import struct, tempfile, os

    # Test 1: read_bin with synthetic data
    n = 10
    data = b""
    for i in range(n):
        data += struct.pack(">q",   1614936922000 + i * 20)
        data += struct.pack("<fff", float(i), float(i)*2, float(i)*3)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(data); tmp = f.name
    df = read_bin(tmp)
    os.unlink(tmp)
    assert df.shape == (10, 4)
    assert df["timestamp"].iloc[0] == 1614936922000
    assert abs(df["X"].iloc[3] - 3.0) < 1e-5
    print(f"Test 1 — read_bin:    shape={df.shape}, ts[0]={df.timestamp.iloc[0]}  PASSED")

    # Test 2: read_screen
    screen_txt = "1614936900000 android.intent.action.SCREEN_OFF\n" \
                 "1614936922288 android.intent.action.USER_PRESENT\n" \
                 "1614936922300 android.intent.action.SCREEN_ON\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(screen_txt); tmp2 = f.name
    sdf = read_screen(tmp2)
    os.unlink(tmp2)
    unlocks = sdf[sdf["event"] == FLAG_USER_PRESENT]
    assert len(unlocks) == 1
    assert unlocks.iloc[0]["timestamp"] == 1614936922288
    print(f"Test 2 — read_screen: {len(sdf)} events, "
          f"{len(unlocks)} USER_PRESENT  PASSED")

    # Test 3: ALL_SENSOR_COLS count
    assert len(ALL_SENSOR_COLS) == 18
    print(f"Test 3 — 18 sensor cols: {ALL_SENSOR_COLS}  PASSED")

    print("\nAll bin_reader unit tests passed.")

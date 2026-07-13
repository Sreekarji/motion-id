# Cell 2: Numpy-only sensor reader (low memory, fast)
# Loads each sensor file as numpy array, not pandas DataFrame.
# Aligns sensors by timestamp using np.interp.

def read_sensor_numpy(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 20
    if n == 0: return np.empty((0, 4), dtype=np.float64)
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n, 20)
    ts = np.zeros(n, dtype=np.float64)
    for i in range(8):
        ts += arr[:, i].astype(np.float64) * (2 ** (56 - 8 * i))
    xyz = arr[:, 8:20].view(np.float32).reshape(n, 3)
    out = np.column_stack([ts, xyz])
    return out[out[:, 0].argsort()]

def read_screen(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try: rows.append((int(parts[0]), parts[1]))
                except ValueError: continue
    return rows

def load_all_sensors_numpy(session_dir):
    sensors = {}
    for name, fname in SENSOR_FILES.items():
        fpath = os.path.join(session_dir, fname)
        if not os.path.exists(fpath): return None
        arr = read_sensor_numpy(fpath)
        if len(arr) == 0: return None
        sensors[name] = arr
    return sensors

def get_aligned_window(sensors, t_start, t_end):
    # Use acc timestamps as reference
    acc = sensors["acc"]
    idx_start = np.searchsorted(acc[:, 0], t_start, side="left")
    idx_end   = np.searchsorted(acc[:, 0], t_end,   side="right")
    if idx_end - idx_start < 10: return None
    acc_ts = acc[idx_start:idx_end, 0]
    result_cols = []
    for name in ["acc", "grav", "gyro", "lin", "mag", "rot"]:
        s = sensors[name]
        s_idx = np.searchsorted(s[:, 0], t_start, side="left")
        s_ie  = np.searchsorted(s[:, 0], t_end,   side="right")
        if s_ie - s_idx < 5: return None
        s_ts = s[s_idx:s_ie, 0]
        for col in range(1, 4):
            result_cols.append(np.interp(acc_ts, s_ts, s[s_idx:s_ie, col]))
    return np.column_stack(result_cols)

print("Numpy reader ready (no pandas for sensor data).")

# Cell 2: Sensor reader (text format, time-windowed for efficiency)

def read_sensor_txt(path, t_start=None, t_end=None):
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4: continue
            try: ts = int(parts[0])
            except ValueError: continue
            if t_start is not None and ts < t_start: continue
            if t_end is not None and ts > t_end: break
            try: rows.append((ts, float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError: continue
    return pd.DataFrame(rows, columns=["timestamp", "X", "Y", "Z"])

def read_screen(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try: rows.append({"timestamp": int(parts[0]), "event": parts[1]})
                except ValueError: continue
    return pd.DataFrame(rows)

def read_sensors_window(session_dir, t_start, t_end):
    sensor_dfs = {}
    for name, fname in SENSOR_FILES.items():
        fpath = os.path.join(session_dir, fname)
        if not os.path.exists(fpath): return None
        df = read_sensor_txt(fpath, t_start=t_start, t_end=t_end)
        if len(df) == 0: return None
        cols = SENSOR_COLS[name]
        df = df.rename(columns={"X": cols[0], "Y": cols[1], "Z": cols[2]})
        sensor_dfs[name] = df.sort_values("timestamp").reset_index(drop=True)
    merged = sensor_dfs["acc"]
    for name in ["grav", "gyro", "lin", "mag", "rot"]:
        merged = pd.merge_asof(merged, sensor_dfs[name], on="timestamp",
                               direction="nearest", tolerance=100)
    return merged.dropna().reset_index(drop=True)

def load_session(session_dir, t_start=None, t_end=None):
    for fname in SENSOR_FILES.values():
        if not os.path.exists(os.path.join(session_dir, fname)): return None, None
    screen_path = os.path.join(session_dir, SCREEN_FILE)
    if not os.path.exists(screen_path): return None, None
    screen_df = read_screen(screen_path)
    if len(screen_df) == 0: return None, None
    merged = read_sensors_window(session_dir, t_start, t_end)
    if merged is None: return None, None
    return merged, screen_df

print("Sensor reader ready (text format, time-windowed).")
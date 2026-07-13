# Cell 3: MPI Preprocessing (numpy arrays, no pandas merge)
from scipy import interpolate
from tqdm import tqdm

def discover_sessions(input_dirs):
    sessions = []
    for root in input_dirs:
        if not os.path.exists(root): continue
        for user in sorted(os.listdir(root)):
            up = os.path.join(root, user)
            if not os.path.isdir(up): continue
            for phone in sorted(os.listdir(up)):
                pp = os.path.join(up, phone)
                if not os.path.isdir(pp): continue
                for session in sorted(os.listdir(pp)):
                    sp = os.path.join(pp, session)
                    if os.path.isdir(sp): sessions.append((user, phone, sp))
    return sessions

def normalize_length(sample, target_len=150):
    n, c = sample.shape
    if n == target_len: return sample.T.astype(np.float32)
    if n < target_len:
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, target_len)
        out = np.zeros((target_len, c), dtype=np.float32)
        for i in range(c):
            out[:, i] = np.interp(x_new, x_old, sample[:, i])
        return out.T
    return sample[-target_len:].T.astype(np.float32)

def process_session(uid, did, session_dir):
    target_len = int(cfg.mpi_sampling_rate * cfg.mpi_window_sec)
    window_ms  = int(cfg.mpi_window_sec * 1000)

    screen_path = os.path.join(session_dir, SCREEN_FILE)
    if not os.path.exists(screen_path): return None, None
    screen_events = read_screen(screen_path)
    if not screen_events: return None, None

    sensors = load_all_sensors_numpy(session_dir)
    if sensors is None: return None, None

    pos = []
    for ts, event in screen_events:
        if event != FLAG_USER_PRESENT: continue
        w = get_aligned_window(sensors, ts - window_ms, ts)
        if w is not None and len(w) >= cfg.mpi_min_readings:
            pos.append(w.astype(np.float32))

    neg, max_read_ms = [], 60000
    screen_ts = [t for t, _ in screen_events]
    screen_ev = [e for _, e in screen_events]
    for i, (off_ts, event) in enumerate(screen_events):
        if event != FLAG_SCREEN_OFF: continue
        end_ts = None
        for j in range(i+1, len(screen_events)):
            if screen_ev[j] in (FLAG_SCREEN_ON, FLAG_USER_PRESENT):
                end_ts = screen_ts[j]; break
        if end_ts is None: continue
        effective_end = end_ts - window_ms
        if effective_end <= off_ts: continue
        read_start = max(off_ts, effective_end - max_read_ms)
        if effective_end - read_start < window_ms: continue
        w = get_aligned_window(sensors, read_start, effective_end)
        if w is None or len(w) < cfg.mpi_min_readings: continue
        lin = w[:, 9:12]  # lin_X, lin_Y, lin_Z
        if np.all(np.abs(lin) < cfg.stationary_threshold): continue
        neg.append(w.astype(np.float32))

    if len(pos) < 10 or len(neg) < 10: return None, None
    X = np.stack([normalize_length(s, target_len) for s in pos + neg])
    y = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int64)
    return X, y

sessions = discover_sessions(MPI_DIRS)
print(f"Found {len(sessions)} MPI sessions")
rows = []
for uid, did, sdir in tqdm(sessions, desc="MPI"):
    X, y = process_session(uid, did, sdir)
    key = f"{uid}_{did}"
    if X is None:
        rows.append({"user_id": uid, "device_id": did, "n_pos": 0, "n_neg": 0, "status": "N/A"})
    else:
        np.savez(os.path.join(PROCESSED_MPI, f"{key}.npz"), X=X, y=y)
        n_pos, n_neg = int((y==1).sum()), int((y==0).sum())
        tqdm.write(f"  {uid}/{did}: X={X.shape} pos={n_pos} neg={n_neg}")
        rows.append({"user_id": uid, "device_id": did, "n_pos": n_pos, "n_neg": n_neg, "status": "OK"})
mf = pd.DataFrame(rows)
mf.to_csv(os.path.join(PROCESSED_MPI, "manifest.csv"), index=False)
print(f"MPI done. Valid: {(mf.status=='OK').sum()}/{len(mf)}")

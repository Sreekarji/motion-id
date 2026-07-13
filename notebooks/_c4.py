# Cell 4: UV Preprocessing
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def rotate_to_earth(v, rot_vec):
    return R.from_rotvec(rot_vec).apply(v).astype(np.float32)

def extract_window(merged_df, unlock_ts):
    n = int(cfg.uv_window_sec * cfg.uv_sampling_rate)
    window_ms = int(cfg.uv_window_sec * 1000)
    sub = merged_df[(merged_df["timestamp"] <= unlock_ts) &
                    (merged_df["timestamp"] > unlock_ts - window_ms)
                    ].sort_values("timestamp").tail(n)
    return sub.reset_index(drop=True) if len(sub) == n else None

def pad4(x, T):
    out = np.zeros((4, T), dtype=np.float32); out[:3, :] = x.T; return out

def diff(x): return np.concatenate([x[:1], np.diff(x, axis=0)], axis=0)

def compute_features(window):
    T = len(window)
    acc = window[["acc_X","acc_Y","acc_Z"]].values.astype(np.float32)
    grav = window[["grav_X","grav_Y","grav_Z"]].values.astype(np.float32)
    gyro = window[["gyro_X","gyro_Y","gyro_Z"]].values.astype(np.float32)
    mag = window[["mag_X","mag_Y","mag_Z"]].values.astype(np.float32)
    rot = window[["rot_X","rot_Y","rot_Z"]].values.astype(np.float32)
    lin_acc = acc - grav
    acc_rot = rotate_to_earth(acc, rot)
    gyro_rot = rotate_to_earth(gyro, rot)
    mag_rot = rotate_to_earth(mag, rot)
    return np.stack([
        pad4(acc, T), pad4(gyro, T), pad4(mag, T), pad4(lin_acc, T),
        pad4(acc_rot, T), pad4(gyro_rot, T), pad4(mag_rot, T),
        pad4(diff(acc), T), pad4(diff(gyro), T), pad4(diff(mag), T),
        pad4(diff(acc_rot), T), pad4(diff(gyro_rot), T), pad4(diff(mag_rot), T),
        pad4(np.cumsum(acc, axis=0), T), pad4(np.cumsum(gyro, axis=0), T),
        pad4(np.cumsum(mag, axis=0), T), pad4(np.cumsum(acc_rot, axis=0), T),
        pad4(np.cumsum(gyro_rot, axis=0), T), pad4(np.cumsum(mag_rot, axis=0), T),
        pad4(diff(lin_acc), T), pad4(np.cumsum(lin_acc, axis=0), T),
        pad4(rot, T),
    ], axis=0)

assert os.path.exists(UV_DIR), f"UV not found: {UV_DIR}"
user_dirs = sorted(d for d in os.listdir(UV_DIR) if os.path.isdir(os.path.join(UV_DIR, d)))
print(f"Found {len(user_dirs)} UV users")

for uid in tqdm(user_dirs, desc="UV"):
    base = os.path.join(UV_DIR, uid, "s20")
    if not os.path.exists(base): continue
    sessions = sorted(os.listdir(base))
    if not sessions: continue
    sdir = os.path.join(base, sessions[0])
    screen_path = os.path.join(sdir, SCREEN_FILE)
    screen_tmp = read_screen(screen_path)
    if len(screen_tmp) == 0: tqdm.write(f"  {uid}: no screen events"); continue
    t_min = int(screen_tmp["timestamp"].min()) - 2000
    t_max = int(screen_tmp["timestamp"].max()) + 2000
    merged, screen = load_session(sdir, t_start=t_min, t_end=t_max)
    if merged is None: tqdm.write(f"  {uid}: load failed"); continue
    unlocks = screen[screen["event"] == FLAG_USER_PRESENT].sort_values("timestamp")
    feats, clusters, n_skip = [], [], 0
    for trial_idx, (_, row) in enumerate(unlocks.iterrows()):
        w = extract_window(merged, int(row["timestamp"]))
        if w is None: n_skip += 1; continue
        feats.append(compute_features(w))
        clusters.append(trial_idx // 50)
    if not feats: continue
    F = np.stack(feats, 0).astype(np.float32)
    C = np.array(clusters, dtype=np.int64)
    np.savez(os.path.join(PROCESSED_UV, f"{uid}.npz"), features=F, cluster_ids=C, user_id=uid)
    tqdm.write(f"  {uid}: {F.shape}")
done = len([f for f in os.listdir(PROCESSED_UV) if f.endswith(".npz")])
print(f"UV done. {done} users saved.")
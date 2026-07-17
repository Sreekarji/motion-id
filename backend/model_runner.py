"""
Motion ID Backend Inference Engine
Loads trained checkpoints and runs MPI + UV prediction pipeline.
"""
import os, json, copy, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy.spatial.transform import Rotation as R

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CONFIG (copied from notebook Cell 1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    mpi_sampling_rate: int = 50
    mpi_window_sec: float = 3.0
    mpi_min_readings: int = 100
    mpi_n_channels: int = 18
    mpi_epochs: int = 30
    stationary_threshold: float = 0.01
    uv_sampling_rate: int = 50
    uv_window_sec: float = 1.0
    uv_augment_window_sec: float = 1.5
    uv_n_features: int = 22
    uv_n_channels_per_feature: int = 4
    uv_n_trials: int = 300
    uv_n_clusters: int = 6
    uv_total_users: int = 101
    uv_test_users: int = 11
    uv_baseline_n: int = 75
    uv_n_splits: List[int] = field(default_factory=lambda: [60, 65, 70, 75, 80, 85])
    baseline_lr: float = 1e-3
    finetune_lr: float = 1e-4
    baseline_epochs: int = 50
    finetune_epochs: int = 10
    batch_size: int = 64
    alpha_tm: float = 1.0
    supcon_temperature: float = 0.07
    n_training_runs: int = 5
    tar_threshold: float = 0.90
    target_far: float = 1 / 50000
    bootstrap_repeats: int = 5000
    far_sweep_steps: int = 100_000

cfg = Config()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MODEL CLASSES (copied from notebook Cell 5)
# ─────────────────────────────────────────────────────────────────────────────

class UVBranch(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Linear(128 * 8, 256)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc(self.pool(x).flatten(1))


class UVModel(nn.Module):
    def __init__(self, n_classes, n_features=22):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = n_features * 256
        self.branches = nn.ModuleList([UVBranch(4) for _ in range(n_features)])
        self.head_a = nn.Linear(self.embed_dim, n_classes)
        self.siamese_proj = nn.Linear(self.embed_dim, 256)
        self.head_b = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64))

    def _augment(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        T_t = int(cfg.uv_window_sec * cfg.uv_sampling_rate)
        T = x.size(-1)
        if T > T_t:
            start = torch.randint(0, T - T_t + 1, (1,)).item()
            x = x[..., start:start+T_t]
        return x + torch.randn_like(x) * 0.01

    def extract_embedding(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32,
                             device=next(self.parameters()).device)
        elif x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        if x.dtype != torch.float32:
            x = x.float()
        return torch.cat(
            [self.branches[i](x[:, i, :, :]) for i in range(self.n_features)],
            dim=1)

    def forward(self, x, augment=False):
        if augment: x = self._augment(x)
        emb = self.extract_embedding(x)
        logits = self.head_a(emb)
        siamese = F.normalize(self.siamese_proj(emb), dim=1)
        return logits, self.head_b(siamese)

    def get_siamese_embed(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.head_a.weight.device)
        with torch.no_grad():
            return F.normalize(self.siamese_proj(self.extract_embedding(x)), dim=1)


class MPIModel(nn.Module):
    def __init__(self, n_channels=18, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256*8, 256), nn.ReLU(), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.net(x))


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        B = embeddings.size(0)
        dist = ((embeddings.unsqueeze(0) - embeddings.unsqueeze(1))**2).sum(2)
        leq = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        losses = []
        for i in range(B):
            pm = leq[i] & ~eye[i]; nm = ~leq[i]
            if not pm.any() or not nm.any(): continue
            d_ap = dist[i][pm].max(); d_neg = dist[i][nm]
            semi = d_neg[(d_neg > d_ap) & (d_neg < d_ap + self.margin)]
            d_an = semi.min() if semi.numel() > 0 else d_neg.min()
            losses.append(F.relu(d_ap - d_an + self.margin))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True, device=embeddings.device)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, proj_embeddings, labels):
        N = proj_embeddings.size(0); B = labels.size(0)
        labels_rep = labels.repeat(2) if N == 2 * B else labels
        z = F.normalize(proj_embeddings, dim=1)
        sim = torch.mm(z, z.T) / self.temperature
        eye = torch.eye(N, dtype=torch.bool, device=z.device)
        pos_mask = (labels_rep.unsqueeze(0) == labels_rep.unsqueeze(1)) & ~eye
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        denom = torch.exp(sim).masked_fill(eye, 0).sum(1, keepdims=True)
        log_prob = sim - torch.log(denom + 1e-8)
        n_pos = pos_mask.sum(1).float(); valid = n_pos > 0
        loss = -(log_prob * pos_mask.float()).sum(1)
        return (loss[valid] / n_pos[valid]).mean()


class TotalLoss(nn.Module):
    def __init__(self, alpha_tm=1.0, temperature=0.07):
        super().__init__()
        self.alpha_tm = alpha_tm
        self.ce = nn.CrossEntropyLoss()
        self.tm = TripletMarginLoss(1.0)
        self.sc = SupervisedContrastiveLoss(temperature)

    def forward(self, logits, proj_embeds, labels):
        lce = self.ce(logits, labels)
        ltm = self.tm(F.normalize(proj_embeds[:labels.size(0)], dim=1), labels)
        lsc = self.sc(proj_embeds, labels)
        return lce + self.alpha_tm * ltm + lsc, {"lce": lce.item(), "ltm": ltm.item(), "lsc": lsc.item()}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — FEATURE COMPUTATION (copied from notebook Cell 4)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(sensor_matrix, rot_matrix):
    T = sensor_matrix.shape[0]
    acc = sensor_matrix[:, 0:3]
    grav = sensor_matrix[:, 3:6]
    gyro = sensor_matrix[:, 6:9]
    lin = sensor_matrix[:, 9:12]
    mag = sensor_matrix[:, 12:15]
    rot = rot_matrix

    rots = R.from_rotvec(rot)
    lin_acc = acc - grav
    acc_rot = rots.apply(acc).astype(np.float32)
    gyro_rot = rots.apply(gyro).astype(np.float32)
    mag_rot = rots.apply(mag).astype(np.float32)

    def diff(x):
        return np.vstack([x[:1], np.diff(x, axis=0)])

    def pad4(x):
        out = np.zeros((4, T), dtype=np.float32)
        out[:3] = x.T
        return out

    return np.stack([
        pad4(acc),
        pad4(gyro),
        pad4(mag),
        pad4(lin_acc),
        pad4(acc_rot),
        pad4(gyro_rot),
        pad4(mag_rot),
        pad4(diff(acc)),
        pad4(diff(gyro)),
        pad4(diff(mag)),
        pad4(diff(acc_rot)),
        pad4(diff(gyro_rot)),
        pad4(diff(mag_rot)),
        pad4(np.cumsum(acc, axis=0)),
        pad4(np.cumsum(gyro, axis=0)),
        pad4(np.cumsum(mag, axis=0)),
        pad4(np.cumsum(acc_rot, axis=0)),
        pad4(np.cumsum(gyro_rot, axis=0)),
        pad4(np.cumsum(mag_rot, axis=0)),
        pad4(diff(lin_acc)),
        pad4(np.cumsum(lin_acc, axis=0)),
        pad4(rot),
    ], axis=0)


def normalize_length(sample, target_len=None):
    if target_len is None:
        target_len = cfg.mpi_window_samples if hasattr(cfg, 'mpi_window_samples') \
                     else int(cfg.mpi_sampling_rate * cfg.mpi_window_sec)
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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SCORE VERIFICATION (copied from notebook Cell 7)
# ─────────────────────────────────────────────────────────────────────────────

def score_verification(model, X_genuine, X_impostor):
    if len(X_genuine) == 0 or len(X_impostor) == 0:
        return np.array([]), np.array([])
    model_device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        ge = model.get_siamese_embed(
            torch.tensor(X_genuine, dtype=torch.float32).to(model_device)).cpu().numpy()
        ie = model.get_siamese_embed(
            torch.tensor(X_impostor, dtype=torch.float32).to(model_device)).cpu().numpy()
    tmpl = ge.mean(0, keepdims=True)
    tmpl /= (np.linalg.norm(tmpl) + 1e-8)
    gn = ge / (np.linalg.norm(ge, axis=1, keepdims=True) + 1e-8)
    in_ = ie / (np.linalg.norm(ie, axis=1, keepdims=True) + 1e-8)
    g = np.atleast_1d((gn @ tmpl.T).squeeze())
    i = np.atleast_1d((in_ @ tmpl.T).squeeze())
    return g, i


def compute_far_at_tar(genuine, impostor, target_tar=0.90, n_steps=100_000):
    gen, imp = np.asarray(genuine, dtype=np.float64), np.asarray(impostor, dtype=np.float64)
    if len(gen) == 0 or len(imp) == 0: return 1.0, 0.0
    all_s = np.concatenate([gen, imp])
    for t in np.linspace(all_s.max(), all_s.min(), n_steps):
        if np.mean(gen >= t) >= target_tar:
            return float(np.mean(imp >= t)), float(t)
    return 1.0, float(all_s.min())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — ModelManager
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelManager:
    def __init__(self, checkpoints_dir: str, uv_processed_dir: str,
                 mpi_processed_dir: str, inventory_path: str = None):
        self.cfg = Config()
        self.checkpoints_dir = checkpoints_dir
        self.uv_processed_dir = uv_processed_dir
        self.mpi_processed_dir = mpi_processed_dir

        if inventory_path is None:
            inventory_path = os.path.join(os.path.dirname(checkpoints_dir), "inventory.json")
        with open(inventory_path) as f:
            self.inventory = json.load(f)

        self.active_n = self.inventory.get("active_n", 75)
        self.mpi_models: Dict[str, MPIModel] = {}
        self.uv_finetuned: Dict[int, UVModel] = {}
        self.user_train_features: Dict[int, np.ndarray] = {}
        self.user_thresholds: Dict[int, float] = {}

        self._load_all()

    def _parse_n_from_filename(self, fname: str) -> int:
        m = re.search(r"_n(\d+)", fname)
        return int(m.group(1)) if m else self.active_n

    def _load_all(self):
        inv = self.inventory

        # 1. MPI checkpoints
        mpi_ckpts = inv.get("mpi_checkpoints", [])
        if not mpi_ckpts:
            print("MPI checkpoints: NONE — MPI stage will return stub (is_unlock=True)")
        else:
            for ckpt_path in mpi_ckpts:
                if not os.path.exists(ckpt_path):
                    print(f"  WARNING: MPI checkpoint not found: {ckpt_path}")
                    continue
                fname = os.path.basename(ckpt_path)
                try:
                    try:
                        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                    except Exception:
                        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    model = MPIModel(n_channels=self.cfg.mpi_n_channels, n_classes=2).to(device)
                    state = ckpt.get("model_state", ckpt)
                    model.load_state_dict(state)
                    model.eval()
                    self.mpi_models[fname] = model
                    print(f"  Loaded MPI: {fname}")
                except Exception as e:
                    print(f"  ERROR loading MPI {fname}: {e}")

        # 2. UV fine-tuned checkpoints for active_n
        ft_group = inv.get("finetuned_checkpoints", {}).get(f"n{self.active_n}", {})
        for uid_str, ckpt_path in ft_group.items():
            if not os.path.exists(ckpt_path):
                print(f"  WARNING: ft checkpoint not found: {ckpt_path}")
                continue
            try:
                uid = int(uid_str)
                try:
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                except Exception:
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = UVModel(n_classes=2).to(device)
                state = ckpt.get("model_state", ckpt)
                model.load_state_dict(state)
                model.eval()
                self.uv_finetuned[uid] = model
                print(f"  Loaded UV ft: user={uid}")
            except Exception as e:
                print(f"  ERROR loading UV ft uid={uid_str}: {e}")

        # 3. Load training features for each fine-tuned user
        for uid in self.uv_finetuned:
            self._load_user_train_features(uid)

        print(f"\nModelManager ready.")
        print(f"  MPI models: {len(self.mpi_models)} {'(stubbed)' if not self.mpi_models else ''}")
        print(f"  UV fine-tuned: {len(self.uv_finetuned)} users")
        print(f"  Available users: {sorted(self.uv_finetuned.keys())}")

    def _find_user_npz(self, user_id: int) -> Optional[str]:
        """Find .npz file for user, handling zero-padded names (091.npz vs 91.npz)."""
        tried = []
        for fmt in [f"{user_id}.npz", f"{user_id:03d}.npz", f"{user_id:04d}.npz"]:
            p = os.path.join(self.uv_processed_dir, fmt)
            tried.append(p)
            if os.path.exists(p):
                return p
        print(f"  _find_user_npz: no file found for user {user_id}. Tried: {tried}")
        return None

    def _load_user_train_features(self, user_id: int):
        npz_path = self._find_user_npz(user_id)
        if npz_path is None:
            print(f"  WARNING: no .npz for user {user_id}")
            return
        data = np.load(npz_path)
        feats = data["features"]
        rng = np.random.default_rng(seed=user_id)
        idx = rng.permutation(feats.shape[0])
        n_train = max(1, int(feats.shape[0] * 0.70))
        train_feats = feats[idx[:n_train]].astype(np.float32)
        val_feats   = feats[idx[n_train:]].astype(np.float32)
        self.user_train_features[user_id] = train_feats

        # Compute per-user threshold from validation split.
        # Use half of training as genuine template, val set as query.
        # This is a demo approximation — real deployment would use held-out other-user data.
        if user_id in self.uv_finetuned and len(val_feats) >= 2:
            try:
                model = self.uv_finetuned[user_id]
                half = max(1, len(train_feats) // 2)
                g_scores, q_scores = score_verification(model, train_feats[:half], val_feats)
                if len(q_scores) > 0:
                    threshold = float(np.percentile(q_scores, 10))  # bottom 10% cut
                    self.user_thresholds[user_id] = max(0.0, min(1.0, threshold))
                    print(f"  Threshold user={user_id}: {self.user_thresholds[user_id]:.4f}")
            except Exception as e:
                print(f"  WARNING: threshold computation failed for user {user_id}: {e}")

    def get_available_users(self) -> list:
        return sorted(self.uv_finetuned.keys())

    def get_random_mpi_sample(self) -> dict:
        """Return a random MPI sample (X shape (18,150)) and its label."""
        mpi_dir = self.mpi_processed_dir
        files = [f for f in os.listdir(mpi_dir) if f.endswith('.npz')]
        if not files:
            return None
        fname = files[np.random.randint(len(files))]
        data = np.load(os.path.join(mpi_dir, fname))
        X = data['X']  # (N_samples, 18, 150)
        y = data['y']  # (N_samples,)
        idx = np.random.randint(len(X))
        # Convert (18, 150) to sensor_data dict for predict_mpi
        sample = X[idx]  # (18, 150)
        sensor_data = {
            "acc":  sample[0:3].T.tolist(),   # (150, 3)
            "grav": sample[3:6].T.tolist(),
            "gyro": sample[6:9].T.tolist(),
            "lin":  sample[9:12].T.tolist(),
            "mag":  sample[12:15].T.tolist(),
            "rot":  sample[15:18].T.tolist(),
        }
        return {
            "sensor_data": sensor_data,
            "label": int(y[idx]),
            "source_file": fname,
            "sample_index": int(idx),
        }

    def get_random_sample(self, user_id: int) -> dict:
        npz_path = self._find_user_npz(user_id)
        if npz_path is None:
            raise ValueError(f"No data for user {user_id}")
        data = np.load(npz_path)
        feats = data["features"]
        n_train = int(feats.shape[0] * 0.70)
        test_feats = feats[n_train:]
        if len(test_feats) == 0:
            test_feats = feats
        idx = np.random.randint(len(test_feats))
        return {
            "features": test_feats[idx].tolist(),
            "n_trials_total": int(feats.shape[0]),
            "trial_index": int(n_train + idx),
        }

    def predict_mpi(self, sensor_data: dict) -> dict:
        if not self.mpi_models:
            return {
                "is_unlock": True,
                "confidence": 1.0,
                "note": "MPI bypassed — no checkpoint available"
            }
        try:
            channels = []
            for key in ["acc", "grav", "gyro", "lin", "mag", "rot"]:
                arr = np.array(sensor_data[key], dtype=np.float32)
                channels.append(arr.T)
            X = np.concatenate(channels, axis=0)
            target = int(self.cfg.mpi_sampling_rate * self.cfg.mpi_window_sec)
            if X.shape[1] != target:
                x_old = np.linspace(0, 1, X.shape[1])
                x_new = np.linspace(0, 1, target)
                X_norm = np.zeros((18, target), dtype=np.float32)
                for c in range(18):
                    X_norm[c] = np.interp(x_new, x_old, X[c])
                X = X_norm
            tensor = torch.tensor(X[np.newaxis], dtype=torch.float32).to(device)
            all_probs = []
            for model in self.mpi_models.values():
                model.eval()
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0, 1].item()
                    all_probs.append(probs)
            confidence = float(np.mean(all_probs))
            is_unlock = confidence > 0.5
            return {
                "is_unlock": bool(is_unlock),
                "confidence": round(confidence, 4),
                "raw_score": round(confidence, 4)
            }
        except Exception as e:
            print(f"  ERROR in predict_mpi: {type(e).__name__}: {e}")
            return {"is_unlock": False, "confidence": 0.0,
                    "note": f"MPI inference error: {str(e)} — defaulting to reject"}

    def predict_uv(self, user_id: int, features: np.ndarray) -> dict:
        if user_id not in self.uv_finetuned:
            raise ValueError(f"No fine-tuned model for user {user_id}")
        if user_id not in self.user_train_features:
            self._load_user_train_features(user_id)
            if user_id not in self.user_train_features:
                raise ValueError(f"No training data for user {user_id}")

        model = self.uv_finetuned[user_id]
        train_feats = self.user_train_features[user_id]
        query = features[np.newaxis].astype(np.float32)

        g_scores, q_scores = score_verification(model, train_feats, query)
        identity_score = float(q_scores[0]) if len(q_scores) > 0 else 0.0

        threshold = self.user_thresholds.get(user_id, 0.5)
        decision = "ACCEPT" if identity_score > threshold else "REJECT"

        return {
            "user_id": user_id,
            "decision": decision,
            "score": round(identity_score, 4),
            "threshold": threshold
        }

    def predict_full(self, user_id: int, features: np.ndarray,
                     sensor_data_3s: dict = None) -> dict:
        mpi_result = None

        if sensor_data_3s is not None:
            mpi_result = self.predict_mpi(sensor_data_3s)
            if not mpi_result.get("is_unlock", True):
                return {
                    "mpi": mpi_result,
                    "uv": None,
                    "final_decision": "REJECT",
                    "pipeline_short_circuited": True,
                    "user_id": user_id,
                    "stage_rejected": "MPI"
                }
        else:
            mpi_result = {
                "is_unlock": True, "confidence": 1.0,
                "note": "MPI bypassed — no 3s sensor data provided"
            }

        uv_result = self.predict_uv(user_id, features)
        return {
            "mpi": mpi_result,
            "uv": uv_result,
            "final_decision": uv_result["decision"],
            "pipeline_short_circuited": False,
            "user_id": user_id
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MODULE GUARD
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mm = ModelManager(
        checkpoints_dir=r"D:\motionid\checkpoints",
        uv_processed_dir=r"D:\motionid\uv_processed",
        mpi_processed_dir=r"D:\motionid\mpi_processed",
        inventory_path=r"D:\motionid\inventory.json"
    )
    print("\nAvailable users:", mm.get_available_users())

    users = mm.get_available_users()
    if users:
        uid = users[0]
        sample = mm.get_random_sample(uid)
        feats = np.array(sample["features"])
        result = mm.predict_uv(uid, feats)
        print(f"UV test (user={uid}): {result}")
    print("ModelManager ready.")

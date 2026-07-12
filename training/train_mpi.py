"""
MPI training. Fully resumable.
Paper: Section 5.1 of arXiv:2302.01751
Single GPU only — no DataParallel.
Run: python training/train_mpi.py --use-dummy
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import cfg
from models.uv_model import MPIModel

PROCESSED_DIR = "data/mpi/processed"
CKPT_DIR      = "models/checkpoints"
RESULTS_DIR   = "evaluation"
RESUME_LOG    = os.path.join(RESULTS_DIR, "mpi_completed.csv")

os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_completed() -> set:
    if os.path.exists(RESUME_LOG):
        return set(map(tuple,
            pd.read_csv(RESUME_LOG)[["user_id","device_id","seed"]]
              .values.tolist()))
    return set()


def mark_completed(uid, did, seed):
    row = pd.DataFrame([{"user_id": uid, "device_id": did, "seed": seed}])
    row.to_csv(RESUME_LOG, mode="a",
               header=not os.path.exists(RESUME_LOG), index=False)


def train_one_pair(X, y, uid, did, seed: int) -> tuple:
    torch.manual_seed(seed); np.random.seed(seed)

    # Stratified 70 / 15 / 15 split by sample
    sss1 = StratifiedShuffleSplit(1, test_size=0.30, random_state=seed)
    idx_tr, idx_tmp = next(sss1.split(X, y))
    sss2 = StratifiedShuffleSplit(1, test_size=0.50, random_state=seed)
    idx_v, idx_te   = next(sss2.split(X[idx_tmp], y[idx_tmp]))
    idx_v  = idx_tmp[idx_v]
    idx_te = idx_tmp[idx_te]

    model = MPIModel(n_channels=X.shape[1], n_classes=2).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.baseline_lr)
    crit  = nn.CrossEntropyLoss()

    def make_loader(Xa, ya, shuffle=True):
        return DataLoader(
            TensorDataset(torch.tensor(Xa),
                          torch.tensor(ya, dtype=torch.long)),
            batch_size=32, shuffle=shuffle)

    tr_lo = make_loader(X[idx_tr], y[idx_tr])
    va_lo = make_loader(X[idx_v],  y[idx_v],  shuffle=False)
    te_lo = make_loader(X[idx_te], y[idx_te], shuffle=False)

    best_auc, best_ckpt = -1.0, None
    for epoch in range(cfg.mpi_epochs):
        model.train()
        for Xb, yb in tr_lo:
            opt.zero_grad()
            crit(model(Xb.to(device)), yb.to(device)).backward()
            opt.step()

        model.eval(); probs, ys = [], []
        with torch.no_grad():
            for Xb, yb in va_lo:
                probs.extend(
                    torch.softmax(model(Xb.to(device)), 1)[:, 1].cpu().numpy())
                ys.extend(yb.numpy())
        try:    auc = roc_auc_score(ys, probs)
        except: auc = 0.5
        if auc > best_auc:
            best_auc  = auc
            best_ckpt = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_ckpt); model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for Xb, yb in te_lo:
            preds.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
            ys.extend(yb.numpy())
    return accuracy_score(ys, preds) * 100.0, np.array(ys), np.array(preds)


def run_on_dummy():
    from utils.dummy_data import load_mpi_dummy
    X, y, _ = load_mpi_dummy()
    print(f"Dummy MPI training: X={X.shape}, device={device}")
    for seed in range(2):
        acc, _, _ = train_one_pair(X, y, 0, 0, seed)
        print(f"  seed={seed}: acc={acc:.2f}%")
    print("Dummy MPI training passed.")


def main(use_dummy=False):
    if use_dummy:
        run_on_dummy(); return

    mf    = pd.read_csv(os.path.join(PROCESSED_DIR, "manifest.csv"))
    valid = mf[mf["status"] == "OK"]
    print(f"Training on {len(valid)} valid pairs.  Device: {device}")
    completed = load_completed()

    out_path = os.path.join(RESULTS_DIR, "results_mpi.csv")
    all_rows = (pd.read_csv(out_path).to_dict("records")
                if os.path.exists(out_path) else [])

    for _, row in valid.iterrows():
        uid, did = int(row["user_id"]), int(row["device_id"])
        data     = np.load(os.path.join(PROCESSED_DIR, f"{uid}_{did}.npz"))
        X, y     = data["X"], data["y"]
        print(f"\nUser={uid} Device={did}: X={X.shape}")

        seed_accs = []
        for seed in range(cfg.n_training_runs):
            if (uid, did, seed) in completed:
                print(f"  seed={seed}: already done — skip"); continue
            acc, _, _ = train_one_pair(X, y, uid, did, seed)
            seed_accs.append(acc)
            print(f"  seed={seed}: acc={acc:.2f}%")
            mark_completed(uid, did, seed)

        if seed_accs:
            all_rows.append({
                "user_id":   uid,   "device_id": did,
                "mean_acc":  round(np.mean(seed_accs), 2),
                "std_acc":   round(np.std(seed_accs),  2),
                "status":    "OK"})
            pd.DataFrame(all_rows).to_csv(out_path, index=False)
            print(f"  → {np.mean(seed_accs):.1f} ± {np.std(seed_accs):.1f}%")

    from evaluation.evaluate import print_table1
    print_table1(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-dummy", action="store_true")
    args = parser.parse_args()
    main(use_dummy=args.use_dummy)

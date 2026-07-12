"""
UV training: baseline → fine-tune → bootstrap final test. Fully resumable.
Paper: Sections 6.3-6.5 of arXiv:2302.01751
Single GPU only — no DataParallel.
Run: python training/train_uv.py --use-dummy
     python training/train_uv.py --n 75
     python training/train_uv.py --all-splits
"""

import os, sys, argparse, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import cfg
from models.uv_model import UVModel
from models.losses import TotalLoss
from evaluation.evaluate import (compute_far_at_tar, bootstrap_far,
                                  format_far, print_table2, print_table3)

PROCESSED_DIR = "data/uv/processed"
CKPT_DIR      = "models/checkpoints"
RESULTS_DIR   = "evaluation"
os.makedirs(CKPT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────

class UVDataset(Dataset):
    """user_features: {uid: (N,22,4,T)}   user_label_map: {uid: int}"""
    def __init__(self, user_features: dict, user_label_map: dict):
        self.samples, self.labels = [], []
        for uid, feats in user_features.items():
            lbl = user_label_map[uid]
            for trial in feats:
                self.samples.append(trial.astype(np.float32))
                self.labels.append(lbl)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        return (torch.tensor(self.samples[i]),
                torch.tensor(self.labels[i], dtype=torch.long))


# ── Helpers ───────────────────────────────────────────────────────────────

def load_all_users(processed_dir: str) -> dict:
    data = {}
    for fname in sorted(os.listdir(processed_dir)):
        if not fname.endswith(".npz"): continue
        try:    uid = int(os.path.splitext(fname)[0])
        except: continue
        data[uid] = np.load(os.path.join(processed_dir, fname))["features"]
    return data


def split_attempts(feats: np.ndarray, seed: int):
    """Split 300 attempts 70/15/15 by index (online approach — no user split)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(feats.shape[0])
    n   = len(idx); n_tr = int(n * 0.70); n_v = int(n * 0.15)
    return idx[:n_tr], idx[n_tr:n_tr+n_v], idx[n_tr+n_v:]


def score_verification(model: nn.Module,
                        X_genuine:  np.ndarray,
                        X_impostor: np.ndarray) -> tuple:
    """
    Cosine similarity against centroid template of genuine embeddings.
    Returns (genuine_scores, impostor_scores).
    """
    if len(X_genuine) == 0 or len(X_impostor) == 0:
        return np.array([]), np.array([])
    model.eval()
    with torch.no_grad():
        ge = model.get_siamese_embed(
            torch.tensor(X_genuine,  dtype=torch.float32).to(device)).cpu().numpy()
        ie = model.get_siamese_embed(
            torch.tensor(X_impostor, dtype=torch.float32).to(device)).cpu().numpy()
    tmpl = ge.mean(0, keepdims=True)
    tmpl /= (np.linalg.norm(tmpl) + 1e-8)
    gn = ge / (np.linalg.norm(ge, axis=1, keepdims=True) + 1e-8)
    in_ = ie / (np.linalg.norm(ie, axis=1, keepdims=True) + 1e-8)
    return (gn @ tmpl.T).squeeze(), (in_ @ tmpl.T).squeeze()


# ── Step 1: Baseline ──────────────────────────────────────────────────────

def train_baseline(all_users: dict, n_baseline: int, seed: int) -> tuple:
    """
    Paper Section 6.5 Step 1: train baseline on n_baseline users,
    split by attempts (online approach).
    Returns (model, acc_val, acc_test, far_val, far_test)
    """
    all_uids    = sorted(all_users.keys())
    subset_base = all_uids[:n_baseline]
    torch.manual_seed(seed); np.random.seed(seed)

    tr_f, vf, tf = {}, {}, {}
    for uid in subset_base:
        f = all_users[uid]
        it, iv, ite = split_attempts(f, seed)
        tr_f[uid] = f[it]; vf[uid] = f[iv]; tf[uid] = f[ite]

    uid2cls = {uid: i for i, uid in enumerate(subset_base)}
    tr_lo   = DataLoader(UVDataset(tr_f, uid2cls),
                         batch_size=cfg.batch_size, shuffle=True,  drop_last=True)
    va_lo   = DataLoader(UVDataset(vf,   uid2cls),
                         batch_size=cfg.batch_size, shuffle=False)
    te_lo   = DataLoader(UVDataset(tf,   uid2cls),
                         batch_size=cfg.batch_size, shuffle=False)

    model   = UVModel(n_classes=n_baseline).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=cfg.baseline_lr)
    loss_fn = TotalLoss(cfg.alpha_tm, cfg.supcon_temperature)

    best_acc, best_ckpt = 0.0, None
    for epoch in range(cfg.baseline_epochs):
        model.train()
        for Xb, yb in tr_lo:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, p1 = model(Xb, augment=True)
            _,      p2 = model(Xb, augment=True)
            loss, _ = loss_fn(logits, torch.cat([p1, p2], 0), yb)
            loss.backward(); opt.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for Xb, yb in va_lo:
                pred    = model(Xb.to(device))[0].argmax(1).cpu()
                correct += (pred == yb).sum().item()
                total   += yb.size(0)
        acc = correct / total * 100
        if acc > best_acc:
            best_acc  = acc
            best_ckpt = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_ckpt); model.eval()
    correct = total = 0
    with torch.no_grad():
        for Xb, yb in te_lo:
            pred    = model(Xb.to(device))[0].argmax(1).cpu()
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
    acc_test = correct / total * 100

    # Sample FAR for Table 2 reporting
    g_v, i_v = score_verification(
        model, all_users[subset_base[0]][:30], all_users[subset_base[1]][:30])
    far_val  = compute_far_at_tar(g_v, i_v, cfg.tar_threshold)[0] \
               if len(g_v) > 0 else 1.0
    g_t, i_t = score_verification(
        model,
        all_users[subset_base[2]][:30] if len(subset_base) > 2 else all_users[subset_base[0]][:30],
        all_users[subset_base[3]][:30] if len(subset_base) > 3 else all_users[subset_base[1]][:30])
    far_test = compute_far_at_tar(g_t, i_t, cfg.tar_threshold)[0] \
               if len(g_t) > 0 else 1.0

    return model, best_acc, acc_test, far_val, far_test


# ── Step 2: Fine-tune ─────────────────────────────────────────────────────

def finetune_user(baseline_model: nn.Module,
                  target_uid:     int,
                  all_users:      dict,
                  subset_base:    list,
                  val_add:        list,
                  seed:           int) -> tuple:
    """
    Paper Section 6.5 Step 2:
    - Freeze feature extractor (branches)
    - Replace classifier with 2-class head
    - Mixed DataLoader: class 0 = subsetbase impostors, class 1 = target user
    - Epoch selection using val_add
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = copy.deepcopy(baseline_model)

    # Freeze all 22 branches
    for p in model.branches.parameters():
        p.requires_grad = False

    # Replace classifier: 2 classes  (BUG FIX: explicit .to(device))
    model.head_a = nn.Linear(model.embed_dim, 2).to(device)
    model         = model.to(device)

    trainable = (list(model.head_a.parameters()) +
                 list(model.head_b.parameters()) +
                 list(model.siamese_proj.parameters()))
    opt  = torch.optim.Adam(trainable, lr=cfg.finetune_lr)
    crit = nn.CrossEntropyLoss()

    # Mixed DataLoader
    rng   = np.random.default_rng(seed)
    n_per = 150
    imp   = np.concatenate(
        [all_users[u][rng.choice(len(all_users[u]),
                                 max(1, n_per // len(subset_base)),
                                 replace=False)]
         for u in subset_base], axis=0)[:n_per]
    own   = all_users[target_uid][:n_per]
    X_mix = np.concatenate([imp, own], axis=0)
    y_mix = np.array([0]*len(imp) + [1]*len(own), dtype=np.int64)
    shuf  = rng.permutation(len(X_mix))
    X_mix, y_mix = X_mix[shuf], y_mix[shuf]
    ft_lo = DataLoader(
        TensorDataset(torch.tensor(X_mix), torch.tensor(y_mix)),
        batch_size=32, shuffle=True)

    # Val-add for epoch selection
    val_imp = (np.concatenate([all_users[u][:9] for u in val_add[:10]], 0)
               if val_add else imp[:90])
    val_gen = all_users[target_uid][:90]

    best_far, best_ckpt = 1.0, None
    for epoch in range(cfg.finetune_epochs):
        model.train()
        for Xb, yb in ft_lo:
            opt.zero_grad()
            crit(model(Xb.to(device))[0], yb.to(device)).backward()
            opt.step()
        g, i = score_verification(model, val_gen, val_imp)
        if len(g) > 0:
            fv = compute_far_at_tar(g, i, cfg.tar_threshold)[0]
            if fv < best_far:
                best_far  = fv
                best_ckpt = {k: v.clone() for k, v in model.state_dict().items()}

    if best_ckpt:
        model.load_state_dict(best_ckpt)
    return model, best_far


# ── Step 3: Bootstrap final test ─────────────────────────────────────────

def final_test_user(model:           nn.Module,
                    target_uid:      int,
                    all_users:       dict,
                    testfinal_uids:  list) -> dict:
    """
    Paper Section 6.5 Step 4: bootstrap 5000 repeats,
    90 genuine + 90 impostor attempts.
    """
    gen = all_users[target_uid][:90]
    imp = np.concatenate(
        [all_users[u][:9] for u in testfinal_uids if u != target_uid],
        axis=0)[:90]
    g, i = score_verification(model, gen, imp)
    if len(g) == 0:
        return {"mean_far": 1.0, "std_far": 0.0}
    m, s = bootstrap_far(g, i, cfg.bootstrap_repeats,
                         cfg.tar_threshold, seed=target_uid)
    d, f = format_far(m)
    print(f"  user={target_uid}: FAR={m*100:.1f}±{s*100:.1f}%  ({d}, {f})")
    return {"mean_far": m, "std_far": s}


# ── Per-split runner ──────────────────────────────────────────────────────

def run_split(n_baseline: int, all_users: dict, all_uids: list):
    # Re-derive deterministically
    subset_base     = all_uids[:n_baseline]
    val_add         = all_uids[n_baseline:90]
    testfinal_uids  = all_uids[90:101]

    b_csv = os.path.join(RESULTS_DIR, "results_baseline.csv")
    f_csv = os.path.join(RESULTS_DIR, "results_uv_final.csv")
    b_rows = pd.read_csv(b_csv).to_dict("records") if os.path.exists(b_csv) else []
    f_rows = pd.read_csv(f_csv).to_dict("records") if os.path.exists(f_csv) else []

    # ── Baseline: 5 seeds, resumable ─────────────────────────────────────
    print(f"\n── n={n_baseline}: Baseline ({cfg.n_training_runs} seeds) ──")
    seed_models, avs, ats, fvs, fts = [], [], [], [], []

    for seed in range(cfg.n_training_runs):
        ckpt_path = os.path.join(CKPT_DIR,
                                 f"baseline_n{n_baseline}_seed{seed}.pt")
        if os.path.exists(ckpt_path):
            print(f"  seed={seed}: loading checkpoint")
            c = torch.load(ckpt_path, map_location=device)
            m = UVModel(n_classes=n_baseline).to(device)
            m.load_state_dict(c["model_state"])
            seed_models.append(m)
            avs.append(c["best_acc"]);   ats.append(c.get("acc_test", 0))
            fvs.append(c.get("far_val", 1.0)); fts.append(c.get("far_test", 1.0))
            continue

        print(f"  seed={seed}: training...")
        m, av, at, fv, ft = train_baseline(all_users, n_baseline, seed)
        torch.save({
            "model_state": m.state_dict(),
            "best_acc":    av,  "acc_test": at,
            "far_val":     fv,  "far_test": ft,
        }, ckpt_path)
        seed_models.append(m)
        avs.append(av); ats.append(at); fvs.append(fv); fts.append(ft)
        print(f"  seed={seed}: acc_val={av:.1f}%  far_val={fv:.4f}")

    b_rows.append({
        "n_baseline":    n_baseline,
        "acc_val_mean":  np.mean(avs), "acc_val_std":  np.std(avs),
        "acc_test_mean": np.mean(ats), "acc_test_std": np.std(ats),
        "far_val_mean":  np.mean(fvs), "far_val_std":  np.std(fvs),
        "far_test_mean": np.mean(fts), "far_test_std": np.std(fts),
    })
    pd.DataFrame(b_rows).to_csv(b_csv, index=False)

    chosen_seed = int(np.argsort(avs)[len(avs) // 2])
    best_model  = seed_models[chosen_seed]
    print(f"  Chosen seed for fine-tuning: {chosen_seed}")

    # ── Fine-tune + test: resumable per user ─────────────────────────────
    print(f"\n── n={n_baseline}: Fine-tune + test ──")
    done = {r["user_id"]
            for r in f_rows if r.get("n_baseline") == n_baseline}

    for target_uid in testfinal_uids:
        if target_uid in done:
            print(f"  user={target_uid}: done — skip"); continue

        ft_ckpt = os.path.join(CKPT_DIR,
                               f"finetune_user{target_uid}_n{n_baseline}.pt")
        if os.path.exists(ft_ckpt):
            print(f"  user={target_uid}: loading fine-tune checkpoint")
            c  = torch.load(ft_ckpt, map_location=device)
            ft = UVModel(n_classes=2).to(device)
            ft.load_state_dict(c["model_state"])
        else:
            print(f"  user={target_uid}: fine-tuning...")
            ft, fv = finetune_user(
                best_model, target_uid, all_users,
                subset_base, val_add, chosen_seed)
            torch.save({"model_state": ft.state_dict()}, ft_ckpt)

        result = final_test_user(ft, target_uid, all_users, testfinal_uids)
        f_rows.append({
            "user_id":    target_uid,
            "n_baseline": n_baseline,
            "far_mean":   result["mean_far"],
            "far_std":    result["std_far"],
        })
        pd.DataFrame(f_rows).to_csv(f_csv, index=False)


# ── Dummy mode ────────────────────────────────────────────────────────────

def run_on_dummy():
    from utils.dummy_data import load_uv_dummy
    X_all, _ = load_uv_dummy()
    print(f"Dummy UV: X={X_all.shape}  device={device}")

    orig = (cfg.baseline_epochs, cfg.finetune_epochs, cfg.bootstrap_repeats)
    cfg.baseline_epochs    = 2
    cfg.finetune_epochs    = 2
    cfg.bootstrap_repeats  = 10

    au = {i: X_all[i] for i in range(15)}
    m, av, at, fv, ft = train_baseline(
        {i: X_all[i] for i in range(9)}, n_baseline=6, seed=0)
    print(f"  baseline: acc_val={av:.1f}%  acc_test={at:.1f}%")

    sb = list(range(6)); va = list(range(6, 9))
    ft_m, fv2 = finetune_user(m, 9, au, sb, va, seed=0)
    print(f"  finetune FAR_val={fv2:.4f}")

    r = final_test_user(ft_m, 9, au, [9, 10])
    print(f"  bootstrap result: {r}")

    cfg.baseline_epochs, cfg.finetune_epochs, cfg.bootstrap_repeats = orig
    print("Dummy UV training passed.")


# ── Main ──────────────────────────────────────────────────────────────────

def main(use_dummy=False, n_baseline=None, all_splits=False):
    if use_dummy:
        run_on_dummy(); return

    print(f"Device: {device}")
    all_users = load_all_users(PROCESSED_DIR)
    all_uids  = sorted(all_users.keys())
    assert len(all_uids) >= 90, f"Need >=90 users, got {len(all_uids)}"

    splits = cfg.uv_n_splits if all_splits else [n_baseline or cfg.uv_baseline_n]
    for n in splits:
        run_split(n, all_users, all_uids)

    b_csv = os.path.join(RESULTS_DIR, "results_baseline.csv")
    f_csv = os.path.join(RESULTS_DIR, "results_uv_final.csv")
    print_table2(b_csv)
    print_table3(f_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-dummy",   action="store_true")
    parser.add_argument("--n",           type=int, default=None)
    parser.add_argument("--all-splits",  action="store_true")
    args = parser.parse_args()
    main(use_dummy=args.use_dummy,
         n_baseline=args.n,
         all_splits=args.all_splits)

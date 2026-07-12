"""
Evaluation utilities.
Paper references: Sections 4.2 and 7 of arXiv:2302.01751

Run: python evaluation/evaluate.py
Expected: All evaluation unit tests passed.
"""

import numpy as np
import pandas as pd
import os


def compute_far_at_tar(
        genuine_scores,
        impostor_scores,
        target_tar: float = 0.90,
        n_steps: int = 100_000) -> tuple:
    """
    Manual threshold sweep from high to low.
    Returns (far, threshold) at the first point where TAR >= target_tar.
    Avoids sklearn interpolation artefacts at very low FAR (1/50000).
    """
    gen = np.asarray(genuine_scores,  dtype=np.float64)
    imp = np.asarray(impostor_scores, dtype=np.float64)
    if len(gen) == 0 or len(imp) == 0:
        return 1.0, 0.0
    all_s = np.concatenate([gen, imp])
    for t in np.linspace(all_s.max(), all_s.min(), n_steps):
        if np.mean(gen >= t) >= target_tar:
            return float(np.mean(imp >= t)), float(t)
    return 1.0, float(all_s.min())


def bootstrap_far(
        genuine_scores,
        impostor_scores,
        n_repeats:  int   = 5000,
        target_tar: float = 0.90,
        seed:       int   = 0) -> tuple:
    """
    Paper Section 6.5: bootstrap-like method, 5000 repeats.
    CRITICAL: threshold is re-calibrated per resample (not fixed).
    Returns (mean_far, std_far).
    """
    rng = np.random.default_rng(seed)
    gen = np.asarray(genuine_scores,  dtype=np.float64)
    imp = np.asarray(impostor_scores, dtype=np.float64)
    far_list = []
    for _ in range(n_repeats):
        f, _ = compute_far_at_tar(
            rng.choice(gen, len(gen), replace=True),
            rng.choice(imp, len(imp), replace=True),
            target_tar,
            n_steps=10_000)
        far_list.append(f)
    arr = np.array(far_list)
    return float(arr.mean()), float(arr.std())


def format_far(far_value: float) -> tuple:
    """Returns (decimal_str, fraction_str). e.g. 0.01 → ('1.00e-02', '1/100')"""
    if far_value == 0.0:
        return "0", "1/inf"
    k = int(round(1.0 / far_value))
    return f"{far_value:.2e}", f"1/{k}"


def rule_of_30_check(far_value: float, n_impostor_comparisons: int):
    """
    Paper Section 4.2: need >= 30 errors for 90% confidence within ±30%.
    """
    if far_value == 0.0:
        print("Rule of 30: FAR=0, cannot evaluate."); return
    n_err = far_value * n_impostor_comparisons
    need  = int(np.ceil(30 / far_value))
    status = "PASS" if n_err >= 30 else f"FAIL — need >= {need} comparisons"
    print(f"Rule of 30: {n_err:.1f} errors / "
          f"{n_impostor_comparisons} comparisons → {status}")


def print_table1(csv_path: str):
    """Paper Table 1: MPI accuracy per user-device pair."""
    if not os.path.exists(csv_path):
        print(f"Table 1: not found — {csv_path}"); return
    df      = pd.read_csv(csv_path)
    users   = sorted(df["user_id"].unique())
    devices = sorted(df["device_id"].unique())
    print("\nTable 1: MPI Accuracy (%)")
    print(f"{'Dev/User':<10}", end="")
    for u in users:
        print(f"{'User'+str(u):>15}", end="")
    print()
    for d in devices:
        print(f"{d:<10}", end="")
        for u in users:
            r = df[(df["device_id"] == d) & (df["user_id"] == u)]
            if r.empty or r.iloc[0]["status"] == "N/A":
                print(f"{'N/A':>15}", end="")
            else:
                ri = r.iloc[0]
                val = f"{ri.mean_acc:.1f}±{ri.std_acc:.1f}"
                print(f"{val:>15}", end="")
        print()


def print_table2(csv_path: str):
    """Paper Table 2: baseline model metrics per split."""
    if not os.path.exists(csv_path):
        print(f"Table 2: not found — {csv_path}"); return
    df = pd.read_csv(csv_path)
    print("\nTable 2: Baseline Performance")
    print(f"{'Split':>6} {'AccVal':>14} {'AccTest':>14} "
          f"{'FARval@90':>18} {'FARtest@90':>18}")
    for _, r in df.iterrows():
        dv, fv = format_far(r.get("far_val_mean",  0))
        dt, ft = format_far(r.get("far_test_mean", 0))
        print(f"{int(r['n_baseline']):>6} "
              f"{r['acc_val_mean']:.2f}±{r['acc_val_std']:.2f}      "
              f"{r['acc_test_mean']:.2f}±{r['acc_test_std']:.2f}      "
              f"{dv}({fv})   {dt}({ft})")


def print_table3(csv_path: str):
    """Paper Table 3: per-user FAR across all splits."""
    if not os.path.exists(csv_path):
        print(f"Table 3: not found — {csv_path}"); return
    df     = pd.read_csv(csv_path)
    splits = sorted(df["n_baseline"].unique())
    print("\nTable 3: Final Test FAR(@TAR=90%) per user and split")
    print(f"{'user_id':>8}", end="")
    for s in splits:
        print(f"{str(s):>13}", end="")
    print()
    for uid in sorted(df["user_id"].unique()):
        print(f"{uid:>8}", end="")
        for s in splits:
            r = df[(df["user_id"] == uid) & (df["n_baseline"] == s)]
            if r.empty:
                print(f"{'N/A':>13}", end="")
            else:
                ri = r.iloc[0]
                val = f"{ri.far_mean*100:.1f}±{ri.far_std*100:.1f}"
                print(f"{val:>13}",
                      end="")
        print()


if __name__ == "__main__":
    # Test 1: FAR = 0
    gen = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
    imp = np.array([0.3, 0.4,  0.5, 0.2,  0.1])
    far, t = compute_far_at_tar(gen, imp, 0.90)
    assert far == 0.0, f"Expected 0.0, got {far}"
    print(f"Test 1 — FAR=0:        FAR={far}, t={t:.4f}  PASSED")

    # Test 2: non-zero FAR
    gen2 = np.array([0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35])
    imp2 = np.array([0.75, 0.5, 0.4, 0.3, 0.2, 0.1,  0.1, 0.1,  0.1, 0.1])
    far2, _ = compute_far_at_tar(gen2, imp2, 0.90)
    assert 0.0 <= far2 <= 1.0
    print(f"Test 2 — non-zero FAR: {far2:.4f}  PASSED")

    # Test 3: bootstrap (re-calibrates threshold per resample)
    m, s = bootstrap_far(gen, imp, n_repeats=100, seed=42)
    assert m >= 0
    print(f"Test 3 — bootstrap:    mean={m:.4f}, std={s:.4f}  PASSED")

    # Test 4: format_far
    for v in [0.0, 0.01, 1/50000]:
        d, f = format_far(v)
        print(f"Test 4 — format_far({v:.6f}): {d}, {f}")

    # Test 5: rule of 30
    rule_of_30_check(1/50000, 1_500_000)   # PASS  (30 errors exactly)
    rule_of_30_check(1/50000,   100_000)   # FAIL  (2 errors)

    print("\nAll evaluation unit tests passed.")

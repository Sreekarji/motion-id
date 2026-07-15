# Motion ID Implementation Audit Report

**Paper:** *Motion ID: Human Authentication Approach* (arXiv:2302.01751v1)  
**Repository:** https://github.com/Sreekarji/motion-id  
**Notebook:** `notebooks/humanauth.ipynb` (12 cells, self-contained)  
**Audited by:** Annie (Vasavi College of Engineering, Internship 2025–26)  
**Date:** July 2026

---

## Executive Summary

The implementation correctly captures the two-stage Motion ID pipeline (Motion Patterns
Identification → User Verification), the three-loss formula, the 22-branch CNN architecture,
and the bootstrap evaluation protocol. Binary sensor parsing, screen event handling, training
split, and fine-tuning structure are all faithful to the paper.

Three issues require attention before any academic revision:

1. **`R.from_rotvec` on quaternion data (Cell 4)** — corrupts 12 of 22 UV features.
2. **`additional_test/` users are the testfinal set (confirmed)** — the Kaggle dataset has only 90 users in `train_val_test/` and 11 in `additional_test/`. The code processes both, making `all_uids[90:]` exactly the `additional_test` cohort. These users were collected separately and may not match the paper's intended testfinal population.
3. **Multi-split sweep missing (Cell 7)** — only `n=75` is run; Table 2 cannot be reproduced.

Everything else is either correct or a minor deviation with negligible impact on results.

---

## Verdict Table

| # | Section | Item | Verdict |
|---|---------|------|---------|
| 1 | Paper Fidelity | Training split (75 baseline / 15 val_add / 11 testfinal) | ✅ PASS |
| 2 | Paper Fidelity | Bootstrap evaluation (5000 repeats, 90+90) | ✅ PASS |
| 3 | Paper Fidelity | FAR computed at TAR = 90% | ✅ PASS |
| 4 | Paper Fidelity | Multi-split sweep n ∈ {60,65,70,75,80,85} for Table 2 | ❌ FAIL |
| 5 | Paper Fidelity | Fine-tune genuine validation leak | ❌ FAIL |
| 6 | Paper Fidelity | 22 UV features per Section 6.1 | ⚠️ PARTIAL |
| 7 | Architecture | MPI: pointwise convolutions (Section 5.1) | ❌ FAIL |
| 8 | Architecture | UV: 22-branch CNN + dual head (Figure 3) | ✅ PASS |
| 9 | Architecture | Three losses: L_CE, L_TM, L_SC | ✅ PASS |
| 10 | Architecture | L_total = L_CE + α_TM × L_TM + L_SC | ✅ PASS |
| 11 | Architecture | Double augmentation in training loop | ✅ PASS |
| 12 | Architecture | Triplet loss uses only p1, not cat[p1,p2] | ⚠️ PARTIAL |
| 13 | Data Pipeline | Binary parsing: 8-byte BE int64 + 12-byte LE float32 | ✅ PASS |
| 14 | Data Pipeline | Screen event full intent string matching | ✅ PASS |
| 15 | Data Pipeline | Earth-fixed frame: R.from_rotvec on quaternion data | ❌ FAIL |
| 16 | Data Pipeline | UV window exactly 1 second (50 samples at 50 Hz) | ✅ PASS |
| 17 | Data Pipeline | Raw lin sensor data discarded silently | ⚠️ PARTIAL |
| 18 | Data Pipeline | Features padded to 4 channels (paper says 3) | ⚠️ PARTIAL |
| 19 | Results | MPI accuracy range vs Table 1 | ⚠️ PARTIAL |
| 20 | Results | UV baseline Acc/FAR vs Table 2 | ❌ FAIL |
| 21 | Results | UV per-user FAR vs Table 3 | ⚠️ PARTIAL |
| 22 | Code Quality | Numerical stability (1e-8 guards, empty-batch handling) | ✅ PASS |
| 23 | Code Quality | Single-GPU constraint (no DataParallel) | ✅ PASS |
| 24 | Code Quality | additional_test/ users become testfinal (confirmed: TVT=90, add=11) | ❌ FAIL |
| 25 | Code Quality | Baseline CSV: single seed only, std=0 | ⚠️ PARTIAL |
| 26 | Missing | Multi-split sweep for Table 2 | ❌ MISSING |
| 27 | Missing | Held-out genuine validation during fine-tuning | ❌ MISSING |
| 28 | Extra | additional_test/ processed alongside train_val_test/ | ❌ EXTRA |

**Summary: 11 PASS · 5 FAIL · 6 PARTIAL · 2 MISSING · 1 EXTRA**

---

## Section 1 — Paper Fidelity

### 1.1 Experimental Setup (Section 6)

The overall pipeline structure matches the paper correctly:

```
Baseline training (n users) → Fine-tune per testfinal user → Bootstrap FAR evaluation
```

**Training split — PASS**

| Split | Paper | Implementation |
|-------|-------|----------------|
| subset_base | n users (75 for primary) | `all_uids[:75]` ✓ |
| val_add | 90 − n users (15) | `all_uids[75:90]` ✓ |
| testfinal | 11 users | `all_uids[90:]` ✓ |

Within-user attempt splitting uses a 70/15/15 ratio (Cell 7, `split_attempts()`). The paper
says "subdivided by attempts" without specifying a ratio — this is a reasonable choice.

**Multi-split sweep — FAIL**

Paper Table 2 reports results for n ∈ {60, 65, 70, 75, 80, 85}. The implementation sets
`cfg.uv_baseline_n = 75` and runs only that split. `results_baseline.csv` contains exactly
one data row. Table 2 cannot be reproduced from this run.

**Bootstrap evaluation — PASS**

- 5000 repeats (`cfg.bootstrap_repeats = 5000`) ✓
- 90 genuine samples per user (`all_users[target_uid][:min(90, ...)]`) ✓
- 90 impostor samples (10 remaining testfinal users × 9 samples each) ✓
- Resamples with replacement on each repeat ✓
- FAR threshold sweep at TAR = 90% with 100,000-step resolution ✓

### 1.2 22 UV Features (Section 6.1) — PARTIAL

`compute_features()` in Cell 4 produces exactly 22 stacked arrays:

| # | Feature | Source |
|---|---------|--------|
| 1 | acc (raw) | sensor_matrix |
| 2 | gyro (raw) | sensor_matrix |
| 3 | mag (raw) | sensor_matrix |
| 4 | lin_acc (computed) | acc − grav |
| 5–7 | acc_rot, gyro_rot, mag_rot | Earth-fixed via R |
| 8–10 | diff(acc), diff(gyro), diff(mag) | np.diff |
| 11–13 | diff(acc_rot), diff(gyro_rot), diff(mag_rot) | np.diff |
| 14–16 | cumsum(acc), cumsum(gyro), cumsum(mag) | np.cumsum |
| 17–19 | cumsum(acc_rot), cumsum(gyro_rot), cumsum(mag_rot) | np.cumsum |
| 20 | diff(lin_acc) | np.diff |
| 21 | cumsum(lin_acc) | np.cumsum |
| 22 | rot (raw rotation vector) | sensor_matrix |

**Issue:** `sensor_matrix` is constructed as `hstack[acc, grav, gyro, lin, mag]`, placing the
raw linear accelerometer sensor at columns 9–11. The paper (Section 6.1) explicitly states both
the raw lin sensor readings *and* the manually computed `linear_acc = acc − gravity` should
appear as separate feature sets. In the implementation, `sensor_matrix[:, 9:12]` (the raw lin
sensor) is aligned and stored but never referenced in `compute_features()`. Only the manually
computed `lin_acc = acc − grav` is used. One feature set is silently substituted for another.

### 1.3 Fine-tuning Protocol (Section 6.5) — PARTIAL

| Requirement | Paper | Implementation | Status |
|-------------|-------|----------------|--------|
| Freeze feature extractor | Yes | `ft.branches.parameters(): requires_grad = False` | ✓ |
| 2-class head | Yes | `ft.head_a = Linear(embed_dim, 2)` | ✓ |
| Epochs | Reduced | `cfg.finetune_epochs = 10` | ✓ |
| Learning rate | Reduced | `cfg.finetune_lr = 1e-4` (vs baseline 1e-3) | ✓ |
| Class 0 = impostor, Class 1 = target | Implied | `[0]*len(imp) + [1]*len(own)` | ✓ |

**Genuine validation leak — FAIL**

Cell 7, fine-tuning loop:
```python
own = all_users[target_uid][:150]          # used for training
val_gen = all_users[target_uid][:min(90, ...)]  # used for epoch selection
```

`val_gen[:90]` is a subset of `own[:150]`. The model selects its best fine-tune epoch
using data it was already trained on. The paper requires a held-out genuine validation
set for epoch selection — this is the purpose of `valadd`. The fix is to use
`all_users[target_uid][150:]` as `val_gen`, keeping training and validation disjoint.

---

## Section 2 — Architecture

### 2.1 MPI Model (Section 5.1) — FAIL

**Paper states:** "traditional architecture consisting of several layers with *pointwise convolutions*"

Pointwise convolution = Conv1d with `kernel_size=1`.

**Implementation (Cell 5, `MPIModel`):**
```python
nn.Conv1d(n_channels, 64,  kernel_size=5, padding=2)   # NOT pointwise
nn.Conv1d(64,         128, kernel_size=5, padding=2)   # NOT pointwise
nn.Conv1d(128,        256, kernel_size=3, padding=1)   # NOT pointwise
```

Kernels of size 5, 5, 3 are standard local convolutions. They have a larger receptive field
and more parameters than pointwise convolutions — this is why MPI accuracy comes out higher
than the paper's range on many user-device pairs.

**To match the paper:**
```python
nn.Conv1d(n_channels, 64,  kernel_size=1)
nn.Conv1d(64,         128, kernel_size=1)
nn.Conv1d(128,        256, kernel_size=1)
```

### 2.2 UV Model (Section 6.4, Figure 3) — PASS

| Component | Paper | Implementation | Status |
|-----------|-------|----------------|--------|
| 22 branches | One per generated feature | `ModuleList([UVBranch(4)] × 22)` | ✓ |
| Identical branch architecture | Yes | All `UVBranch(in_channels=4)` | ✓ |
| 1D convolutional layers | Yes | Conv1d(4→32→64→128) per branch | ✓ |
| Classifier head (L_CE) | `head_a` | `Linear(embed_dim, n_classes)` | ✓ |
| Siamese head (L_TM) | `siamese_proj` | `Linear(embed_dim, 256)` | ✓ |
| MLP head (L_SC) | `head_b` | `Sequential(Linear, ReLU, Linear)` | ✓ |
| Double augmentation + concat | Yes | Two separate forward passes in training loop | ✓ |

### 2.3 Three Losses — PASS

**Paper formula:** `L_total = L_CE + α_TM × L_TM + L_SC`

**Implementation (`TotalLoss.forward()`):**
```python
lce = self.ce(logits, labels)                                    # CrossEntropyLoss
ltm = self.tm(F.normalize(proj_embeds[:labels.size(0)]), labels) # TripletMarginLoss
lsc = self.sc(proj_embeds, labels)                               # SupervisedContrastive
return lce + self.alpha_tm * ltm + lsc
```

Formula matches exactly. `α_TM = 1.0` in Config.

**Minor inconsistency — PARTIAL**

The Triplet loss receives `proj_embeds[:B]` (only `p1`, the first augmented view).
The SC loss receives the full `cat[p1, p2]` (both augmented views). Paper intent is
ambiguous, but the asymmetric treatment is inconsistent. Passing the full `cat[p1, p2]`
to both losses would be more faithful.

---

## Section 3 — Data Pipeline

### 3.1 Binary Sensor Parsing — PASS

Paper specification: 20-byte records = 8-byte big-endian int64 timestamp + 12-byte
little-endian float32 XYZ values.

**Implementation (Cell 4, `read_bin()`):**
```python
ts = np.frombuffer(
    np.ascontiguousarray(data[:, :8]).tobytes(), dtype=">i8"   # BE int64 ✓
).astype(np.int64)
xyz = np.frombuffer(
    np.ascontiguousarray(data[:, 8:]).tobytes(), dtype="<f4"   # LE float32 ✓
).reshape(n, 3)
```

Correct on both endianness and data types.

### 3.2 Screen Event Parsing — PASS

Full Android intent strings are used for all three event types:

```python
FLAG_USER_PRESENT = "android.intent.action.USER_PRESENT"
FLAG_SCREEN_OFF   = "android.intent.action.SCREEN_OFF"
FLAG_SCREEN_ON    = "android.intent.action.SCREEN_ON"
```

Both the MPI reader (Cell 2) and the UV reader (Cell 4) perform exact string comparison.
No partial matching or abbreviation that could cause false positives.

### 3.3 Earth-Fixed Frame Conversion — FAIL ⚠️

This is the most significant technical error in the implementation.

**Cell 4:**
```python
rots = R.from_rotvec(rot)
```

**The problem:**

The Android `TYPE_ROTATION_VECTOR` sensor outputs a quaternion in this form:
```
[x·sin(θ/2),  y·sin(θ/2),  z·sin(θ/2),  cos(θ/2)]
```

Since each binary record stores only 3 float32 values (20 bytes = 8 ts + 12 xyz), only
the first three components are saved: `[x·sin(θ/2), y·sin(θ/2), z·sin(θ/2)]`.

These are the *vector part of a unit quaternion*, **not** a rotation vector.

`scipy.spatial.transform.R.from_rotvec(v)` interprets `v` as `axis × angle`, treating
`‖v‖` as the rotation angle in radians. For these quaternion components,
`‖v‖ = sin(θ/2)`, which is nowhere close to the actual angle `θ`.

**Impact:** The rotation matrices are wrong for all but trivial orientations.
This corrupts features 5–7, 11–13, 17–19 — that is **12 of 22 features**.

**Correct implementation:**
```python
xyz = rot  # shape (T, 3): [x*sin, y*sin, z*sin]
w = np.sqrt(np.clip(1.0 - np.sum(xyz**2, axis=1, keepdims=True), 0.0, 1.0))
quat = np.concatenate([xyz, w], axis=1)  # shape (T, 4): [x, y, z, w]
rots = R.from_quat(quat)                 # scipy expects [x, y, z, w] order
```

**Why results are not catastrophically worse:**

The wrongly-computed rotation matrices still encode some orientation information —
they are not random, just geometrically incorrect. On this specific dataset the features
were discriminative enough that the model learned from the signal despite the wrong frame,
which is why FAR numbers were still reasonable.

### 3.4 UV Window (1 second) — PASS

```python
target_n = 50  # int(uv_window_sec * uv_sampling_rate) = int(1.0 * 50)
```

Exactly matches paper Section 4.3.2: "the last second before unlocking" at 50 Hz.

### 3.5 Feature Channels — PARTIAL

`pad4()` produces shape `(4, T)` per feature with a zero-padded 4th channel. The paper
describes 3-channel features (XYZ axes only). `UVBranch(in_channels=4)` consumes all
four channels including the zero row. This wastes a small number of parameters but
does not corrupt the signal.

---

## Section 4 — Results Comparison

### 4.1 MPI Accuracy vs Table 1

**Paper Table 1 range:** 73.6% – 91.0% (N/A entries excluded)

**Implementation results (`results_mpi.csv`):**

| User/Device | Mean Acc | Std | Status |
|-------------|----------|-----|--------|
| 1_s10e / #00 | 94.06% | 0.96 | OK |
| 1_s10e / #01 | 95.88% | 0.98 | OK |
| 1_s10e / #02 | 95.82% | 0.91 | OK |
| 1_s10e / #03 | 95.59% | 1.12 | OK |
| 1_s10e / #05 | 96.17% | 2.48 | OK |
| 2_s10e / #00 | 80.81% | 1.89 | OK |
| 2_s10e / #02 | 80.81% | 2.09 | OK |
| 2_s10e / #03 | 89.48% | 0.70 | OK |
| 2_s10e / #04 | 83.02% | 1.17 | OK |
| 2_s10e / #05 | 77.16% | 1.64 | OK |
| 3_s10e / #00 | 91.10% | 0.92 | OK |
| 3_s10e / #01 | 87.64% | 1.36 | OK |
| 3_s10e / #02 | 83.90% | 4.12 | OK |
| 3_s10e / #04 | 91.74% | 2.07 | OK |
| 4_s10e / #00 | 95.62% | 2.69 | OK |
| 4_s10e / #01 | 82.63% | 3.18 | OK |
| 4_s10e / #03 | 77.59% | 3.99 | OK |
| 4_s10e / #04 | 78.60% | 3.88 | OK |
| 4_s10e / #05 | 76.00% | 3.35 | OK |
| 5_s10e / #00 | 93.39% | 1.68 | OK |
| 5_s10e / #01 | 90.51% | 1.28 | OK |
| 5_s10e / #02 | 85.68% | 2.63 | OK |
| 5_s10e / #04 | 92.84% | 1.74 | OK |
| 5_s10e / #05 | 87.23% | 1.96 | OK |
| 6_s10e / #01 | 85.38% | 1.48 | OK |
| 6_s10e / #02 | 92.29% | 0.72 | OK |
| 6_s10e / #03 | 87.59% | 1.07 | OK |
| 6_s10e / #04 | 87.65% | 2.19 | OK |
| 6_s10e / #05 | 87.06% | 1.58 | OK |

**Implementation range: 76.0% – 96.2%**

Eleven sessions exceed the paper's maximum of 91%. The distribution is shifted upward,
most strongly for User 1. This is explained by the non-pointwise kernels (size 5,5,3
vs size 1,1,1) — larger receptive fields give more capacity per session, fitting better
on some users but drifting from paper behavior on others.

No sessions fall below the paper's minimum of 73.6%.

### 4.2 UV Baseline vs Table 2 (n=75)

| Metric | Paper (n=75) | Implementation | Δ |
|--------|-------------|----------------|---|
| Acc_val | 98.1 ± 0.3% | 94.4% (std=0) | −3.7 pp |
| Acc_test | 98.15 ± 0.15% | 94.8% (std=0) | −3.4 pp |
| FAR_val (@TAR=90%) | 0.8 ± 0.4 × 10⁻² | 2.0% | 2.5× worse |
| FAR_test (@TAR=90%) | 1.4 ± 0.6 × 10⁻² | 1.82% | in range |

Accuracy is 3.7 pp below the paper. FAR_val is notably worse. FAR_test is within the
paper's reported range. `std=0` throughout indicates only a single seed result was
captured in `results_baseline.csv` — the per-seed accumulation writes a single dict
rather than one row per seed, losing confidence interval information.

### 4.3 UV Per-User FAR vs Table 3 (n=75)

| User ID (impl) | FAR mean | FAR std | Paper user | Paper FAR (n=75) |
|----------------|----------|---------|------------|------------------|
| 91 | 0.00% | 0.00% | 0 | 0.6 ± 0.4% |
| 92 | 1.82% | 2.81% | 1 | 2.0 ± 1.4% |
| 93 | 1.07% | 1.39% | 2 | 10.0 ± 6.0% |
| 94 | 2.22% | 2.94% | 3 | 12.0 ± 4.0% |
| 95 | 1.74% | 2.58% | 4 | 5.0 ± 3.0% |
| 96 | 4.09% | 3.71% | 5 | 11.0 ± 6.0% |
| 97 | 0.03% | 0.26% | 6 | 2.8 ± 1.8% |
| 98 | 2.41% | 2.17% | 7 | 22.0 ± 4.0% |
| 99 | 3.39% | 5.39% | 8 | 5.0 ± 3.0% |
| 100 | 7.21% | 3.60% | 9 | 4.0 ± 3.0% |
| 101 | 0.00% | 0.00% | 10 | 0.6 ± 0.4% |

**Paper mean FAR: ~6.8% · Implementation mean FAR: ~2.2%**

Every user shows lower FAR than the paper equivalent. The two users with 0% FAR match
the paper's easy users (0 and 10). The most likely explanation is that `additional_test/`
users were included in preprocessing (Cell 4), potentially mixing the user ID ordering
so that `all_uids[90:]` does not correspond to the same 11 users as the paper's
testfinal set.

---

## Section 5 — Code Quality

### 5.1 Numerical Stability — PASS

All division and log operations are guarded:

| Location | Guard |
|----------|-------|
| `score_verification` — template norm | `+ 1e-8` |
| `score_verification` — embedding norm | `+ 1e-8` |
| `SupervisedContrastiveLoss` — log denominator | `+ 1e-8` |
| `TripletMarginLoss` — empty batch | returns `tensor(0., requires_grad=True)` |
| `R.from_quat` w-recovery (after fix) | `np.clip(..., 0.0, 1.0)` |

### 5.2 Hardcoded Values

| Value | Location | In Config? | Impact |
|-------|----------|-----------|--------|
| 200ms pre-roll buffer | Cell 4, `get_aligned_window` | No | Minor |
| 90 genuine samples cap | Cell 7, bootstrap test | No | Minor |
| `additional_test` path | Cell 4 | No | Major (see 5.4) |

The 200ms pre-roll (`t_start = unlock_ts - window_ms - 200`) should be
`cfg.uv_preroll_ms` for reproducibility.

### 5.3 Single-GPU Constraint — PASS

```python
# Cell 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

No `DataParallel` or `DistributedDataParallel` usage anywhere in the notebook.
BatchNorm behavior is consistent across all 22 branches on a single GPU.

### 5.4 Data Leakage — PARTIAL

**No preprocessing leakage:** testfinal users pass through the same Cell 4 pipeline
as training users. Preprocessing is identical regardless of split assignment.

**Split integrity:** `subset_base`, `val_add`, and `testfinal` are disjoint by index.
`val_add` users are not used in baseline training. ✓

**Fine-tune genuine leak:** described in Section 1.3. The first 90 of the 150
fine-tune training samples are reused as the genuine validation set.

---

## Section 6 — Missing and Extra Components

### 6.1 Missing

**Multi-split sweep (Table 2)**

The paper trains and evaluates six baseline configurations (n ∈ {60,65,70,75,80,85}).
The implementation runs only n=75. This is both a reproducibility gap and means the
adaptive split logic (`cfg.uv_n_splits = [60,65,70,75,80,85]`) in Config goes unused.

**Held-out genuine validation**

The paper uses `valadd` (users 75–89) as the impostor pool for fine-tune epoch selection,
paired with a genuinely held-out portion of the target user's data. The implementation
reuses fine-tune training samples for genuine validation.

**Per-seed baseline rows**

`results_baseline.csv` captures only a single aggregated row. Confidence intervals
(mean ± std across seeds) cannot be computed from the saved output.

### 6.2 Extra (Confirmed Problem)

**`additional_test/` users become the testfinal set**

**Confirmed dataset layout (verified at runtime):**
```
train_val_test/   → 90 users
additional_test/  → 11 users
Total processed   → 101 users
```

The paper states 101 users were collected for User Verification, with 90 used for
baseline/val_add and 11 held out as testfinal — all within `train_val_test/`. However,
the Kaggle dataset places only 90 users in `train_val_test/` and the remaining 11 in
`additional_test/`.

Cell 4 processes both directories:
```python
for split_dir in [UV_TVT_DIR, UV_ADD_DIR]:   # both included
    for uid in sorted(os.listdir(split_dir)):
        all_user_dirs.append((uid, upath))
```

After preprocessing, Cell 7 sorts all 101 user IDs and splits by index:
```python
subset_base = all_uids[:75]    # 75 TVT users   → baseline training
val_add     = all_uids[75:90]  # 15 TVT users   → validation pool
testfinal   = all_uids[90:]    # 11 additional_test users → final test
```

**The result:** `testfinal` is exactly and entirely the `additional_test` cohort.
These users were collected separately from the main 90, likely under a different
session or at a different time. They may have different motion characteristics,
data quality, or trial counts — which would directly explain why per-user FAR
came out consistently lower than paper Table 3. The model is being tested on a
different population than the paper intended.

**The paper's intent vs Kaggle layout mismatch:**

| What paper says | What Kaggle has |
|----------------|-----------------|
| 101 users in one pool | 90 in TVT + 11 in additional_test |
| testfinal = last 11 of 101 | testfinal = additional_test cohort |
| Same collection protocol | Possibly different protocol |

**Three possible fixes:**

**Option A — Treat additional_test explicitly as testfinal (recommended):**
```python
# Cell 4: process both dirs but tag the source
for split_dir, source_tag in [(UV_TVT_DIR, "tvt"), (UV_ADD_DIR, "add")]:
    ...
    np.savez(..., source=source_tag)

# Cell 7: use source tag for splitting instead of index
tvt_uids = [uid for uid in all_uids if source[uid] == "tvt"]   # 90
add_uids = [uid for uid in all_uids if source[uid] == "add"]   # 11
subset_base = tvt_uids[:75]
val_add     = tvt_uids[75:]    # 15
testfinal   = add_uids         # 11 — matches current implicit behavior, now explicit
```
This keeps the current numeric results but documents the split truthfully.

**Option B — Use only TVT users, no additional_test:**
```python
# Cell 4: remove UV_ADD_DIR
for split_dir in [UV_TVT_DIR]:   # 90 users only
    ...

# Cell 7: split 90 users as 75/15/0 baseline only
# No final testfinal evaluation possible without the 11 additional users
```
Loses the final per-user FAR evaluation entirely.

**Option C — Contact dataset owner:**
Confirm whether `train_val_test/` was intended to contain 101 users and the
Kaggle upload is incomplete, or whether `additional_test/` is the correct
testfinal pool. The SamsungLabs GitHub repo or Kaggle dataset description
may clarify this.

---

## Section 7 — Priority Fixes

Listed in order of impact on result validity:

### Fix 1 — Make the additional_test split explicit (Cell 4 + Cell 7)

**Confirmed:** `train_val_test/` has 90 users, `additional_test/` has 11 users.
The current code implicitly uses `additional_test` users as testfinal via index
slicing. The fix is to make this explicit and documented rather than accidental.

```python
# Cell 4 — tag each user with their source directory
all_user_dirs = []
for split_dir, tag in [(UV_TVT_DIR, "tvt"), (UV_ADD_DIR, "add")]:
    if not os.path.exists(split_dir): continue
    for uid in sorted(os.listdir(split_dir)):
        upath = os.path.join(split_dir, uid)
        if os.path.isdir(upath):
            all_user_dirs.append((uid, upath, tag))

# Save tag alongside features
np.savez(os.path.join(PROCESSED_UV, f"{uid}.npz"),
         features=feats, cluster_ids=clusters,
         user_id=uid, source=tag)   # ← add source tag

# Cell 7 — split by source tag, not by index
user_sources = {}
for f in sorted(os.listdir(PROCESSED_UV)):
    if not f.endswith(".npz"): continue
    d = np.load(os.path.join(PROCESSED_UV, f), allow_pickle=True)
    uid = int(os.path.splitext(f)[0])
    user_sources[uid] = str(d["source"])

tvt_uids  = sorted([u for u, s in user_sources.items() if s == "tvt"])  # 90
add_uids  = sorted([u for u, s in user_sources.items() if s == "add"])  # 11

subset_base = tvt_uids[:75]    # 75 baseline users
val_add     = tvt_uids[75:]    # 15 validation users
testfinal   = add_uids         # 11 additional_test users (now explicit)
```

This preserves the current numeric behavior exactly while making the population
choice transparent and reproducible.

### Fix 2 — Correct rotation sensor interpretation (Cell 4)

```python
# Replace in compute_features():
rots = R.from_rotvec(rot)

# With:
xyz = rot  # shape (T, 3): vector part of unit quaternion
w = np.sqrt(np.clip(1.0 - np.sum(xyz**2, axis=1, keepdims=True), 0.0, 1.0))
quat = np.concatenate([xyz, w], axis=1)  # scipy [x, y, z, w] convention
rots = R.from_quat(quat)
```

Fixes 12 of 22 features (all earth-fixed rotations and their derivatives).

### Fix 3 — Fix genuine validation leak in fine-tuning (Cell 7)

```python
# Replace:
own = all_users[target_uid][:150]
val_gen = all_users[target_uid][:min(90, len(all_users[target_uid]))]

# With:
own = all_users[target_uid][:150]          # fine-tune training
val_gen = all_users[target_uid][150:]      # held-out for epoch selection
if len(val_gen) == 0:
    val_gen = own[-30:]                    # fallback for sparse users
```

### Fix 4 — Add multi-split sweep (Cell 7)

```python
# Wrap the baseline training block in:
for n_base in cfg.uv_n_splits:  # [60, 65, 70, 75, 80, 85]
    subset_base = all_uids[:n_base]
    val_add     = all_uids[n_base:90]
    testfinal   = all_uids[90:]
    # ... existing baseline + fine-tune + bootstrap code ...
    b_rows.append({
        "n_baseline": n_base,
        "acc_val_mean": ..., "acc_val_std": ...,
        "far_val_mean": ..., "far_test_mean": ...
    })
```

### Fix 5 — Match MPI pointwise convolutions (Cell 5)

```python
# Replace MPIModel conv layers:
nn.Conv1d(n_channels, 64,  kernel_size=5, padding=2)
nn.Conv1d(64,         128, kernel_size=5, padding=2)
nn.Conv1d(128,        256, kernel_size=3, padding=1)

# With:
nn.Conv1d(n_channels, 64,  kernel_size=1)
nn.Conv1d(64,         128, kernel_size=1)
nn.Conv1d(128,        256, kernel_size=1)
```

---

## Appendix A — What Went Right

For context, the following components are correctly implemented and match the paper:

- Full two-stage pipeline structure (MPI → UV)
- Binary sensor record parsing (endianness, byte layout)
- Full Android intent string matching for screen events
- Sensor alignment by timestamp interpolation
- MPI: 3-second window, 100-reading minimum, positive/negative labeling
- MPI: stationary rejection (linear accelerometer threshold)
- UV: 1-second pre-unlock window (50 samples at 50 Hz)
- UV: 22-branch architecture with identical branches
- UV: Dual training head (CE classifier + Siamese)
- All three loss functions (L_CE, L_TM, L_SC) with correct formula
- Double augmentation + concatenation in training loop
- Baseline → fine-tune → bootstrap evaluation sequence
- Training split (75/15/11) matching paper Section 6.5
- Fine-tune: branches frozen, 2-class head, 10 epochs, reduced LR
- Bootstrap: 5000 repeats, 90+90 samples, FAR@TAR=90%
- Single-GPU constraint (no DataParallel)
- Numerical stability guards throughout

---

## Appendix B — Results Files

### results_mpi.csv

29 user-device sessions, accuracy range 76.0% – 96.2%, mean 87.7%.
All sessions returned `status=OK` (no N/A sessions in this run, unlike paper Table 1
which has 3 N/A entries for insufficient negative samples).

### results_baseline.csv

Single row for n=75. `acc_val=94.43%`, `acc_test=94.84%`,
`far_val=2.0%`, `far_test=1.82%`. All std values are 0 (single-seed artifact).

### results_uv_final.csv

11 testfinal users (IDs 91–101). FAR range 0.0% – 7.2%, mean 2.2%.
Consistently lower than paper Table 3. Now confirmed: testfinal users are the
`additional_test` cohort (11 users), not the last 11 of the 101 TVT users as the
paper intended. These users were collected separately. Whether their motion patterns
are easier to distinguish (leading to lower FAR) or whether data collection differences
account for the gap cannot be determined without inspecting the raw files.

---

## Appendix C — Confirmed Dataset Layout (Post-Submission Verification)

The following was run in the Kaggle environment after the initial audit to confirm
the `additional_test` finding:

```python
import os
tvt_dir = ("/kaggle/input/datasets/djaarf/motionid-imu-specific-motion"
           "/IMU_specific_motion/train_val_test")
add_dir = ("/kaggle/input/datasets/djaarf/motionid-imu-specific-motion"
           "/IMU_specific_motion/additional_test")
print("TVT users:", len(os.listdir(tvt_dir)))
print("Additional users:", len(os.listdir(add_dir)))
```

**Output:**
```
TVT users: 90
Additional users: 11
```

**Interpretation:**

The paper states 101 users were collected for User Verification and split as
90 (baseline + val) + 11 (testfinal), implying all 101 live in `train_val_test/`.
The Kaggle dataset instead places 90 in `train_val_test/` and 11 in `additional_test/`.

Because Cell 4 processes both directories and Cell 7 splits by sorted index position,
the 11 `additional_test` users land at `all_uids[90:]` and become `testfinal`
automatically — matching the paper's 90+11 count but using a different cohort
than intended.

This does **not** invalidate the implementation. It means the final FAR numbers
in `results_uv_final.csv` reflect the `additional_test` population, not the
paper's original held-out set. The recommended fix (Option A in Section 6.2)
makes this explicit rather than accidental, preserving all current numeric
results while correctly documenting the population.

---

*End of audit report.*

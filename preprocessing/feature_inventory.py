"""
22 UV feature vectors. Section 6.1 of arXiv:2302.01751.

IMPORTANT CORRECTIONS vs naive implementation:
- Rotation sensor stores 3-axis rotation VECTOR (azimuth, pitch, roll)
  NOT a quaternion. File format is 3 floats = same 20-byte record as all other sensors.
- Earth-fixed frame conversion uses scipy R.from_rotvec(), not R.from_quat().
- ALL 22 features are 3-channel. All padded to (4, T) for uniform branch input.
- Total channels: 22 * 3 = 66 (not 67).

Run: python preprocessing/feature_inventory.py
Expected:
  Feature inventory: 22 features
  All 22 are 3-channel
  Total channels (pre-pad): 66
  Model input shape per sample: (22, 4, T)
"""

from collections import namedtuple

FeatureDef = namedtuple(
    "FeatureDef",
    ["name", "source_sensor", "transform", "description"])

# ALL features are 3-channel (rotation is azimuth/pitch/roll, not quaternion)
FEATURE_LIST = [
    # ── Raw readings, device frame ──────────────────────────────────────
    FeatureDef("acc_raw",           "acc",   "none",
               "Raw accelerometer, device frame"),
    FeatureDef("gyro_raw",          "gyro",  "none",
               "Raw gyroscope, device frame"),
    FeatureDef("mag_raw",           "mag",   "none",
               "Raw magnetometer, device frame"),
    # ── Manually computed lin_acc (paper Section 6.1 formula) ───────────
    FeatureDef("lin_acc_manual",    "acc",   "minus_grav",
               "lin_acc = acc - grav  (device frame)"),
    # ── Earth-fixed frame via R.from_rotvec(rot) ─────────────────────────
    FeatureDef("acc_rot",           "acc",   "earth",
               "Accelerometer in Earth-fixed frame"),
    FeatureDef("gyro_rot",          "gyro",  "earth",
               "Gyroscope in Earth-fixed frame"),
    FeatureDef("mag_rot",           "mag",   "earth",
               "Magnetometer in Earth-fixed frame"),
    # ── Differences, unrotated ─────────────────────────────────────────
    FeatureDef("acc_delta",         "acc",   "diff",
               "Consecutive differences of raw acc"),
    FeatureDef("gyro_delta",        "gyro",  "diff",
               "Consecutive differences of raw gyro"),
    FeatureDef("mag_delta",         "mag",   "diff",
               "Consecutive differences of raw mag"),
    # ── Differences, rotated ───────────────────────────────────────────
    FeatureDef("acc_rot_delta",     "acc",   "earth+diff",
               "Consecutive differences of rotated acc"),
    FeatureDef("gyro_rot_delta",    "gyro",  "earth+diff",
               "Consecutive differences of rotated gyro"),
    FeatureDef("mag_rot_delta",     "mag",   "earth+diff",
               "Consecutive differences of rotated mag"),
    # ── Integrals, unrotated ───────────────────────────────────────────
    FeatureDef("acc_integral",      "acc",   "cumsum",
               "Cumulative integral of raw acc"),
    FeatureDef("gyro_integral",     "gyro",  "cumsum",
               "Cumulative integral of raw gyro"),
    FeatureDef("mag_integral",      "mag",   "cumsum",
               "Cumulative integral of raw mag"),
    # ── Integrals, rotated ─────────────────────────────────────────────
    FeatureDef("acc_rot_integral",  "acc",   "earth+cumsum",
               "Cumulative integral of rotated acc"),
    FeatureDef("gyro_rot_integral", "gyro",  "earth+cumsum",
               "Cumulative integral of rotated gyro"),
    FeatureDef("mag_rot_integral",  "mag",   "earth+cumsum",
               "Cumulative integral of rotated mag"),
    # ── lin_acc delta + integral ──────────────────────────────────────
    FeatureDef("lin_acc_delta",     "acc",   "minus_grav+diff",
               "Consecutive differences of lin_acc"),
    FeatureDef("lin_acc_integral",  "acc",   "minus_grav+cumsum",
               "Cumulative integral of lin_acc"),
    # ── Raw rotation vector (azimuth, pitch, roll) — 3-axis ───────────
    FeatureDef("rot_raw",           "rot",   "none",
               "Raw rotation vector (azimuth, pitch, roll)"),
]

N_FEATURES    = 22
N_CHANNELS    = 3   # all features are 3-channel
MODEL_INPUT_C = 4   # padded to 4 for uniform branch input

assert len(FEATURE_LIST) == N_FEATURES


def get_feature_shapes(T: int = 50):
    """Returns list of (MODEL_INPUT_C, T) for each of the 22 features."""
    return [(MODEL_INPUT_C, T)] * N_FEATURES


def verify():
    assert len(FEATURE_LIST) == 22
    print(f"Feature inventory: {len(FEATURE_LIST)} features")
    print(f"All 22 are {N_CHANNELS}-channel")
    print(f"Total channels (pre-pad): {N_FEATURES * N_CHANNELS}")
    print(f"Model input shape per sample: (22, {MODEL_INPUT_C}, T)")
    print()
    print(f"{'Idx':>3}  {'Name':<22} {'Source':<6} {'Transform':<18}")
    print("-" * 56)
    for i, f in enumerate(FEATURE_LIST):
        print(f"  {i+1:2d}. {f.name:<22} {f.source_sensor:<6} {f.transform}")


if __name__ == "__main__":
    verify()

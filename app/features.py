"""Shared schema and deterministic engineering; no learned state or target columns."""
import numpy as np
import pandas as pd

FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
NUMERIC = FEATURES[1:] + ["Temperature difference [K]", "Power [W]", "Wear torque [minNm]"]

def validate_dataset(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    target = "Target" if "Target" in frame.columns else "Machine failure"
    if target not in frame:
        raise ValueError("Expected Target or Machine failure column")
    if frame[FEATURES + [target]].isna().any().any():
        raise ValueError("Missing feature or target values")
    if not set(frame["Type"]).issubset({"L", "M", "H"}):
        raise ValueError("Type must be L, M or H")
    numbers = frame[FEATURES[1:]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError("Nonfinite measurements")
    if (numbers[:, :4] <= 0).any() or (numbers[:, 4] < 0).any():
        raise ValueError("Temperatures/speed/torque must be positive; wear nonnegative")
    if not set(frame[target].unique()).issubset({0, 1}) or frame[target].nunique() != 2:
        raise ValueError("Expected binary target containing both classes")
    # An allowlist prevents IDs, Failure Type, TWF/HDF/PWF/OSF/RNF leaking labels.
    return frame[FEATURES].copy(), frame[target].astype(int).copy()

def engineer_features(frame):
    out = frame[FEATURES].copy()
    out["Temperature difference [K]"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["Power [W]"] = out["Torque [Nm]"] * out["Rotational speed [rpm]"] * (2 * np.pi / 60)
    out["Wear torque [minNm]"] = out["Tool wear [min]"] * out["Torque [Nm]"]
    return out

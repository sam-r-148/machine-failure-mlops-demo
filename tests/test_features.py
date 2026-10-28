import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from app.features import FEATURES, engineer_features, validate_dataset
from app.main import Measurements, to_frame

SAMPLE = {"product_type": "M", "air_temperature_k": 298.1,
          "process_temperature_k": 308.6, "rotational_speed_rpm": 1551,
          "torque_nm": 42.8, "tool_wear_min": 0}

def test_input_mapping_and_physical_features():
    frame = to_frame(Measurements(**SAMPLE))
    assert list(frame.columns) == FEATURES
    engineered = engineer_features(frame)
    assert engineered["Temperature difference [K]"].iloc[0] == pytest.approx(10.5)
    assert engineered["Power [W]"].iloc[0] == pytest.approx(42.8 * 1551 * 2 * np.pi / 60)
    assert engineered["Wear torque [minNm]"].iloc[0] == 0

@pytest.mark.parametrize("target", ["Target", "Machine failure"])
def test_dataset_allowlist_prevents_target_leakage(target):
    row = to_frame(Measurements(**SAMPLE))
    frame = pd.concat([row, row], ignore_index=True)
    frame[target] = [0, 1]
    frame["Failure Type"] = ["No Failure", "Power Failure"]
    frame["TWF"] = [0, 1]
    frame["UDI"] = [1, 2]
    X, y = validate_dataset(frame)
    assert list(X.columns) == FEATURES
    assert list(y) == [0, 1]

@pytest.mark.parametrize("patch", [
    {"product_type": "X"}, {"air_temperature_k": 0}, {"torque_nm": -1},
    {"tool_wear_min": -1}, {"rotational_speed_rpm": 1.5},
    {"air_temperature_k": float("nan")}, {"unexpected": 2},
])
def test_bad_input(patch):
    with pytest.raises(ValidationError):
        Measurements(**{**SAMPLE, **patch})

def test_missing_input():
    with pytest.raises(ValidationError):
        Measurements()

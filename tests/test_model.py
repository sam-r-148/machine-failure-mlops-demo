"""Recompute held-out results from the serialized model, without HTTP or training."""
import hashlib
import json

import joblib
import pandas as pd
import pytest
import sklearn
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.features import validate_dataset
from app.main import ROOT, Measurements, app, predict
from tests.test_features import SAMPLE


@pytest.fixture(scope="module")
def saved():
    meta = json.loads((ROOT / "artifacts/metadata.json").read_text())
    model = joblib.load(ROOT / "artifacts/model.joblib")
    data = ROOT / "data/predictive_maintenance.csv"
    assert hashlib.sha256(data.read_bytes()).hexdigest() == meta["dataset"]["sha256"]
    X, y = validate_dataset(pd.read_csv(data))
    _, test_X, _, test_y = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    return model, meta, test_X, test_y


def test_saved_model_reproduces_held_out_metrics(saved):
    model, meta, X, y = saved
    p = model.predict_proba(X)[:, list(model.classes_).index(1)]
    labels = p >= meta["decision_threshold"]
    measured = {
        "average_precision": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "precision": precision_score(y, labels, zero_division=0),
        "recall": recall_score(y, labels, zero_division=0),
    }
    assert meta["sklearn_version"] == sklearn.__version__
    for name, value in measured.items():
        assert value == pytest.approx(meta["metrics"][name], abs=1e-12)
    assert measured["average_precision"] > y.mean()
    assert measured["recall"] >= .4


def test_inference_function_uses_saved_threshold(saved, monkeypatch):
    model, meta, _, _ = saved
    monkeypatch.setattr(app.state, "model", model, raising=False)
    monkeypatch.setattr(app.state, "metadata", meta, raising=False)
    monkeypatch.setattr(app.state, "predictions", 0, raising=False)
    monkeypatch.setattr(app.state, "flagged_predictions", 0, raising=False)
    result = predict(Measurements(**SAMPLE))
    assert 0 <= result["failure_probability"] <= 1
    assert result["machine_failure"] == (result["failure_probability"] >= meta["decision_threshold"])
    assert app.state.predictions == 1

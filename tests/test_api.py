import json
import hashlib
import joblib
import pytest
from fastapi.testclient import TestClient
from app.main import app, ROOT, to_frame, Measurements
from tests.test_features import SAMPLE

@pytest.fixture
def client():
    with TestClient(app) as instance:
        yield instance

def test_health_and_dataset(client):
    assert client.get("/health").json()["model_loaded"]
    info = client.get("/model-info").json()
    assert info["model_version"] == "maintenance-rf-v1"
    assert info["train_rows"] + info["test_rows"] == 10000
    assert info["dataset"]["dataset_slug"] == "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
    assert info["dataset"]["sha256"] == hashlib.sha256((ROOT / "data/predictive_maintenance.csv").read_bytes()).hexdigest()

def test_served_probability_matches_saved_model(client):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    result = response.json()
    model = joblib.load(ROOT / "artifacts/model.joblib")
    p = float(model.predict_proba(to_frame(Measurements(**SAMPLE)))[0, list(model.classes_).index(1)])
    assert result["failure_probability"] == pytest.approx(p)
    threshold = json.loads((ROOT / "artifacts/metadata.json").read_text())["decision_threshold"]
    assert result["machine_failure"] == (p >= threshold)
    assert client.get("/metrics").json()["predictions"] == 1

def test_http_validation(client):
    assert client.post("/predict", json={**SAMPLE, "torque_nm": -1}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422

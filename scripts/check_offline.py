"""Socket-free checks of the actual Kaggle-trained model; no training or fallback."""
import json
import joblib
from app.main import ROOT, app, Measurements, predict
from tests.test_features import SAMPLE

meta = json.loads((ROOT / "artifacts/metadata.json").read_text())
assert meta["model_version"] == "maintenance-rf-v1"
assert meta["dataset"]["assignment_dataset"]
app.state.model = joblib.load(ROOT / "artifacts/model.joblib")
app.state.metadata = meta
app.state.predictions = 0
app.state.flagged_predictions = 0
result = predict(Measurements(**SAMPLE))
assert 0 <= result["failure_probability"] <= 1
assert result["machine_failure"] == (result["failure_probability"] >= result["decision_threshold"])
print(json.dumps({"offline_prediction_check": "PASS", "result": result}, indent=2))

"""Verify the committed Kaggle snapshot before notebook training."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from app.features import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data/predictive_maintenance.csv"
provenance = json.loads((ROOT / "data/provenance.json").read_text())
if provenance["dataset_slug"] != "stephanmatzka/predictive-maintenance-dataset-ai4i-2020" or provenance["source"] != "Kaggle":
    raise ValueError("Wrong data source")
if not provenance["assignment_dataset"]:
    raise ValueError("Assignment requires actual Kaggle data")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != provenance["sha256"]:
    raise ValueError("Dataset checksum mismatch")
X, y = validate_dataset(pd.read_csv(path))
if len(X) != 10000:
    raise ValueError("Expected 10000 Kaggle records")
print(f"Verified Kaggle snapshot: {len(X)} rows, {int(y.sum())} failures; SHA-256 {digest}")

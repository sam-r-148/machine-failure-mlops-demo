"""Download and validate only the actual Kaggle predictive-maintenance archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from app.features import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
SLUG = "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
URL = "https://www.kaggle.com/api/v1/datasets/download/" + SLUG

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an existing Kaggle-downloaded ZIP")
    args = parser.parse_args()
    try:
        if args.archive:
            raw = args.archive.read_bytes()
        else:
            print("Downloading Kaggle dataset: " + SLUG, flush=True)
            request = urllib.request.Request(URL, headers={"User-Agent": "maintenance-learning-project/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
        archive = zipfile.ZipFile(io.BytesIO(raw))
        candidates = []
        for name in archive.namelist():
            if name.lower().endswith(".csv"):
                content = archive.read(name)
                frame = pd.read_csv(io.BytesIO(content))
                try:
                    validate_dataset(frame)
                except ValueError:
                    continue
                candidates.append((name, content, frame))
        if len(candidates) != 1:
            raise ValueError(f"Expected one compatible CSV; found {len(candidates)}")
        name, content, frame = candidates[0]
        if len(frame) != 10000:
            raise ValueError(f"Expected AI4I dataset with 10000 records, got {len(frame)}")
        destination = ROOT / "data"
        destination.mkdir(exist_ok=True)
        (destination / "predictive_maintenance.csv").write_bytes(content)
        provenance = {
            "source": "Kaggle", "dataset_slug": SLUG,
            "page": "https://www.kaggle.com/datasets/" + SLUG,
            "download_url": URL, "archive_member": name,
            "download_method": "provided Kaggle ZIP" if args.archive else "Kaggle download API",
            "sha256": hashlib.sha256(content).hexdigest(), "rows": len(frame),
            "assignment_dataset": True
        }
        (destination / "provenance.json").write_text(json.dumps(provenance, indent=2))
        print(f"Validated and saved {len(frame)} rows to data/predictive_maintenance.csv")
    except (urllib.error.URLError, zipfile.BadZipFile, ValueError, OSError) as error:
        raise SystemExit(f"Kaggle download failed: {error}\nNo substitute dataset was created. Retry on a network-enabled laptop or use --archive PATH_TO_KAGGLE_ZIP.")

if __name__ == "__main__":
    main()

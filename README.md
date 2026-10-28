# Industrial Machine Failure Detection â€” Python, Jupyter, FastAPI and Docker

This project replaces Iris with Kaggle's **AI4I 2020 Predictive Maintenance** dataset.
It uses operating measurements to classify a machine's labeled failure state.

Dataset: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
Original work: S. Matzka (2020), AI4I 2020 Predictive Maintenance Dataset, UCI, DOI https://doi.org/10.24432/C5HS5C (CC BY 4.0).

## Current status (28 October 2026)

- Kaggle dataset snapshot validated: 10,000 rows, 339 failures; provenance SHA-256 matches.
- Trained maintenance model and evaluation artifacts are present: 8,000 training rows and 2,000 holdout rows. Saved metrics: average precision 0.8804, ROC-AUC 0.9789, precision 0.8906, recall 0.8382 and F2 0.8482.
- Verified saved test report: all 16 tests passed, including API, feature and saved-model tests; zero failures, errors or skips.
- Fixed pytest collection of scripts/smoke_test.py: HTTP requests now run only when invoked as a script.
- Verified saved notebook: execution backend is Jupyter kernel via nbclient; all four code cells executed in order with no error outputs.
- Docker build was attempted from the user terminal but reported as failed; its final error has not yet been captured. Retrying from the assistant session is blocked by permission denied on the Docker Desktop Linux engine named pipe.
- User reports completing GitHub browser authentication. The assistant session could not verify the account: the latest API request returned HTTP 403 (rate limit exceeded). No Git remote is configured yet.
- No repository creation, Docker smoke-test success or successful GitHub Actions run is claimed.

## Why this case study

This is a practical industrial operations example with 10,000 synthetic records, mixed categorical/numeric features and rare failures. It demonstrates more than a basic flower classifier:
- exclude IDs and failure-type labels to prevent target leakage;
- add physical features: temperature difference, power and wear multiplied by torque;
- use class weighting for rare failures;
- select a decision threshold using training-only out-of-fold predictions;
- evaluate average precision (PR-AUC), ROC-AUC, precision, recall, F2 and a confusion matrix;
- serialize the full pipeline and expose validated inference through FastAPI.

This is current-state classification, not a forecast of when a future failure will occur. The random row split is an educational benchmark; realistic future-time or unseen-machine performance needs separate time/group holdouts.

## Assignment mapping

| Requirement | Implementation |
| --- | --- |
| Kaggle dataset | scripts/download_data.py downloads the author's Kaggle archive and validates 10,000 rows |
| Train/test in Jupyter or Colab | notebooks/01_train_maintenance.ipynb; use Jupyter Notebook, not VS Code |
| model.joblib and parameters | Notebook produces artifacts/model.joblib, parameters.json and metadata.json |
| FastAPI prediction in VS Code | app/main.py, POST /predict and interactive /docs |
| API through Docker | Dockerfile, compose.yaml, scripts/smoke_test.py |
| New GitHub repository | scripts/publish.ps1 creates a private machine-failure-mlops-demo repository |
| Actions CI pipeline | .github/workflows/ci.yml downloads Kaggle data, executes Jupyter, tests and builds/runs Docker |
| Successful execution | Must be verified from a real Actions run; publishing script waits for its result |

## 1. Install locally (PowerShell, Python 3.12)

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name python3 --display-name "Maintenance Python"
~~~

Use this environment consistently for notebook training, local serving and tests. The Docker image uses the same runtime requirements.

## 2. Download directly from Kaggle

~~~powershell
python scripts/download_data.py
~~~

The downloader finds the compatible CSV inside Kaggle's archive, checks the feature schema, both target classes and the expected row count, then saves:

- data/predictive_maintenance.csv
- data/provenance.json (Kaggle source, archive member and SHA-256)

It supports Kaggle's Target/Failure Type layout as well as the Machine failure layout. No generated data or alternate source is used.

If Kaggle requires an authenticated browser download, download the ZIP from the dataset page and use:

~~~powershell
python scripts/download_data.py --archive "C:\path\to\archive.zip"
~~~

The script does not need or store a Kaggle token for the public endpoint. An HTML login response, invalid ZIP or wrong dataset stops with an error instead of silently creating data.

## 3. Train in Jupyter Notebook

~~~powershell
jupyter notebook notebooks/01_train_maintenance.ipynb
~~~

Select Maintenance Python, then **Run All Cells**. Do this in Jupyter, not the VS Code notebook editor.

The notebook verifies the downloaded checksum, holds out a stratified 20% test partition, selects an F2 threshold using three-fold training-only predictions, fits the full pipeline, evaluates once on the holdout and saves the results. Its feature allowlist excludes UDI, Product ID, Target, Machine failure, Failure Type and every failure flag.

Generated outputs:

| File | Purpose |
| --- | --- |
| artifacts/model.joblib | Fitted feature engineering, encoding and random forest |
| artifacts/parameters.json | Hyperparameters, feature importances and decision threshold |
| artifacts/metadata.json | Data source/hash, versions, threshold and measured metrics |
| artifacts/classification_report.json | Per-class evaluation |
| artifacts/confusion_matrix.png | False-positive and missed-failure counts |

For automated notebook execution through a Jupyter kernel:

~~~powershell
python scripts/execute_notebook.py
~~~

The --offline-shell option is only a diagnostic cell runner. It still requires the actual Kaggle data, and does not replace the required Jupyter session.

## 4. Serve in VS Code

~~~powershell
code .
python -m pytest -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
~~~

Open http://localhost:8000/docs. Try POST /predict:

~~~json
{
  "product_type": "M",
  "air_temperature_k": 298.1,
  "process_temperature_k": 308.6,
  "rotational_speed_rpm": 1551,
  "torque_nm": 42.8,
  "tool_wear_min": 0
}
~~~

The response contains machine_failure, failure_probability, decision_threshold, recommendation and model_version. Predictions and scores must come from the trained artifact; no fixed example score is claimed. Random-forest scores are uncalibrated.

Allowed product types are L/M/H. Units are kelvin, rpm, Nm and minutes. Wear may be zero. Missing, nonfinite, negative and unknown fields are rejected. Broad upper limits catch obvious input errors; they do not establish in-distribution inputs.

| Endpoint | Purpose |
| --- | --- |
| GET /health | Model readiness |
| GET /model-info | Provenance, metrics and threshold |
| POST /predict | Failure classification and score |
| GET /metrics | Request, prediction, flagged-prediction and latency counters |
| GET /docs | Swagger UI |

Run a real HTTP smoke test in another terminal:

~~~powershell
python scripts/smoke_test.py
~~~

The API never trains at startup. It fails clearly if model.joblib is missing or metadata identifies a different model. Metrics are single-process counters that reset on restart; production drift detection is outside this assignment.

## 5. Execute endpoints through Docker

Start Docker Desktop's Linux engine. Stop local Uvicorn to release port 8000.

~~~powershell
docker info
docker compose up --build -d
docker compose ps
python scripts/smoke_test.py
docker compose logs api
~~~

Execute /predict through http://localhost:8000/docs while the container is running.
The container runs as a non-root user with a health check. The published port is limited to localhost.

~~~powershell
docker compose down
~~~

## 6â€“8. Create repo, execute CI and verify green result

~~~powershell
gh auth login -h github.com
gh auth status
powershell -ExecutionPolicy Bypass -File scripts/publish.ps1
~~~

The script checks the maintenance model, runs tests, commits files, creates a new private machine-failure-mlops-demo repository, pushes main and waits for CI. If that repository name already exists, pass -RepoName a unique name.

CI validates the committed Kaggle dataset snapshot and its provenance checksum, so it can reproduce training on a clean runner. It then executes the notebook through Jupyter, runs the API tests, builds Docker, calls the container's endpoints and uploads the notebook/model/evaluation/provenance evidence.

The final requirement is complete only when an actual Actions run is green. Save the successful run URL and a screenshot for your submission.

## Checks that work without a trained model

~~~powershell
python -m pytest tests/test_features.py -q
~~~

These verify input mapping, physical feature calculations, both Kaggle target schemas, exclusion of leakage columns, and invalid input rejection. They do not substitute for model evaluation, HTTP testing or Docker execution.

After training, a socket-free model check is also available:

~~~powershell
python -m scripts.check_offline
~~~

## Suggested demonstration sequence

1. Show the Kaggle dataset, local CSV and provenance hash.
2. Show the completed Jupyter notebook, class imbalance, threshold selection and final test metrics.
3. Show model.joblib, parameters.json and the successful reload check.
4. Show app/main.py in VS Code and execute Swagger predictions.
5. Show the running Docker container and passing HTTP smoke test.
6. Show the new GitHub repo and actual successful Actions run.

Only load trusted joblib files. Keep credentials out of the repository.

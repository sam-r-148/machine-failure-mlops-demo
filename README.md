# Machine Failure Prediction API

FastAPI serves a saved Random Forest pipeline trained on the Kaggle AI4I 2020 predictive maintenance dataset. Serving and CI do not download data or retrain the model.

## Published files

- `main.py`: API with health, prediction, model information, metrics and Swagger endpoints.
- `model.joblib`: fitted pipeline with its selected decision threshold and metadata embedded.
- `app/features.py` and `app/__init__.py`: preprocessing module required to reload the pipeline.
- `requirements.txt`: pinned runtime dependencies.
- `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`: container and CI configuration.
- `.gitignore`: allowlist that keeps training files local.

## Run locally

Use Python 3.12:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000/docs and try `POST /predict`:

```json
{
  "product_type": "M",
  "air_temperature_k": 298.1,
  "process_temperature_k": 308.6,
  "rotational_speed_rpm": 1551,
  "torque_nm": 42.8,
  "tool_wear_min": 0
}
```

The response includes the failure flag, probability, selected threshold and model version. Product type must be L, M or H; missing or invalid measurements return HTTP 422.

## Docker

```powershell
docker build -t maintenance-api .
docker run --rm -p 127.0.0.1:8000:8000 maintenance-api
```

CI builds this image from the committed model, starts it, checks readiness, exercises predictions and rejects invalid input. Check the repository's Actions tab for the actual run result.

## Local training files

The existing `data/`, `notebooks/`, `artifacts/`, `scripts/`, `tests/`, `requirements-dev.txt`, `compose.yaml` and original `app/main.py` remain on the training computer, excluded from new commits and Docker builds. A fresh clone only needs the published inference files. The root model is a deployment copy; the original `artifacts/model.joblib` is preserved. Retraining locally does not automatically replace the root deployment model; export it with matching metadata before publishing an updated model.

Dataset: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

Original dataset: S. Matzka (2020), AI4I 2020 Predictive Maintenance Dataset, UCI, https://doi.org/10.24432/C5HS5C (CC BY 4.0).

This is current-state classification on synthetic data. Random Forest probabilities are uncalibrated; the model does not forecast future failure times. Load only trusted joblib artifacts.

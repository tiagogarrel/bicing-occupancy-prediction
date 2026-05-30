# Deployment Architecture

This document describes how the trained model would be deployed as a production service.
Nothing here is implemented — the purpose is to show that the model was designed
with deployment in mind from the start.

---

## Overview

The deployment exposes a REST API that accepts a station ID, a timestamp and a
prediction horizon, fetches live weather data, and returns a predicted occupancy value.

```
Client
  │
  ▼
POST /predict
  │
  ├─► fetch weather forecast (Open-Meteo API)
  ├─► build feature vector
  └─► model.predict()
        │
        ▼
      { predicted_occupancy: 0.72, horizon_hours: 6 }
```

---

## API

### Endpoint

```
POST /predict
Content-Type: application/json
```

### Request body

```json
{
  "station_id": 42,
  "datetime":   "2024-11-15T10:00:00",
  "horizon_hours": 6
}
```

All three fields are required. `horizon_hours` must be one of: 1, 3, 6, 12, 24.

### Response

```json
{
  "station_id":            42,
  "datetime":              "2024-11-15T10:00:00",
  "horizon_hours":         6,
  "predicted_occupancy":   0.72,
  "target_datetime":       "2024-11-15T16:00:00"
}
```

`predicted_occupancy` is a float in [0, 1]: fraction of docks occupied.

### Implementation

FastAPI is the natural choice — minimal boilerplate, automatic request validation
via Pydantic, auto-generated OpenAPI docs at `/docs`.

```python
# deployment/app.py (not implemented)
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd
from src.data.features import build_feature_vector   # to be implemented

app   = FastAPI()
model = joblib.load("models/final_model.joblib")     # loaded once at startup

class PredictRequest(BaseModel):
    station_id:    int
    datetime:      str
    horizon_hours: int

@app.post("/predict")
def predict(req: PredictRequest):
    features = build_feature_vector(req.station_id, req.datetime, req.horizon_hours)
    occ      = float(model.predict(features)[0])
    return {"predicted_occupancy": round(occ, 4), ...}
```

---

## The weather problem

For a historical model trained on known weather, deployment requires a **weather
forecast**, not the current conditions.

- For horizons of 1–6 h: the Open-Meteo free forecast API returns hourly forecasts
  up to 7 days ahead at the resolution we need. The API call takes ~200 ms.
- For horizons of 12–24 h: forecast accuracy degrades but the features are the same.

The deployment service should cache weather forecasts and refresh them every hour
to avoid hammering the external API.

```
Request arrives
    │
    ├─ cache hit? ─► use cached forecast
    │
    └─ cache miss? ─► call Open-Meteo ─► cache result (TTL: 1h)
```

---

## Containerisation

The service should be packaged as a Docker image so it can run anywhere without
dependency conflicts.

```dockerfile
# Dockerfile (not implemented)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t bicing-api .
docker run -p 8000:8000 bicing-api
```

---

## Model loading

The model is loaded **once at application startup** using `joblib.load()`.
Reloading it on each request would add ~1–2 seconds of latency.

The saved artifact (`models/final_model.joblib`) contains the complete
fitted estimator. No separate scaler file is needed because Linear Regression
uses a `Pipeline` (scaler + model in one object), and tree-based models
(Random Forest, LightGBM) are scale-invariant.

---

## Retraining

The model should be retrained periodically as new data accumulates.

**Suggested cadence:** monthly, after the previous month's Open Data BCN
files are published (~2 weeks after the month ends).

**Retraining pipeline:**
1. Download and extract the new month's `.7z` file
2. Run `02_feature_engineering.ipynb` to regenerate the full dataset
3. Run the winning model notebook (e.g. `05_model_lightgbm.ipynb`) with
   an updated `SPLIT_DATE` that includes all but the last two months as train
4. Compare the new model's test metrics against the production model
5. If new model is better (lower RMSE), replace `models/final_model.joblib`
   and redeploy

**What to monitor in production:**
- Prediction error against actual occupancy (requires logging requests + outcomes)
- Occupancy distribution shift — if Barcelona's cycling patterns change
  (new infrastructure, major events), the model may degrade before the
  retraining cycle catches it

---

## Scaling considerations

At ~548 stations and a typical use case of one prediction per station per
minute, the load is around 32k requests/min. A single FastAPI instance with
a preloaded LightGBM model can handle this comfortably — LightGBM inference
on a single row takes < 1 ms.

If the service needed to scale horizontally, the model artifact could be
stored in object storage (e.g. S3) and pulled at container startup, with
multiple instances behind a load balancer.

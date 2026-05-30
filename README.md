# Bicing Barcelona — Occupancy Prediction

Predicts the occupancy of any Bicing station in Barcelona at a configurable time horizon (1, 3, 6, 12 or 24 hours ahead) using supervised regression on historical station data, weather, and temporal features.

## Problem definition

Each prediction answers: *given the current state of station S at time T, what fraction of its docks will be occupied at T+h?*

- **Target:** `target_occ` — occupancy as a fraction [0, 1] at T+h
- **Horizon h:** variable input to the model (1, 3, 6, 12, 24 h)
- **Granularity:** one prediction per station per horizon

## Data sources

| Source | Content | Access |
|---|---|---|
| [Open Data BCN](https://opendata-ajuntament.barcelona.cat) | Station status (5-min snapshots, 12 months 2024) | Free download (.7z per month) |
| [Open Data BCN](https://opendata-ajuntament.barcelona.cat) | Station info (location, capacity, name) | Free download |
| [Open-Meteo Historical API](https://open-meteo.com) | Hourly temperature, precipitation, wind speed | Free API |
| `holidays` Python library | Catalan public holidays 2024 | pip install |

## Project structure

```
bicing-occupancy-prediction/
├── data/
│   ├── raw/          # original .7z files from Open Data BCN (not in git)
│   └── processed/    # generated CSVs (not in git)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_baseline.ipynb
│   ├── 04_model_random_forest.ipynb
│   ├── 05_model_lightgbm.ipynb
│   └── 06_model_comparison.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py    # raw data loading and cleaning
│   │   └── features.py  # all feature engineering logic
│   ├── models/
│   │   └── evaluate.py  # shared metrics and plots
│   └── utils/
│       └── config.py    # paths, constants, feature list
├── models/              # serialized model artifacts (.joblib)
├── reports/figures/     # exported plots
└── deployment/
    └── DESIGN.md        # deployment architecture documentation
```

## Setup

```bash
git clone https://github.com/<your-username>/bicing-occupancy-prediction.git
cd bicing-occupancy-prediction
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the project

Run notebooks in order. Each notebook saves its outputs so the next one can pick up from there.

1. Put the raw `.7z` files from Open Data BCN in `data/raw/` (see `data/README.md` for the exact filenames).
2. Run `02_feature_engineering.ipynb` — this generates `data/processed/bicing_model_dataset.csv`.
3. Run notebooks 03–05 in any order to train and evaluate each model.
4. Run `06_model_comparison.ipynb` to compare results and select the final model.

Notebook `01_eda.ipynb` is exploratory and does not produce pipeline artifacts.

## Models

| Model | Role |
|---|---|
| Linear Regression | Baseline — establishes a performance floor |
| Random Forest | Strong non-linear benchmark |
| LightGBM | Primary model — best expected performance on tabular data |

All models are trained with `RandomizedSearchCV` and `TimeSeriesSplit` cross-validation to respect the temporal ordering of the data.

## Key design decisions

- **Variable horizon as a feature:** instead of one model per horizon, a single model receives `horizon_hours` as an input. This allows deployment at any horizon without retraining.
- **Temporal train/test split:** training on Jan–Oct 2024, testing on Nov–Dec 2024. No random splits — they leak future data into training.
- **Sin/cos encoding for cyclic features:** hour, day of week, and month are encoded as (sin, cos) pairs so the model understands that 23:00 and 00:00 are adjacent.
- **Leakage-safe historical baseline:** `hist_mean_occ` is computed on the training set only, then joined to the full dataset.

## Results

*To be filled after running the comparison notebook.*

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Naive baseline | — | — | — |
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| LightGBM | — | — | — |

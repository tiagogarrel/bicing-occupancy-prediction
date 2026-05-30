# Data

Neither raw nor processed files are tracked in git — they're too large.

## How to get the raw data

### Station status (12 monthly files)

1. Go to: https://opendata-ajuntament.barcelona.cat/data/ca/dataset/bicing
2. Download the 12 monthly files for 2024 — they are named like:
   `2024_01_Gener_BicingNou_ESTACIONS.7z`
3. Place all `.7z` files in `data/raw/`.

### Station info

Download the station information file (INFORMACIO) from the same page.
The January file is sufficient since station info rarely changes:
`2024_01_Gener_BicingNou_INFORMACIO.7z`

Place it in `data/raw/` alongside the status files.

### Weather

Fetched automatically from the Open-Meteo Historical API during notebook `02_feature_engineering.ipynb`. No download needed.

## Processed files (generated)

Running `02_feature_engineering.ipynb` produces:

| File | Description |
|---|---|
| `data/processed/bicing_model_dataset.csv` | Full feature-engineered dataset ready for modelling (~24M rows after horizon expansion) |

All other intermediate CSVs (decompressed from .7z) are also written to `data/processed/` and can be deleted after the model dataset is generated.

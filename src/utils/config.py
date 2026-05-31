from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PATHS = {
    "raw":       ROOT / "data" / "raw",
    "processed": ROOT / "data" / "processed",
    "models":    ROOT / "models",
    "figures":   ROOT / "reports" / "figures",
}

# Temporal split: train on Jan-Oct, test on Nov-Dec 2024
SPLIT_DATE = "2024-11-01"

# Prediction horizons (hours ahead)
HORIZONS = [1, 3, 6, 12, 24]

# Output file produced by 02_feature_engineering.ipynb.
# Parquet preserves float32 dtypes and is ~4x smaller than CSV.
DATASET_FILE = PATHS["processed"] / "bicing_model_dataset.parquet"

# Features fed into all models
# Note: raw hour/dow/month are excluded — sin/cos versions used instead
FEATURES = [
    # Cyclic temporal (current time T)
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    # Cyclic temporal (target time T+h)
    "target_hour_sin", "target_hour_cos",
    # Day type
    "is_weekend", "is_holiday",
    # Weather at T, used as a proxy for weather at T+h. For short horizons this
    # is a reasonable approximation; for 24h ahead it's less precise, but a proper
    # weather forecast integration is a deployment concern, not a training one.
    "temperature", "precipitation", "windspeed",
    # Station occupancy history
    "current_occ",       # strongest single predictor for short horizons
    "lag_24h",           # same station, 18h ago (= 24h before a T+6 target)
    "lag_168h",          # same station, 162h ago (= same slot last week)
    "rolling_mean_3h",   # recent trend in the 3h leading up to T
    "hist_mean_occ",     # long-run average for this station/hour/weekday — computed on train only
    # Distance features capture geographic behaviour without needing to one-hot
    # encode hundreds of station IDs. Beach stations behave very differently from
    # central ones; this lets the model learn that gradient continuously.
    "dist_beach", "dist_center",
    # Larger stations have more buffer — a station with 30 docks absorbs demand
    # swings that would saturate a 10-dock station.
    "capacity",
    # Horizon
    "horizon_hours",
]

TARGET = "target_occ"

# ---------------------------------------------------------------------------
# V2 feature set — used by 02b_feature_engineering_v2.ipynb
# ---------------------------------------------------------------------------

DATASET_FILE_V2 = PATHS["processed"] / "bicing_model_dataset_v2.parquet"

FEATURES_V2 = [
    # --- Cyclic temporal (current time T) ---
    # month_sin/cos removed: hist_mean_occ already encodes the seasonal pattern
    # per (station, hour, dow), so month adds redundancy without signal.
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    # Cyclic temporal (target time T+h)
    "target_hour_sin", "target_hour_cos",
    # Day type
    # is_weekend removed: dow_sin/cos already encode it; keeping both is redundant.
    "is_holiday",
    "is_holiday_tomorrow",   # horizon>=12h: tomorrow's pattern matters
    # Weather
    "temperature", "precipitation", "windspeed",
    # --- Occupancy history (richer lag set) ---
    "current_occ",          # strongest predictor for short horizons
    "lag_1h",               # very recent state
    "lag_2h",
    "lag_3h",
    "lag_24h",              # same station ~24h ago (via 18-step shift)
    "lag_48h",              # 2 days ago
    "lag_168h",             # same day-of-week last week
    "rolling_mean_3h",      # trend in last 3h
    "rolling_mean_6h",      # medium-term trend
    "rolling_mean_24h",     # daily average trend
    "hist_mean_occ",        # long-run (station, hour, dow) baseline — train only
    # --- Station features ---
    "dist_beach", "dist_center",
    "capacity",
    # --- Neighbor context ---
    # Mean occupancy of the N nearest stations at time T.
    # Captures flow: when neighbors fill up, this station tends to fill up too.
    "neighbor_mean_occ",
    # --- Horizon ---
    "horizon_hours",
]

RANDOM_STATE = 42

# Reference points for distance features
BARCELONETA = (41.3795, 2.1928)
PLACA_CAT   = (41.3874, 2.1700)

# Open-Meteo API settings
WEATHER_LAT = 41.39
WEATHER_LON = 2.16
WEATHER_TZ  = "Europe/Madrid"

"""
Feature engineering.

All transformations are pure functions that take a DataFrame and return a new one.
No side effects, no I/O.

Key design decisions:
  - Cyclic features (hour, day of week, month) encoded as (sin, cos) pairs
    so the model understands that 23:00 and 00:00 are adjacent.
  - hist_mean_occ is always computed on a separate reference DataFrame (the
    training set) and then joined, never on the full dataset, to avoid leakage.
  - Variable horizon: the dataset is expanded so each (station, T) row becomes
    one row per horizon. horizon_hours and target_hour_{sin,cos} are added.
"""

import numpy as np
import pandas as pd
import holidays
from math import radians, sin, cos, sqrt, atan2

from src.utils.config import HORIZONS, BARCELONETA, PLACA_CAT


# ---------------------------------------------------------------------------
# Cyclic encoding
# ---------------------------------------------------------------------------

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode hour, day_of_week and month as (sin, cos) pairs.

    A plain integer (hour=23, hour=0) looks like opposite ends of a scale
    to a model, but they're actually one minute apart. The sin/cos projection
    onto a circle fixes that: 23 and 0 end up close together in 2D space.
    Both components are needed — sin alone can't distinguish 1h from 11h.
    """
    df = df.copy()
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"]       / 12)
    return df


def add_target_hour_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the hour at which the prediction lands (T + horizon_hours).
    This tells the model whether it is predicting a morning rush, midnight, etc.
    """
    df = df.copy()
    target_hour = (df["hour"] + df["horizon_hours"]) % 24
    df["target_hour_sin"] = np.sin(2 * np.pi * target_hour / 24)
    df["target_hour_cos"] = np.cos(2 * np.pi * target_hour / 24)
    return df


# ---------------------------------------------------------------------------
# Temporal and day-type features
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day of week and month from datetime_hour."""
    df = df.copy()
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df["hour"]          = df["datetime_hour"].dt.hour
    df["day_of_week"]   = df["datetime_hour"].dt.dayofweek   # 0 = Monday
    df["month"]         = df["datetime_hour"].dt.month
    df["date"]          = df["datetime_hour"].dt.date
    return df


def add_day_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_weekend and is_holiday flags."""
    df = df.copy()
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    years = df["datetime_hour"].dt.year.unique().tolist()
    cat_holidays = holidays.Spain(prov="CT", years=years)
    # Barcelona local holidays
    for d in ["2024-04-23", "2024-09-24"]:
        cat_holidays[pd.to_datetime(d).date()] = "Local"

    df["is_holiday"] = df["date"].apply(lambda d: int(d in cat_holidays))
    return df


# ---------------------------------------------------------------------------
# Station distance features
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def add_distance_features(df_stations: pd.DataFrame) -> pd.DataFrame:
    """Add dist_beach and dist_center to the station info DataFrame."""
    df = df_stations.copy()
    df["dist_beach"]  = df.apply(lambda r: _haversine(r["lat"], r["lon"], *BARCELONETA), axis=1).round(3)
    df["dist_center"] = df.apply(lambda r: _haversine(r["lat"], r["lon"], *PLACA_CAT),   axis=1).round(3)
    return df


# ---------------------------------------------------------------------------
# Occupancy and lag features
# ---------------------------------------------------------------------------

def add_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute occupancy = bikes_total / capacity, clipped to [0, 1].

    clip() handles the rare cases where real-time counts briefly exceed capacity
    (e.g. a station reports 26 bikes with capacity 25 due to API lag). Without it
    those rows would become targets > 1 and pull regression predictions upward.
    """
    df = df.copy()
    df["occupancy"] = (df["bikes_total"] / df["capacity"]).clip(0, 1)
    df = df.dropna(subset=["occupancy"])
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features relative to T (the prediction moment).
    The dataset must be sorted by [station_id, datetime_hour] before calling this.

    Lag offsets are anchored to T, not to the target. The naming (lag_24h, lag_168h)
    refers to how far back they sit relative to a 6h-ahead target (T-18 = 24h before
    T+6), but since we later expand to multiple horizons, the actual semantic shifts
    slightly per horizon. For most use cases this is fine — the model learns the
    relationship implicitly through horizon_hours.
    """
    df = df.copy()
    grp = df.groupby("station_id")["occupancy"]

    df["current_occ"] = df["occupancy"]

    # Same station, 18h ago — captures the "same time of day yesterday" pattern
    df["lag_24h"]  = grp.shift(18)

    # Same station, 162h ago — same day-of-week last week, strong weekly seasonality
    df["lag_168h"] = grp.shift(162)

    # shift(1) excludes T itself so we're not using the current hour in its own average
    df["rolling_mean_3h"] = grp.transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    return df


def add_historical_baseline(df: pd.DataFrame,
                            reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hist_mean_occ: the average occupancy for each (station, hour, day_of_week)
    computed on reference_df (always the training set — never the full dataset).

    This prevents data leakage: the model cannot know future occupancy patterns
    at training time.
    """
    baseline = (
        reference_df
        .groupby(["station_id", "hour", "day_of_week"])["occupancy"]
        .mean()
        .reset_index()
        .rename(columns={"occupancy": "hist_mean_occ"})
    )
    df = df.merge(baseline, on=["station_id", "hour", "day_of_week"], how="left")
    return df


# ---------------------------------------------------------------------------
# Horizon expansion
# ---------------------------------------------------------------------------

# Columns kept during expansion — everything else is either redundant or
# an intermediate that was already used to derive a feature.
_EXPAND_KEEP = [
    "station_id", "datetime_hour",
    # cyclic temporal
    "hour", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    # day type
    "is_weekend", "is_holiday",
    # weather
    "temperature", "precipitation", "windspeed",
    # occupancy history
    "current_occ", "lag_24h", "lag_168h", "rolling_mean_3h", "hist_mean_occ",
    # station
    "dist_beach", "dist_center", "capacity",
    # raw occupancy still needed to compute the target via shift
    "occupancy",
]


def expand_horizons(df: pd.DataFrame,
                    horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """
    Convert a single-row-per-(station, T) DataFrame into one row per
    (station, T, horizon).

    Memory strategy:
      - Trim to _EXPAND_KEEP columns before looping so we don't copy
        dead columns five times.
      - Downcast float64 → float32 (halves footprint, plenty of precision
        for a regression on [0, 1] occupancy values).
      - Build each horizon chunk, drop nulls, then release the reference
        before moving to the next one so GC can reclaim memory.
    """
    # Work only with the columns we actually need downstream
    available = [c for c in _EXPAND_KEEP if c in df.columns]
    df_slim = df[available].copy()

    # float32 halves memory with no meaningful loss for this problem
    float_cols = df_slim.select_dtypes("float64").columns
    df_slim[float_cols] = df_slim[float_cols].astype("float32")

    # Pre-compute all horizon targets on the slim frame to avoid repeated groupby
    occ_by_station = df_slim.groupby("station_id")["occupancy"]

    chunks = []
    for h in horizons:
        chunk = df_slim.copy()
        chunk["target_occ"]    = occ_by_station.shift(-h).astype("float32")
        chunk["horizon_hours"] = np.int8(h)   # tiny int, no need for int64
        chunk = chunk.dropna(subset=["target_occ"])
        chunks.append(chunk)
        # Drop local reference so the garbage collector can free it after concat
        del chunk

    expanded = pd.concat(chunks, ignore_index=True)
    del chunks

    expanded = add_target_hour_cyclic(expanded)
    return expanded


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_model_dataset(df_full: pd.DataFrame,
                        train_mask: pd.Series,
                        horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """
    Apply all feature engineering steps and return the final model-ready DataFrame.

    df_full:    merged DataFrame with occupancy, weather, station info and temporal features
    train_mask: boolean Series (True for training rows) used to compute hist_mean_occ safely
    horizons:   list of prediction horizons to expand to
    """
    df = df_full.copy()
    df = df.sort_values(["station_id", "datetime_hour"]).reset_index(drop=True)
    df = add_lag_features(df)

    # Compute historical baseline on training rows only
    df = add_historical_baseline(df, reference_df=df[train_mask])

    # Drop rows with insufficient history (first 162h per station)
    df = df.dropna(subset=["lag_168h", "rolling_mean_3h"])

    df = add_cyclic_features(df)
    df = expand_horizons(df, horizons=horizons)

    return df

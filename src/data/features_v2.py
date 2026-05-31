"""
Feature engineering — V2.

Changes vs features.py (V1):
  - Richer lag set: lag_1h, lag_2h, lag_3h, lag_48h added.
  - Wider rolling windows: rolling_mean_6h, rolling_mean_24h added.
  - Neighbor context: neighbor_mean_occ — mean occupancy of the N nearest
    stations at time T, computed via a spatial join on the station info.
  - is_holiday_tomorrow: relevant for horizons >= 12h.
  - Redundant features removed: is_weekend (encoded by dow_sin/cos),
    month_sin/cos (captured by hist_mean_occ).

All transformations are pure functions. No side effects, no I/O.
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
    Encode hour and day_of_week as (sin, cos) pairs.

    month_sin/cos dropped vs V1: hist_mean_occ already captures the seasonal
    pattern per (station, hour, day_of_week). Adding month on top only
    introduces correlated signal that can confuse tree splits.
    """
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_target_hour_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the hour at which the prediction lands (T + horizon_hours)."""
    target_hour = (df["hour"] + df["horizon_hours"]) % 24
    df["target_hour_sin"] = np.sin(2 * np.pi * target_hour / 24)
    df["target_hour_cos"] = np.cos(2 * np.pi * target_hour / 24)
    return df


# ---------------------------------------------------------------------------
# Temporal and day-type features
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day_of_week and month from datetime_hour."""
    df = df.copy()
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df["hour"]          = df["datetime_hour"].dt.hour
    df["day_of_week"]   = df["datetime_hour"].dt.dayofweek   # 0 = Monday
    df["month"]         = df["datetime_hour"].dt.month
    df["date"]          = df["datetime_hour"].dt.date
    return df


def add_day_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_holiday and is_holiday_tomorrow flags.

    is_weekend removed vs V1: it is fully encoded by dow_sin/cos (the model
    can learn the weekend boundary from those two features alone).

    is_holiday_tomorrow is new: for horizons >= 12h the model needs to know
    whether the *target* day is a holiday. A pre-holiday evening behaves very
    differently from a regular weekday evening.
    """
    years = df["datetime_hour"].dt.year.unique().tolist()
    cat_holidays = holidays.Spain(prov="CT", years=years)
    # Barcelona local holidays
    for d in ["2024-04-23", "2024-09-24"]:
        cat_holidays[pd.to_datetime(d).date()] = "Local"

    df["is_holiday"] = df["date"].apply(lambda d: int(d in cat_holidays))

    # Tomorrow's holiday flag — shift date by one day
    tomorrow = pd.to_datetime(df["date"]) + pd.Timedelta(days=1)
    df["is_holiday_tomorrow"] = tomorrow.dt.date.apply(lambda d: int(d in cat_holidays))
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
    df_stations["dist_beach"]  = df_stations.apply(lambda r: _haversine(r["lat"], r["lon"], *BARCELONETA), axis=1).round(3)
    df_stations["dist_center"] = df_stations.apply(lambda r: _haversine(r["lat"], r["lon"], *PLACA_CAT),   axis=1).round(3)
    return df_stations


# ---------------------------------------------------------------------------
# Occupancy and lag features
# ---------------------------------------------------------------------------

def add_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute occupancy = bikes_total / capacity, clipped to [0, 1]."""
    df = df.copy()
    df["occupancy"] = (df["bikes_total"] / df["capacity"]).clip(0, 1)
    df = df.dropna(subset=["occupancy"])
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features relative to T (the prediction moment).
    The dataset must be sorted by [station_id, datetime_hour] before calling.

    V2 vs V1 changes:
      - Added lag_1h, lag_2h, lag_3h: short-horizon models benefit greatly
        from very recent state. Current occupancy + 1-3h lags together form
        a mini time-series view that captures the immediate trend direction.
      - Added lag_48h: 2-day-ago state, useful for weekly patterns with gaps.
      - Added rolling_mean_6h: bridges the 3h and 24h windows.
      - Added rolling_mean_24h: captures the day's average level, e.g. whether
        today is unusually busy relative to the historical baseline.

    All lags are anchored to T so they are always available at prediction time
    (no lookahead). shift(N) means N hours back from T.
    """
    grp = df.groupby("station_id")["occupancy"]

    df["current_occ"] = df["occupancy"]

    # Short lags — most informative for horizons <= 3h
    df["lag_1h"]  = grp.shift(1)
    df["lag_2h"]  = grp.shift(2)
    df["lag_3h"]  = grp.shift(3)

    # Medium / seasonal lags
    df["lag_24h"]  = grp.shift(18)    # ~24h before a 6h-ahead target
    df["lag_48h"]  = grp.shift(42)    # ~48h before a 6h-ahead target
    df["lag_168h"] = grp.shift(162)   # ~same slot last week

    # Rolling means — shift(1) to exclude T itself
    df["rolling_mean_3h"] = grp.transform(
        lambda x: x.shift(1).rolling(window=3,  min_periods=1).mean()
    )
    df["rolling_mean_6h"] = grp.transform(
        lambda x: x.shift(1).rolling(window=6,  min_periods=1).mean()
    )
    df["rolling_mean_24h"] = grp.transform(
        lambda x: x.shift(1).rolling(window=24, min_periods=6).mean()
    )
    return df


def add_historical_baseline(df: pd.DataFrame,
                             reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hist_mean_occ: average occupancy per (station, hour, day_of_week)
    computed on reference_df (always the training set — never the full dataset).
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
# Neighbor context
# ---------------------------------------------------------------------------

def build_neighbor_index(df_stations: pd.DataFrame,
                          n_neighbors: int = 5) -> dict[int, list[int]]:
    """
    For each station, find the N nearest stations by Haversine distance.

    Returns a dict: {station_id: [neighbor_id_1, ..., neighbor_id_N]}

    We exclude the station itself and cap at n_neighbors. Computed once
    from the station info DataFrame, then reused across all time steps.
    """
    ids   = df_stations["station_id"].values
    lats  = df_stations["lat"].values
    lons  = df_stations["lon"].values

    neighbor_map = {}
    for i, sid in enumerate(ids):
        dists = [
            (j, _haversine(lats[i], lons[i], lats[j], lons[j]))
            for j in range(len(ids)) if j != i
        ]
        dists.sort(key=lambda x: x[1])
        neighbor_map[int(sid)] = [int(ids[d[0]]) for d in dists[:n_neighbors]]

    return neighbor_map


def add_neighbor_mean_occ(df: pd.DataFrame,
                           neighbor_map: dict[int, list[int]]) -> pd.DataFrame:
    """
    Add neighbor_mean_occ: mean occupancy of the N nearest stations at time T.

    Algorithm:
      1. Pivot to a (datetime_hour × station_id) matrix of occupancy values.
      2. For each station, average the columns of its neighbors.
      3. Melt back to long format and merge.

    This is much faster than a row-wise apply over a grouped frame.

    Why this feature helps: demand at nearby stations is correlated.
    When all neighbors are full, this station is likely filling up too;
    when neighbors are empty, demand is low. This captures spatial spillover
    without requiring explicit station-to-station flow data.
    """
    # Build pivot: index=datetime_hour, columns=station_id, values=occupancy
    pivot = df.pivot_table(
        index="datetime_hour", columns="station_id", values="occupancy", aggfunc="mean"
    )

    # For each station compute the mean of its neighbors' columns
    neighbor_means = {}
    for station_id, neighbors in neighbor_map.items():
        valid_neighbors = [n for n in neighbors if n in pivot.columns]
        if valid_neighbors:
            neighbor_means[station_id] = pivot[valid_neighbors].mean(axis=1)
        else:
            neighbor_means[station_id] = pd.Series(np.nan, index=pivot.index)

    neighbor_df = pd.DataFrame(neighbor_means)  # shape: (hours, stations)
    neighbor_df.index.name = "datetime_hour"
    neighbor_df = neighbor_df.reset_index()

    # Melt to (datetime_hour, station_id, neighbor_mean_occ)
    neighbor_long = neighbor_df.melt(
        id_vars="datetime_hour",
        var_name="station_id",
        value_name="neighbor_mean_occ"
    )
    neighbor_long["station_id"] = neighbor_long["station_id"].astype(int)

    df = df.merge(neighbor_long, on=["datetime_hour", "station_id"], how="left")
    return df


# ---------------------------------------------------------------------------
# Horizon expansion
# ---------------------------------------------------------------------------

_EXPAND_KEEP_V2 = [
    "station_id", "datetime_hour",
    # cyclic temporal (month excluded in V2)
    "hour", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # day type (is_weekend excluded in V2)
    "is_holiday", "is_holiday_tomorrow",
    # weather
    "temperature", "precipitation", "windspeed",
    # occupancy history
    "current_occ",
    "lag_1h", "lag_2h", "lag_3h",
    "lag_24h", "lag_48h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_6h", "rolling_mean_24h",
    "hist_mean_occ",
    # neighbor context
    "neighbor_mean_occ",
    # station
    "dist_beach", "dist_center", "capacity",
    # target
    "occupancy",
]


def expand_horizons(df: pd.DataFrame,
                    horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """
    Convert a single-row-per-(station, T) DataFrame into one row per
    (station, T, horizon).

    Identical strategy to V1: trim columns, downcast to float32, build
    each horizon chunk separately to control memory.
    """
    available = [c for c in _EXPAND_KEEP_V2 if c in df.columns]
    df_slim = df[available].copy()
    del df  # free the full df before building chunks

    float_cols = df_slim.select_dtypes("float64").columns
    df_slim[float_cols] = df_slim[float_cols].astype("float32")

    chunks = []
    for h in horizons:
        chunk = df_slim.copy()
        chunk["target_occ"]    = df_slim.groupby("station_id")["occupancy"].shift(-h).values.astype("float32")
        chunk["horizon_hours"] = np.int8(h)
        chunk = chunk.dropna(subset=["target_occ"])
        chunks.append(chunk)
        del chunk

    del df_slim
    expanded = pd.concat(chunks, ignore_index=True)
    del chunks

    target_hour = (expanded["hour"] + expanded["horizon_hours"]) % 24
    expanded["target_hour_sin"] = np.sin(2 * np.pi * target_hour / 24).astype("float32")
    expanded["target_hour_cos"] = np.cos(2 * np.pi * target_hour / 24).astype("float32")

    return expanded


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_model_dataset_v2(df_full: pd.DataFrame,
                            df_stations: pd.DataFrame,
                            train_mask: pd.Series,
                            horizons: list[int] = HORIZONS,
                            n_neighbors: int = 5) -> pd.DataFrame:
    """
    Apply all V2 feature engineering steps and return the model-ready DataFrame.

    df_full:      merged DataFrame with occupancy, weather, station info and temporal features
    df_stations:  station info DataFrame with lat/lon (needed for neighbor index)
    train_mask:   boolean Series (True for training rows) — used for hist_mean_occ
    horizons:     list of prediction horizons
    n_neighbors:  how many nearest stations to average for neighbor_mean_occ
    """
    df = df_full.sort_values(["station_id", "datetime_hour"]).reset_index(drop=True)
    # Don't hold df_full in scope — caller should del it after calling this function.

    df = add_lag_features(df)
    # train_mask was built on df_full's original index; after sort+reset_index we must
    # remap it via datetime_hour to avoid index-misalignment silently selecting wrong rows.
    train_datetimes = df_full.loc[train_mask, "datetime_hour"]
    train_mask_sorted = df["datetime_hour"].isin(train_datetimes)
    df = add_historical_baseline(df, reference_df=df[train_mask_sorted])

    # Drop rows with insufficient history
    # rolling_mean_24h needs min 6 periods; lag_168h needs 162 steps
    df = df.dropna(subset=["lag_168h", "rolling_mean_24h"])

    # Neighbor context (computed after lag features so occupancy column exists)
    neighbor_map = build_neighbor_index(df_stations, n_neighbors=n_neighbors)
    df = add_neighbor_mean_occ(df, neighbor_map)

    add_cyclic_features(df)  # in-place, no reassignment needed
    return expand_horizons(df, horizons=horizons)

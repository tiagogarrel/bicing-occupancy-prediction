"""
Raw data loading and cleaning.

All functions read from data/raw/ and write to data/processed/.
Nothing here does feature engineering — that lives in features.py.
"""

import os
import py7zr
import pandas as pd
import requests
from pathlib import Path

from src.utils.config import PATHS, WEATHER_LAT, WEATHER_LON, WEATHER_TZ


def extract_all_7z(raw_path: Path = PATHS["raw"],
                   out_path: Path = PATHS["processed"]) -> list[str]:
    """Extract all .7z files in raw_path to out_path. Returns list of extracted filenames."""
    out_path.mkdir(parents=True, exist_ok=True)
    archives = sorted(raw_path.glob("*.7z"))

    if not archives:
        raise FileNotFoundError(
            f"No .7z files found in {raw_path}. "
            "See data/README.md for download instructions."
        )

    extracted = []
    for archive in archives:
        print(f"Extracting {archive.name}...")
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=out_path)
        extracted.append(archive.name)

    csvs = sorted(out_path.glob("*.csv"))
    print(f"\n{len(csvs)} CSV files available in {out_path}")
    return [a.name for a in archives]


def load_station_info(processed_path: Path = PATHS["processed"]) -> pd.DataFrame:
    """
    Load station metadata (location, capacity, name) from the INFORMACIO file.
    Uses the first available INFORMACIO CSV (January is enough — station info rarely changes).
    """
    info_files = sorted(processed_path.glob("*INFORMACIO*.csv"))
    if not info_files:
        raise FileNotFoundError("No INFORMACIO CSV found. Run extract_all_7z() first.")

    df = pd.read_csv(info_files[0], encoding="utf-8-sig")
    df = df.drop_duplicates(subset="station_id", keep="last")
    df["station_id"] = df["station_id"].astype(int)
    df["lat"]        = df["lat"].astype(float)
    df["lon"]        = df["lon"].astype(float)
    df["capacity"]   = df["capacity"].astype(int)

    df = df[["station_id", "name", "lat", "lon", "capacity"]].copy()
    print(f"Station info loaded: {len(df)} stations")
    return df


def load_station_status(processed_path: Path = PATHS["processed"]) -> pd.DataFrame:
    """
    Load and concatenate all 12 monthly ESTACIONS files.
    Aggregates the ~5-min raw snapshots to hourly averages per station.
    Keeps only IN_SERVICE stations.
    """
    estacions_files = sorted(processed_path.glob("*ESTACIONS*.csv"))
    if not estacions_files:
        raise FileNotFoundError("No ESTACIONS CSVs found. Run extract_all_7z() first.")

    dfs = []
    for path in estacions_files:
        print(f"  Loading {path.name}...")
        df = pd.read_csv(path, encoding="utf-8-sig")

        df["datetime"] = pd.to_datetime(df["last_updated"], unit="s")
        df = df.rename(columns={
            "num_bikes_available_types.mechanical": "bikes_mechanical",
            "num_bikes_available_types.ebike":      "bikes_electric",
            "num_bikes_available":                  "bikes_total",
            "num_docks_available":                  "docks_available",
        })
        df = df[df["status"] == "IN_SERVICE"].copy()
        df = df[["station_id", "datetime", "bikes_total",
                 "bikes_mechanical", "bikes_electric", "docks_available"]]
        dfs.append(df)

    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"\nRaw rows (IN_SERVICE): {len(df_raw):,}")

    # Aggregate to hourly averages
    df_raw["datetime_hour"] = df_raw["datetime"].dt.floor("h")
    df_hourly = (
        df_raw
        .groupby(["station_id", "datetime_hour"], as_index=False)
        .agg(
            bikes_total      =("bikes_total",      "mean"),
            bikes_mechanical =("bikes_mechanical",  "mean"),
            bikes_electric   =("bikes_electric",    "mean"),
            docks_available  =("docks_available",   "mean"),
        )
    )

    print(f"After hourly aggregation: {len(df_hourly):,} rows")
    return df_hourly


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly weather from Open-Meteo Historical API.
    Returns a DataFrame with columns: datetime_hour, temperature, precipitation, windspeed.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   WEATHER_LAT,
        "longitude":  WEATHER_LON,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     ["temperature_2m", "precipitation", "windspeed_10m"],
        "timezone":   WEATHER_TZ,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    h = resp.json()["hourly"]

    df = pd.DataFrame({
        "datetime_hour": pd.to_datetime(h["time"]),
        "temperature":   h["temperature_2m"],
        "precipitation": h["precipitation"],
        "windspeed":     h["windspeed_10m"],
    })
    print(f"Weather fetched: {len(df):,} hourly rows ({start_date} → {end_date})")
    return df


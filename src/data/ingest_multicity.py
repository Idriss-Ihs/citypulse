"""
ingest_multicity.py
-------------------
Historical ingestion for multiple Moroccan cities:
- Weather (daily) from Open-Meteo Archive API
- Air Quality (hourly -> daily mean) from Open-Meteo Air Quality API

Cities + dates are read from settings.yaml.

Outputs (per city) in data/raw/:
- weather_<city>.parquet
- air_quality_<city>_hourly.parquet
- air_quality_<city>_daily.parquet
- citypulse_<city>_daily.parquet   (merged weather + AQ daily)

Run:
    python -m src.data.ingest_multicity
"""

from __future__ import annotations
import time
from typing import Dict, Tuple
import pandas as pd
import requests
from pathlib import Path

from src.utils.logger import setup_logger
from src.data.utils import load_config


WEATHER_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_API = "https://air-quality-api.open-meteo.com/v1/air-quality"


def _retry_get(url: str, params: Dict, logger, retries: int = 4, backoff: float = 2.0) -> dict | None:
    """Simple GET with exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=40)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"[Attempt {attempt}/{retries}] GET failed: {e}")
            if attempt < retries:
                time.sleep(backoff ** attempt)
    logger.error(f"Failed after {retries} attempts for {url}")
    return None


def fetch_weather_daily(lat: float, lon: float, start_date: str, end_date: str, tz: str, logger) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max"
        ]),
        "timezone": tz
    }
    logger.info(f"Fetching WEATHER daily {start_date}→{end_date} for lat={lat}, lon={lon}")
    data = _retry_get(WEATHER_ARCHIVE, params, logger)
    if not data or "daily" not in data:
        logger.warning("Weather: empty response.")
        return pd.DataFrame()

    daily = data["daily"]
    df = pd.DataFrame(daily)
    # Expect a 'time' column plus metrics
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df = df.drop(columns=["time"])
    df = df.rename(columns={
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum": "precip",
        "windspeed_10m_max": "wind_max"
    })
    # Ensure numeric
    for c in ["temp_max", "temp_min", "precip", "wind_max"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_air_quality_hourly(lat: float, lon: float, start_date: str, end_date: str, tz: str, logger) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "us_aqi"
        ]),
        "timezone": tz
    }
    logger.info(f"Fetching AIR QUALITY hourly {start_date}→{end_date} for lat={lat}, lon={lon}")
    data = _retry_get(AIR_QUALITY_API, params, logger)
    if not data or "hourly" not in data:
        logger.warning("Air Quality: empty response.")
        return pd.DataFrame()

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    # Expect 'time' column (ISO strings)
    df["datetime"] = pd.to_datetime(df["time"])
    df["date"] = df["datetime"].dt.date
    df = df.drop(columns=["time"])
    # tidy type conversion
    for c in ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "us_aqi"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def aggregate_aq_daily(aq_hourly: pd.DataFrame) -> pd.DataFrame:
    if aq_hourly.empty:
        return pd.DataFrame()
    # daily mean of pollutants and AQI
    daily = aq_hourly.groupby("date").mean(numeric_only=True).reset_index()
    # rename caps for readability
    return daily.rename(columns={
        "pm2_5": "PM2_5",
        "pm10": "PM10",
        "ozone": "O3",
        "nitrogen_dioxide": "NO2",
        "sulphur_dioxide": "SO2",
        "carbon_monoxide": "CO",
        "us_aqi": "AQI"
    })


def merge_city_daily(weather_daily: pd.DataFrame, aq_daily: pd.DataFrame) -> pd.DataFrame:
    if weather_daily.empty and aq_daily.empty:
        return pd.DataFrame()
    if weather_daily.empty:
        df = aq_daily.copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")
    if aq_daily.empty:
        df = weather_daily.copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")
    df = pd.merge(weather_daily, aq_daily, on="date", how="outer")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def run():
    cfg = load_config()
    logger = setup_logger("ingest_multicity",f"{cfg['paths']['logs']}")
    

    tz = cfg["citypulse"]["tz"]
    start_date = cfg["citypulse"]["start_date"]
    end_date = cfg["citypulse"]["end_date"]
    cities: Dict[str, Tuple[float, float]] = cfg["citypulse"]["cities"]

    raw_dir = Path(cfg["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    for city, (lat, lon) in cities.items():
        safe = city.replace(" ", "_")
        logger.info(f"=== City: {city} ({lat},{lon}) ===")

        # WEATHER daily
        w = fetch_weather_daily(lat, lon, start_date, end_date, tz, logger)
        if not w.empty:
            w_path = raw_dir / f"weather_{safe}.parquet"
            w.to_parquet(w_path, index=False)
            logger.info(f"Saved {w_path} ({w.shape})")
        else:
            logger.warning(f"Weather empty for {city}")

        # AIR QUALITY hourly
        aq_h = fetch_air_quality_hourly(lat, lon, start_date, end_date, tz, logger)
        if not aq_h.empty:
            aqh_path = raw_dir / f"air_quality_{safe}_hourly.parquet"
            aq_h.to_parquet(aqh_path, index=False)
            logger.info(f"Saved {aqh_path} ({aq_h.shape})")
            # aggregate to daily
            aq_d = aggregate_aq_daily(aq_h)
            aqd_path = raw_dir / f"air_quality_{safe}_daily.parquet"
            aq_d.to_parquet(aqd_path, index=False)
            logger.info(f"Saved {aqd_path} ({aq_d.shape})")
        else:
            aq_d = pd.DataFrame()
            logger.warning(f"AQ hourly empty for {city}")

        # MERGE daily (weather + AQ daily)
        merged = merge_city_daily(w, aq_d)
        if not merged.empty:
            out = raw_dir / f"citypulse_{safe}_daily.parquet"
            merged.to_parquet(out, index=False)
            logger.info(f"Saved {out} ({merged.shape})")
        else:
            logger.warning(f"Merged daily empty for {city}")

    logger.info("✅ Multi-city ingestion completed.")


if __name__ == "__main__":
    run()

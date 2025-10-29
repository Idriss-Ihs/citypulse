"""
ingest_city_data.py
--------------------
Fetch air quality and weather data from open APIs and save to data/raw.
"""

import requests
import pandas as pd
from pathlib import Path
import yaml
from src.utils.logger import setup_logger


def load_config(path="src/config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_air_quality(cfg, logger):
    """Fetch air quality data from Open-Meteo (free & global)."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 33.5731,   # Casablanca
        "longitude": -7.5898,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
    }

    logger.info("Fetching air quality data (Open-Meteo)...")
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data["hourly"])
        logger.info(f"Fetched {len(df)} air quality hourly records.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch air quality: {e}")
        return pd.DataFrame()
def fetch_mobility_data(cfg, logger):
    """Fetch recent mobility data (Google COVID mobility reports)."""
    url = "https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip"
    try:
        logger.info("Fetching mobility data (Google)...")
        zip_path = Path(cfg["paths"]["raw"]) / "mobility.zip"
        Path(cfg["paths"]["raw"]).mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        df = pd.read_csv(f"zip://{zip_path}!2023_MA_Region_Mobility_Report.csv")
        logger.info(f"Fetched {len(df)} mobility rows.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch mobility data: {e}")
        return pd.DataFrame()


def fetch_weather(cfg, logger):
    """Fetch weather data from Open-Meteo API."""
    url = cfg["data_sources"]["weather"]
    params = {
        "latitude": 33.5731,  # Casablanca
        "longitude": -7.5898,
        "hourly": "temperature_2m,precipitation,wind_speed_10m"
    }
    logger.info("Fetching weather data...")
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data["hourly"])
        logger.info(f"Fetched {len(df)} weather records.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch weather: {e}")
        return pd.DataFrame()


def save_data(df, name, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    output_file = path / f"{name}.csv"
    df.to_csv(output_file, index=False)
    return output_file


if __name__ == "__main__":
    cfg = load_config()
    logger = setup_logger("ingest_city_data",  f"{cfg['paths']['logs']}/ingest_city_data.log")

    air_df = fetch_air_quality(cfg, logger)
    weather_df = fetch_weather(cfg, logger)
    mobility_df = fetch_mobility_data(cfg, logger)

    if not air_df.empty:
        save_data(air_df, "air_quality", cfg["paths"]["raw"])
    if not weather_df.empty:
        save_data(weather_df, "weather", cfg["paths"]["raw"])
    if not mobility_df.empty:
        save_data(mobility_df, "mobility", cfg["paths"]["raw"])

    logger.info("✅ City data ingestion complete.")
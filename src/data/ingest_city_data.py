"""
CityPulse - Unified Data Ingestion
Fetches weather, air quality, mobility, energy and light activity data.
"""

import requests, pandas as pd, zipfile, io
from pathlib import Path
import yaml
from src.utils.logger import setup_logger


def load_config(path="src/config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_weather(cfg, logger):
    lat, lon = cfg["city"]["latitude"], cfg["city"]["longitude"]
    url = cfg["data_sources"]["weather"]
    params = {"latitude": lat, "longitude": lon,
              "hourly": "temperature_2m,precipitation,wind_speed_10m"}
    logger.info("Fetching weather data...")
    try:
        r = requests.get(url, params=params, timeout=20); r.raise_for_status()
        df = pd.DataFrame(r.json()["hourly"])
        logger.info(f"Weather rows: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return pd.DataFrame()


def fetch_air_quality(cfg, logger):
    lat, lon = cfg["city"]["latitude"], cfg["city"]["longitude"]
    url = cfg["data_sources"]["air_quality"]
    params = {"latitude": lat, "longitude": lon,
              "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,us_aqi"}
    logger.info("Fetching air quality data...")
    try:
        r = requests.get(url, params=params, timeout=20); r.raise_for_status()
        df = pd.DataFrame(r.json()["hourly"])
        logger.info(f"Air-quality rows: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Air quality fetch failed: {e}")
        return pd.DataFrame()


def fetch_mobility(cfg, logger):
    url = cfg["data_sources"]["mobility_zip"]
    logger.info("Fetching mobility data...")
    try:
        r = requests.get(url, timeout=30); r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        # pick Morocco file dynamically
        fname = [f for f in z.namelist() if "MA" in f and f.endswith(".csv")][0]
        df = pd.read_csv(z.open(fname))
        logger.info(f"Mobility rows: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Mobility fetch failed: {e}")
        return pd.DataFrame()


def fetch_energy(cfg, logger):
    url = cfg["data_sources"]["energy_csv"]
    logger.info("Fetching energy data...")
    try:
        df = pd.read_csv(url)
        df = df[df["Country"] == "Morocco"]
        logger.info(f"Energy rows: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Energy fetch failed: {e}")
        return pd.DataFrame()


def save(df, name, cfg):
    outdir = Path(cfg["paths"]["raw"]); outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    cfg = load_config()
    logger = setup_logger("ingest_city_data")

    datasets = {
        "weather": fetch_weather(cfg, logger),
        "air_quality": fetch_air_quality(cfg, logger),
        "mobility": fetch_mobility(cfg, logger),
        "energy": fetch_energy(cfg, logger)
    }

    for name, df in datasets.items():
        if not df.empty:
            save(df, name, cfg)
            logger.info(f"✅ Saved {name} data.")
        else:
            logger.warning(f"⚠️ {name} data unavailable.")

    logger.info("🌆 CityPulse ingestion completed successfully.")

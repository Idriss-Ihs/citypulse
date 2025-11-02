"""
CityPulse - Phase 2: Data Fusion & Cleaning
------------------------------------------
Combines weather, air quality, mobility and energy datasets into one clean file.
"""

import pandas as pd
from pathlib import Path
import yaml
from src.utils.logger import setup_logger


def load_config(path="src/config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_csv_safe(path, logger):
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {path} ({df.shape[0]} rows, {df.shape[1]} cols)")
        return df
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return pd.DataFrame()


def preprocess_weather(df, logger):
    if df.empty:
        return df
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"])
    else:
        logger.warning("Weather: no 'time' column found.")
        return pd.DataFrame()
    df = df.rename(columns={"temperature_2m": "temperature",
                            "precipitation": "rain",
                            "wind_speed_10m": "wind"})
    df = df[["date", "temperature", "rain", "wind"]]
    df = df.resample("D", on="date").mean().reset_index()
    return df


def preprocess_air_quality(df, logger):
    if df.empty:
        return df
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"])
    else:
        logger.warning("Air quality: no 'time' column found.")
        return pd.DataFrame()
    rename_map = {
        "pm10": "PM10", "pm2_5": "PM2_5", "ozone": "O3",
        "carbon_monoxide": "CO", "nitrogen_dioxide": "NO2", "us_aqi": "AQI"
    }
    df = df.rename(columns=rename_map)
    df = df[["date"] + list(rename_map.values())]
    df = df.resample("D", on="date").mean().reset_index()
    return df


def preprocess_mobility(df, logger):
    if df.empty:
        return df
    if "date" not in df.columns:
        logger.warning("Mobility: expected 'date' column.")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    mask = df["sub_region_1"].isna() & (df["country_region_code"] == "MA")
    df = df[mask].copy()
    cols = [
        "date",
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline"
    ]
    df = df[cols].rename(columns=lambda c: c.replace("_percent_change_from_baseline", ""))
    df = df.resample("D", on="date").mean().reset_index()
    return df


def preprocess_energy(df, logger):
    if df.empty:
        return df
    if "Date" not in df.columns:
        logger.warning("Energy: expected 'Date' column.")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.rename(columns={"Electricity consumption (MWh)": "energy_MWh"})
    df = df[["date", "energy_MWh"]].dropna()
    df = df.resample("D", on="date").mean().reset_index()
    return df


def fuse_and_clean(cfg, logger):
    raw_path = Path(cfg["paths"]["raw"])
    interim_path = Path(cfg["paths"]["interim"])
    interim_path.mkdir(parents=True, exist_ok=True)

    weather = preprocess_weather(read_csv_safe(raw_path / "weather.csv", logger), logger)
    air = preprocess_air_quality(read_csv_safe(raw_path / "air_quality.csv", logger), logger)
    mob = preprocess_mobility(read_csv_safe(raw_path / "mobility.csv", logger), logger)
    energy = preprocess_energy(read_csv_safe(raw_path / "energy_daily_estimate.csv", logger), logger)

    logger.info("Merging datasets ...")
    df = (
        weather.merge(air, on="date", how="outer")
        .merge(mob, on="date", how="outer")
        .merge(energy, on="date", how="outer")
        .sort_values("date")
    )

    # Fill missing values
    df = df.fillna(method="ffill").fillna(method="bfill")

    # Clip impossible values
    if "AQI" in df.columns:
        df["AQI"] = df["AQI"].clip(lower=0, upper=500)
    if "temperature" in df.columns:
        df["temperature"] = df["temperature"].clip(-10, 50)

    out_path = interim_path / "citypulse_master.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f" Saved unified dataset to {out_path} ({df.shape})")


if __name__ == "__main__":
    cfg = load_config()
    logger = setup_logger("fuse_clean_city_data", f"{cfg['paths']['logs']}")
    fuse_and_clean(cfg, logger)


"""
city_health_score.py
--------------------
Compute interpretable daily sub-scores and a 0–100 City Health Score per city.

Inputs:
  data/interim/citypulse_master.parquet    # from fuse_multicity.py

Outputs:
  data/processed/citypulse_health_daily.parquet
    columns:
      date, city,
      air_score, temp_score, precip_score, wind_score,
      health_score, health_band
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.utils import load_config
from src.utils.logger import setup_logger


def _clip01(x):
    return np.minimum(1.0, np.maximum(0.0, x))


def air_quality_to_score(aqi: pd.Series) -> pd.Series:
    """
    Map US AQI (0..500) to a 0..100 score (higher is better).
    Linear, simple, transparent.
    """
    s = 1.0 - (aqi.fillna(0) / 500.0)
    return (_clip01(s) * 100.0).round(1)


def temperature_to_score(tmin: pd.Series, tmax: pd.Series, city: str) -> pd.Series:
    """
    Comfort band around ~21°C mean. Penalize distance from 21°C.
    Score = 100 - k * |t_mean - 21|
    k chosen so ~15°C away => ~25 points penalty.
    """
    tmean = (tmin.fillna(tmax) + tmax.fillna(tmin)) / 2.0
    k = 5.0  # penalty per °C
    score = 100.0 - k * (tmean - 21.0).abs()
    return score.clip(0, 100).round(1)


def precip_to_score(precip: pd.Series) -> pd.Series:
    """
    Penalize strong anomalies (both very dry and very wet) via rolling Z-score.
    """
    x = precip.fillna(0.0)
    roll = x.rolling(window=60, min_periods=15)
    mu = roll.mean()
    sd = roll.std(ddof=0).replace(0, np.nan)
    z = (x - mu) / sd
    # Convert |z| to score (|z|=0 => 100, |z|>=3 => ~40)
    score = 100.0 - (np.abs(z).clip(lower=0, upper=3) / 3.0) * 60.0
    return score.fillna(80.0).clip(0, 100).round(1)


def wind_to_score(wind_max: pd.Series) -> pd.Series:
    """
    Penalize high wind gusts; assume 20 m/s is very windy.
    """
    x = wind_max.fillna(0.0)
    score = 100.0 - (x / 20.0) * 100.0
    return score.clip(0, 100).round(1)


def band_from_score(s: pd.Series) -> pd.Series:
    """
    Map 0..100 to qualitative bands.
    """
    bins = [-1, 40, 60, 80, 100]
    labels = ["Poor", "Fair", "Good", "Excellent"]
    return pd.cut(s, bins=bins, labels=labels)


def compute_city_health(df: pd.DataFrame, logger) -> pd.DataFrame:
    req_cols = ["date", "city"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Normalize column names we expect (present if AQ was ingested)
    # Weather
    tmin = df.get("temp_min")
    tmax = df.get("temp_max")
    precip = df.get("precip")
    wind_max = df.get("wind_max")
    # Air quality (from Open-Meteo AQ daily aggregate)
    aqi = df.get("AQI")

    # Build sub-scores with graceful fallbacks
    air_score = air_quality_to_score(aqi) if aqi is not None else pd.Series(80.0, index=df.index)
    temp_score = temperature_to_score(tmin, tmax, city="") if (tmin is not None or tmax is not None) else pd.Series(75.0, index=df.index)
    precip_score = precip_to_score(precip) if precip is not None else pd.Series(80.0, index=df.index)
    wind_score = wind_to_score(wind_max) if wind_max is not None else pd.Series(85.0, index=df.index)

    # Weights (transparent & editable)
    w_air, w_temp, w_precip, w_wind = 0.5, 0.3, 0.1, 0.1

    health = (
        w_air * air_score +
        w_temp * temp_score +
        w_precip * precip_score +
        w_wind * wind_score
    ).round(1)

    out = df[["date", "city"]].copy()
    out["air_score"] = air_score
    out["temp_score"] = temp_score
    out["precip_score"] = precip_score
    out["wind_score"] = wind_score
    out["health_score"] = health
    out["health_band"] = band_from_score(health)

    # clean types
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


def run():
    logger = setup_logger("city_health_score")
    cfg = load_config()
    interim = Path(cfg["paths"]["interim"])
    processed = Path(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    master_path = interim / "citypulse_master.parquet"
    if not master_path.exists():
        logger.error(f"Master dataset not found: {master_path}")
        return

    df = pd.read_parquet(master_path)
    logger.info(f"Loaded master: {df.shape}")

    # Sort and group per city for rolling calcs stability
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    scores = df.groupby("city", group_keys=True).apply(lambda g: compute_city_health(g, logger))
    scores = scores.reset_index(drop=True)

    out_path = processed / "citypulse_health_daily.parquet"
    scores.to_parquet(out_path, index=False)
    logger.info(f"Saved health scores: {out_path} ({scores.shape})")

    # Quick distribution sanity logs
    desc = scores.groupby("city")["health_score"].describe().round(2)
    logger.info(f"\nHealth score summary by city:\n{desc.to_string()}")


if __name__ == "__main__":
    run()
  
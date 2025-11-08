"""
detect_events.py
----------------
Flag urban stress events (bad days) from health score.

Rules:
- acute_low: lowest 10% health score per city
- sudden_drop: drop >= 15 points vs 7 days ago
- pollution_spike: AQI > 150

Output: data/processed/citypulse_events.parquet
"""

import pandas as pd
from pathlib import Path
from src.utils.logger import setup_logger
from src.data.utils import load_config


def run():
    cfg = load_config()
    logger = setup_logger("events",f"{cfg['paths']['logs']}")

    
    processed = Path(cfg["paths"]["processed"])
    out = processed / "citypulse_events.parquet"

    src = processed / "citypulse_health_daily.parquet"
    if not src.exists():
        logger.error(f"Health file not found: {src}")
        return

    df = pd.read_parquet(src)
    df["date"] = pd.to_datetime(df["date"])

    events = []

    for city, g in df.groupby("city"):
        g = g.sort_values("date").reset_index(drop=True)

        # 1) bottom 10% health
        th = g["health_score"].quantile(0.10)
        acute = g[g["health_score"] <= th].copy()
        acute["event_type"] = "acute_low"
        acute["severity"] = (100 - acute["health_score"]).round(1)
        events.append(acute[["date", "city", "event_type", "severity"]])

        # 2) sudden drop vs 7d
        g["lag7"] = g["health_score"].shift(7)
        drop = g[(g["lag7"].notna()) & ((g["health_score"] - g["lag7"]) <= -15)].copy()
        drop["event_type"] = "sudden_drop"
        drop["severity"] = (drop["lag7"] - drop["health_score"]).round(1)
        events.append(drop[["date", "city", "event_type", "severity"]])

        # 3) pollution spike
        if "AQI" in g.columns:
            pol = g[g["AQI"] > 150].copy()
            pol["event_type"] = "pollution_spike"
            pol["severity"] = (pol["AQI"] - 150).round(1)
            events.append(pol[["date", "city", "event_type", "severity"]])

    if not events:
        logger.warning("No events detected.")
        return

    all_events = pd.concat(events, ignore_index=True)
    all_events = all_events.sort_values(["city", "date"])

    out.parent.mkdir(parents=True, exist_ok=True)
    all_events.to_parquet(out, index=False)
    logger.info(f"Saved events to {out} ({len(all_events)} rows).")


if __name__ == "__main__":
    run()

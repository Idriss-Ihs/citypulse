"""
fuse_multicity.py
-----------------
Combine all citypulse_<city>_daily parquet files into a single master table.
"""

import pandas as pd
from pathlib import Path
from src.data.utils import load_config
from src.utils.logger import setup_logger


def fuse_multicity():
    cfg = load_config()
    logger = setup_logger("fuse_multicity",f"{cfg['paths']['logs']}")

    raw = Path(cfg["paths"]["raw"])
    interim = Path(cfg["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)

    cities = cfg["citypulse"]["cities"].keys()

    dfs = []
    for city in cities:
        safe = city.replace(" ", "_")
        path = raw / f"citypulse_{safe}_daily.parquet"
        if not path.exists():
            logger.warning(f"Missing file for city: {city}")
            continue
        df = pd.read_parquet(path)
        df["city"] = city
        dfs.append(df)

    if not dfs:
        logger.error("No citypulse parquet files found.")
        return

    master = pd.concat(dfs, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"])
    master = master.sort_values(["city", "date"]).reset_index(drop=True)

    out = interim / "citypulse_master.parquet"
    master.to_parquet(out, index=False)

    logger.info(f"Created master table: {out} ({len(master)} rows)")


if __name__ == "__main__":
    fuse_multicity()

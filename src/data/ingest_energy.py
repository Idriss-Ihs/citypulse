import pandas as pd
from pathlib import Path
import requests
from src.utils.logger import setup_logger
from src.data.utils import save_data, load_config

RAW_CSV = "https://raw.githubusercontent.com/wri/global-power-plant-database/refs/heads/master/output_database/global_power_plant_database.csv"


def ingest_energy():
    cfg = load_config()
    logger = setup_logger("energy_ingest")
    

    logger.info("Fetching WRI Global Power Plant DB (public CSV)...")
    try:
        df = pd.read_csv(RAW_CSV)
    except Exception as e:
        logger.error(f"Failed to download energy data: {e}")
        return

    # Keep Morocco (ISO3 'MAR')
    df_ma = df[df["country"] == "MAR"].copy()
    if df_ma.empty:
        logger.warning("No Morocco power plant data found in the dataset.")
        return

    # Simple capacity factor mapping
    default_cf = {
        "Hydro": 0.45, "Solar": 0.22, "Wind": 0.38,
        "Oil": 0.40, "Gas": 0.50, "Coal": 0.70, "Nuclear": 0.85,
    }
    def estimate_cf(fuel):
        fuel = str(fuel).lower()
        for k, v in default_cf.items():
            if k.lower() in fuel:
                return v
        return 0.35  # fallback

    df_ma["capacity_factor"] = df_ma["primary_fuel"].apply(estimate_cf)
    df_ma["daily_estimated_MWh"] = df_ma["capacity_mw"] * 24 * df_ma["capacity_factor"]

    # Save per-plant table + national baseline (constant)
    save_data(df_ma, "energy_plants", cfg["paths"]["raw"])
    total_daily = df_ma["daily_estimated_MWh"].sum()
    save_data(pd.DataFrame({"daily_MWh_estimate":[total_daily]}), "energy_daily_estimate", cfg["paths"]["raw"])

    logger.info(f"Saved energy_plants.csv ({len(df_ma)} rows) and energy_daily_estimate.csv (MWh={total_daily:,.0f}).")

    
if __name__ == "__main__":
    ingest_energy()
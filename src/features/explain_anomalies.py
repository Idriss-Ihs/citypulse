"""
Explain events by attributing which sub-scores deviated most (z-scores).
Outputs:
  data/processed/citypulse_event_explanations.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.data.utils import load_config
from src.utils.logger import setup_logger

SUBS = ["air_score","temp_score","precip_score","wind_score"]

def trailing_stats(g: pd.DataFrame, on: str, window=60):
    # rolling stats up to previous day
    m = g[on].rolling(window, min_periods=15).mean()
    s = g[on].rolling(window, min_periods=15).std(ddof=0)
    return m.shift(1), s.shift(1)

def main():
    logger = setup_logger("explain_anomalies")
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])

    scores = pd.read_parquet(processed / "citypulse_health_daily.parquet")
    scores["date"] = pd.to_datetime(scores["date"])
    events = pd.read_parquet(processed / "citypulse_events.parquet")
    events["date"] = pd.to_datetime(events["date"])

    out_rows = []
    for city, g in scores.sort_values("date").groupby("city"):
        g = g.reset_index(drop=True)
        # compute trailing mean/std for each sub-score
        stats = {}
        for c in SUBS:
            mu, sd = trailing_stats(g, c, window=60)
            stats[c] = (mu, sd)
        g_stats = g.copy()
        for c in SUBS:
            mu, sd = stats[c]
            z = (g[c] - mu) / sd
            g_stats[c+"_z"] = z.replace([np.inf,-np.inf], np.nan)

        # join with events of this city
        ev = events[events.city==city].merge(g_stats[["date"] + SUBS + [s+"_z" for s in SUBS]], on="date", how="left")
        if ev.empty:
            continue

        # rank contributors by lowest sub-score z (most negative shock)
        def top_contributors(row):
            zs = {f: row[f+"_z"] for f in SUBS}
            zs = {k:v for k,v in zs.items() if pd.notna(v)}
            if not zs:
                return pd.Series({"driver_1": None, "driver_2": None})
            top2 = sorted(zs.items(), key=lambda kv: kv[1])[:2]
            return pd.Series({"driver_1": top2[0][0] if len(top2)>0 else None,
                              "driver_2": top2[1][0] if len(top2)>1 else None})

        drivers = ev.apply(top_contributors, axis=1)
        ev = pd.concat([ev, drivers], axis=1)
        out_rows.append(ev[["date","city","event_type","severity","driver_1","driver_2","air_score","temp_score","precip_score","wind_score","air_score_z","temp_score_z","precip_score_z","wind_score_z"]])

    if out_rows:
        expl = pd.concat(out_rows, ignore_index=True).sort_values(["city","date"])
        out_path = processed / "citypulse_event_explanations.parquet"
        expl.to_parquet(out_path, index=False)
        logger.info(f"Saved explanations → {out_path} ({len(expl)} rows)")
    else:
        logger.warning("No events found to explain.")

if __name__ == "__main__":
    main()

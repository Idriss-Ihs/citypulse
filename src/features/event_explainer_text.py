from datetime import date as _date
import pandas as pd
from typing import Optional
from src.features.utils_event_text import score_state

def _fmt(z: Optional[float]) -> str:
    """Format z-scores compactly with sign; handle NaNs."""
    try:
        if pd.isna(z): 
            return "n/a"
        return f"{z:+.2f}"
    except Exception:
        return "n/a"

def _sev_label(x) -> str:
    """Map numeric/str severity to a clean label."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "unspecified"
    s = str(x).lower()
    if s in {"0","low","mild"}: return "Low"
    if s in {"1","med","medium","moderate"}: return "Moderate"
    if s in {"2","high","severe"}: return "High"
    return s.title()

def explain_event(row: pd.Series, health_df: pd.DataFrame) -> str:
    """
    Build a deterministic markdown explanation for one event.

    Parameters
    ----------
    row : pd.Series
        Row from citypulse_event_explanations.parquet
        Required fields:
        ['city','date','event_type','severity','driver_1','driver_2',
         'air_score','temp_score','precip_score','wind_score',
         'air_score_z','temp_score_z','precip_score_z','wind_score_z']
    health_df : pd.DataFrame
        data/processed/citypulse_health_daily.parquet
        Must contain ['city','date','health_score'].
    """
    city = row["city"]
    dt   = pd.to_datetime(row["date"]).date() if not isinstance(row["date"], _date) else row["date"]

    # fetch overall health for that day
    hrow = health_df[(health_df["city"]==city) & (pd.to_datetime(health_df["date"])==pd.to_datetime(dt))]
    health = float(hrow["health_score"].iloc[0]) if not hrow.empty else None

    # sub-scores & z
    air_s, tmp_s, prc_s, wnd_s = row.get("air_score"), row.get("temp_score"), row.get("precip_score"), row.get("wind_score")
    air_z, tmp_z, prc_z, wnd_z = row.get("air_score_z"), row.get("temp_score_z"), row.get("precip_score_z"), row.get("wind_score_z")

    # states
    air_state  = score_state(air_s)
    temp_state = score_state(tmp_s)
    prcp_state = score_state(prc_s)
    wind_state = score_state(wnd_s)

    # drivers
    d1 = row.get("driver_1")
    d2 = row.get("driver_2")

    sev = _sev_label(row.get("severity"))
    etype = (row.get("event_type") or "Event").title()

    # Build markdown
    header = f"**{etype} — {city} — {dt}**  \n"
    hline  = f"**City health**: {health:.1f}/100  \n" if health is not None else "**City health**: n/a  \n"
    drivers = f"**Top drivers**: {d1 or '—'}{', ' + d2 if d2 else ''}  \n"
    sevline = f"**Severity**: {sev}  \n"

    detail = (
        "**Condition details**  \n"
        f"- Air quality: **{air_state}** (score {air_s:.1f} · z={_fmt(air_z)})  \n"
        f"- Temperature stress: **{temp_state}** (score {tmp_s:.1f} · z={_fmt(tmp_z)})  \n"
        f"- Precipitation stress: **{prcp_state}** (score {prc_s:.1f} · z={_fmt(prc_z)})  \n"
        f"- Wind stress: **{wind_state}** (score {wnd_s:.1f} · z={_fmt(wnd_z)})  \n"
    )

    # one-sentence summary (rule-based)
    summary_bits = []
    for lab, z in [("air", air_z), ("temperature", tmp_z), ("precipitation", prc_z), ("wind", wnd_z)]:
        try:
            if pd.notna(z) and z <= -1.0:
                summary_bits.append(f"{lab} anomaly (z≈{z:+.1f})")
        except Exception:
            pass
    summary = "Primary stressors: " + ", ".join(summary_bits) if summary_bits else "Primary stressors: mixed / moderate."

    return f"""
**{city} — {dt}**  
**City Health:** {health:.1f} / 100  
**Event type:** {etype} (severity: {sev})

---

**What happened?**  
This day shows anomalies mainly linked to **{d1}**{', and ' + d2 if d2 else ''}.  

---

**Drivers behind this day**

| Factor | Situation | Z-anomaly |
|---|---|---:|
| Air Quality | {air_state} | {air_z:+.2f} |
| Temperature | {temp_state} | {tmp_z:+.2f} |
| Precipitation | {prcp_state} | {prc_z:+.2f} |
| Wind | {wind_state} | {wnd_z:+.2f} |

---

**Short interpretation**  
{summary}
"""


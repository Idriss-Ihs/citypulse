import streamlit as st
import pandas as pd
from pathlib import Path
import pydeck as pdk

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

st.set_page_config(page_title="CityPulse • Map", layout="wide")
st.title("CityPulse — City Map (Health & Air Quality)")

scores = pd.read_parquet(DATA / "citypulse_health_daily.parquet")
scores["date"] = pd.to_datetime(scores["date"])

# Latest day per city
latest = scores.sort_values("date").groupby("city").tail(1).copy()

# Attach coordinates from config
import yaml
cfg = yaml.safe_load(open(ROOT / "src" / "config" / "settings.yaml"))
city_coords = pd.DataFrame(
    [(c, lat, lon) for c,(lat,lon) in cfg["citypulse"]["cities"].items()],
    columns=["city","lat","lon"]
)
latest = latest.merge(city_coords, on="city", how="left")

st.sidebar.header("Filters")
metric = st.sidebar.selectbox("Metric", ["health_score", "AQI", "temp_max", "precip", "wind_max"])
st.sidebar.write("Circle size ~ abs(metric). Color ~ health_score (green good, red bad).")

# Scale size
val = latest[metric].fillna(0)
size = (val - val.min()) / (val.max() - val.min() + 1e-9)
latest["size"] = (size * 1000 + 500).clip(300, 2000)

# Color by health (0 → red, 100 → green)
h = latest["health_score"].fillna(60)
latest["color"] = list(zip((100-h)*2.55, h*2.55, 80 + 0*h))

layer = pdk.Layer(
    "ScatterplotLayer",
    data=latest,
    get_position='[lon, lat]',
    get_radius='size',
    get_fill_color='color',
    pickable=True,
    auto_highlight=True,
)

view = pdk.ViewState(latitude=33.9, longitude=-6.6, zoom=5.2)
r = pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text":"{city}\nHealth: {health_score}\nAQI: {AQI}"})
st.pydeck_chart(r)

st.subheader("Latest Snapshot")
st.dataframe(latest[["city","date","health_score","AQI","temp_max","precip","wind_max"]].sort_values("city"))

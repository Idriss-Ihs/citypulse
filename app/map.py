import streamlit as st
import pandas as pd
from pathlib import Path
import pydeck as pdk
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

st.set_page_config(page_title="CityPulse • Map", layout="wide")
st.title("CityPulse — Urban Health Map")

# ---------------------------
# Load data
# ---------------------------
df = pd.read_parquet(DATA / "citypulse_health_daily.parquet")
df["date"] = pd.to_datetime(df["date"])

cfg = yaml.safe_load(open(ROOT / "src" / "config" / "settings.yaml"))
coords = pd.DataFrame(
    [(c, latlon[0], latlon[1]) for c, latlon in cfg["citypulse"]["cities"].items()],
    columns=["city", "lat", "lon"]
)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Filters")

all_dates = sorted(df["date"].unique())
default_date = all_dates[-1]
picked_date = st.sidebar.date_input(
    "Select date",
    value=default_date,
    min_value=all_dates[0],
    max_value=all_dates[-1],
    key="picked_date_map"
)

metric = st.sidebar.selectbox(
    "Metric",
    ["health_score", "air_score", "temp_score", "precip_score", "wind_score"],
    key="metric_map"
)

# ---------------------------
# Filter date + attach coords
# ---------------------------
day = df[df["date"]==pd.to_datetime(picked_date)].copy()
day = day.merge(coords, on="city", how="left")

# scale radius by metric
v = day[metric].fillna(0)
if v.max() != v.min():
    norm = (v - v.min()) / (v.max() - v.min())
else:
    norm = (v*0)
day["radius"] = (norm * 25000) + 12000

# color by health
h = day["health_score"].fillna(60)
day["color"] = list(zip((100-h)*2.55, h*2.55, 90 + 0*h))

# ---------------------------
# Map
# ---------------------------
view = pdk.ViewState(latitude=33.9, longitude=-6.3, zoom=5.8)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=day,
    get_position='[lon, lat]',
    get_radius='radius',
    get_fill_color='color',
    pickable=True,
    auto_highlight=True
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view,
    tooltip={"html": f"<b>{{city}}</b><br>health: {{health_score}}<br>{metric}: {{{metric}}}"}
)
st.pydeck_chart(deck)


# ---------------------------
# Professional legend card
# ---------------------------
with st.container():
    st.markdown(
        """
        <div style="
            padding:12px; 
            background-color:#F5F5F5; 
            border-radius:10px;
            border:1px solid #DDD;
            width:350px;
            margin-top:10px;
        ">
        <b>Legend</b><br>
        <ul style="padding-left:16px; margin-top:6px;">
          <li>Dot color → <b>Health Score</b> (Green = healthy, Red = stressed)</li>
          <li>Dot size → intensity of <b>{metric}</b></li>
          <li>Date shown → <b>{date}</b></li>
        </ul>
        </div>
        """.format(metric=metric, date=str(picked_date)),
        unsafe_allow_html=True
    )

# ---------------------------
# Table
# ---------------------------
st.subheader("Values")
st.dataframe(
    day[["city","health_score","air_score","temp_score","precip_score","wind_score"]]
    .sort_values("health_score", ascending=False),
    use_container_width=True
)

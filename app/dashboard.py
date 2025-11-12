import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import yaml
from pathlib import Path
from datetime import timedelta

# ===============================
# CONFIG
# ===============================
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

st.set_page_config(page_title="CityPulse Dashboard", layout="wide", page_icon="🌆")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_parquet(DATA / "citypulse_health_daily.parquet")
df["date"] = pd.to_datetime(df["date"])

# merge coordinates
cfg = yaml.safe_load(open(ROOT / "src" / "config" / "settings.yaml"))
coords = pd.DataFrame(
    [(c, latlon[0], latlon[1]) for c, latlon in cfg["citypulse"]["cities"].items()],
    columns=["city", "lat", "lon"]
)
df = df.merge(coords, on="city", how="left")

# events (optional)
events_path = DATA / "citypulse_event_explanations.parquet"
events = pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("🎛️ Filters")
picked_date = st.sidebar.date_input("Select date", value=df["date"].max())
selected_city = st.sidebar.selectbox("City", sorted(df["city"].unique()))
metric = st.sidebar.selectbox(
    "Metric (circle size)",
    ["health_score", "air_score", "temp_score", "precip_score", "wind_score"],
)
st.sidebar.markdown("---")
st.sidebar.info("CityPulse fuses air quality and meteorology into one interpretable city health index.")

# ===============================
# FILTERS
# ===============================
date = pd.to_datetime(picked_date)
day = df[df["date"] == date].copy()
city_df = df[df["city"] == selected_city].sort_values("date")

# normalize for radius scaling
v = day[metric].fillna(0)
norm = (v - v.min()) / (v.max() - v.min()) if v.max() != v.min() else v * 0
day["radius"] = (norm * 15000) + 8000

# color mapping (green = good, red = bad)
h = day["health_score"].fillna(60)
day["color"] = list(zip((100 - h) * 2.2, h * 2.0, 80 + 0 * h))

# ===============================
# LAYOUT
# ===============================
col1, col2, col3 = st.columns([1.3, 2.7, 1.5])

# 1️⃣ COLUMN — METRICS
with col1:
    st.markdown(f"### 📊 {selected_city} — Daily Metrics ({date.date()})")
    latest = city_df[city_df["date"] == date]
    if not latest.empty:
        r = latest.iloc[0]
        scores = {
            "Overall Health": r["health_score"],
            "Air Quality Score": r["air_score"],
            "Temperature Score": r["temp_score"],
            "Precipitation Score": r["precip_score"],
            "Wind Score": r["wind_score"],
        }
        for k, v in scores.items():
            color = "#2ecc71" if v > 70 else "#f1c40f" if v > 40 else "#e74c3c"
            st.markdown(
                f"""
                <div style='background-color:{color}22;padding:10px;border-radius:8px;
                            border-left:4px solid {color};margin-bottom:6px;'>
                <b>{k}</b><br><span style='font-size:22px;color:{color}'>{v:.1f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("No data for this date.")

# 2️⃣ COLUMN — MAP
with col2:
    st.markdown("### 🗺️ Urban Health Map")
    view = pdk.ViewState(latitude=33.9, longitude=-6.3, zoom=5.8)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=day,
        get_position='[lon, lat]',
        get_radius='radius',
        get_fill_color='color',
        get_line_color=[0, 0, 0],
        pickable=True,
        opacity=0.7
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="mapbox://styles/mapbox/light-v11",
        tooltip={"html": "<b>{city}</b><br>Health: {health_score:.1f}<br>" + metric + ": {" + metric + ":.1f}"}
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.markdown(
        f"""
        <div style="padding:10px;background:#f8f9fa;border-radius:8px;border:1px solid #ddd;">
        <b>Legend</b><br>
        • Color → <b>Health Score</b> (Green = healthy, Red = stressed)<br>
        • Size → intensity of <b>{metric}</b><br>
        • Base map: light for readability
        </div>
        """, unsafe_allow_html=True)

# 3️⃣ COLUMN — TREND & EVENTS
with col3:
    st.markdown("### 📈 Health Trend")
    smooth = city_df["health_score"].rolling(15, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_df["date"], y=smooth,
        mode="lines", line=dict(color="#2ecc71", width=3),
        name="Smoothed Health"
    ))
    fig.update_layout(
        template="plotly_white",
        height=260,
        margin=dict(l=0, r=0, t=25, b=0),
        yaxis_title="Health (0–100)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🩺 Event Insight")
    if not events.empty:
        ev = events[
            (events["city"] == selected_city) &
            (abs((pd.to_datetime(events["date"]) - date).dt.days) <= 3)
        ]
        if not ev.empty:
            e = ev.iloc[0]
            st.markdown(
                f"""
                **Event:** {e['event_type']} ({e['date'][:10]})  
                **Severity:** {e['severity']:.1f}  
                **Top Drivers:** {e['driver_1']}, {e['driver_2']}  

                <br>• Air = {e['air_score']:.1f}  
                • Temp = {e['temp_score']:.1f}  
                • Precip = {e['precip_score']:.1f}  
                • Wind = {e['wind_score']:.1f}
                """, unsafe_allow_html=True)
        else:
            st.info("No significant event detected near this date.")
    else:
        st.info("No event data available yet.")

# FOOTER
st.markdown("---")
st.caption("CityPulse © 2025 — Environmental Intelligence for Smarter Cities.")

import streamlit as st
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

df = pd.read_parquet(DATA / "citypulse_health_daily.parquet")
df["date"] = pd.to_datetime(df["date"])

events = pd.read_parquet(DATA / "citypulse_events.parquet")
events["date"] = pd.to_datetime(events["date"])

st.set_page_config(page_title="CityPulse", layout="wide")

st.title("CityPulse — Urban Wellbeing Observatory")

city = st.selectbox("Select City", sorted(df.city.unique()))
d = df[df.city==city].sort_values("date")
e = events[events.city==city]

st.subheader("Daily Health Score")
st.line_chart(d.set_index("date")["health_score"])

st.subheader("Recent Events")
st.dataframe(e.sort_values("date", ascending=False).head(20))

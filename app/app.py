import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  
from src.features.event_explainer_text import explain_event

DATA = ROOT / "data" / "processed"

df = pd.read_parquet(DATA / "citypulse_health_daily.parquet")
df["date"] = pd.to_datetime(df["date"])

events = pd.read_parquet(DATA / "citypulse_events.parquet")
events["date"] = pd.to_datetime(events["date"])

explanations = pd.read_parquet("data/processed/citypulse_event_explanations.parquet")
explanations["date"] = pd.to_datetime(explanations["date"])

st.set_page_config(page_title="CityPulse", layout="wide")

st.title("CityPulse — Urban Wellbeing Observatory")

# city = st.selectbox("Select City", sorted(df.city.unique()))

city = st.selectbox("City", sorted(df.city.unique()))

dfc = df[df.city==city].sort_values("date")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dfc["date"], y=dfc["health_score"],
    mode="lines+markers",
    marker=dict(size=6, color="black"),
    name="health"
))

clicked = plotly_events(fig, click_event=True)

st.plotly_chart(fig, use_container_width=True)

# event table
events_city = events[events.city==city].sort_values("date")
st.subheader("Events")
selected_row = st.dataframe(events_city)

# state placeholder
st.subheader("Explanation")

# logic:  X1 table click (via selection_box) OR X2 chart click
selected_date = None

# X2 - chart click
if clicked:
    selected_date = pd.to_datetime(clicked[0]["x"])

# X1 - table click (streamlit doesn’t give row selection natively)
# user selects date via dropdown directly under table
opt_date = st.selectbox("Select event date", events_city.date.tolist())
if opt_date:
    selected_date = pd.to_datetime(opt_date)

if selected_date is not None:
    row = explanations[(explanations.city==city) & (explanations.date==selected_date)]
    if not row.empty:
        txt = explain_event(row.iloc[0], df)
        st.text(txt)
    else:
        st.text("No explanation entry yet for this date.")
else:
    st.text("Click a point OR pick a date to see explanation")

# d = df[df.city==city].sort_values("date")
# e = events[events.city==city]

# st.subheader("Daily Health Score")
# st.line_chart(d.set_index("date")["health_score"])

# st.subheader("Recent Events")
# st.dataframe(e.sort_values("date", ascending=False).head(20))

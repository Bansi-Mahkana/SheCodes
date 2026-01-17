import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="City Health Risk Map",
    layout="wide"
)

# ---------------- CUSTOM MAP UI ----------------
st.markdown("""
<style>
.map-container {
    background: #ffffff;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.06);
}
.js-plotly-plot {
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "final.db")

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM health_data", conn)
conn.close()

# ---------------- DATA CLEANING ----------------
df.columns = df.columns.str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"])

# ---------------- CITY COORDINATES (SAFE) ----------------
city_coords = {
    "Ahmedabad": (23.0225, 72.5714),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Bangalore": (12.9716, 77.5946),  # 🔥 FIXED ERROR
    "Chennai": (13.0827, 80.2707)
}

df["lat"] = df["city"].map(lambda x: city_coords.get(x, (None, None))[0])
df["lon"] = df["city"].map(lambda x: city_coords.get(x, (None, None))[1])

df = df.dropna(subset=["lat", "lon"])

# ---------------- FEATURE ENGINEERING ----------------
df["resp_ratio"] = df["respiratory_cases"] / df["total_hospital_visits"]
df["heat_ratio"] = df["heat_related_cases"] / df["total_hospital_visits"]

df["health_risk_score"] = (
    0.6 * df["resp_ratio"] +
    0.4 * df["heat_ratio"]
)

def risk_label(score):
    if score < 0.05:
        return "Low"
    elif score < 0.15:
        return "Medium"
    else:
        return "High"

df["risk_level"] = df["health_risk_score"].apply(risk_label)

# ---------------- SIDEBAR FILTER ----------------
st.sidebar.header("🧭 Map Controls")

selected_city = st.sidebar.multiselect(
    "Select Cities",
    sorted(df["city"].unique()),
    default=sorted(df["city"].unique())
)

map_df = df[df["city"].isin(selected_city)]

# ---------------- PAGE TITLE ----------------
st.title("🗺️ City Health Risk Map")
st.markdown(
    "Interactive visualization of **urban public health risks** "
    "with responsive design and clean aesthetics."
)
st.markdown("---")

# ---------------- MAP CARD ----------------
st.markdown('<div class="map-container">', unsafe_allow_html=True)

fig = px.scatter_mapbox(
    map_df,
    lat="lat",
    lon="lon",
    size="health_risk_score",
    color="risk_level",
    size_max=42,
    zoom=4,
    hover_name="city",
    hover_data={
        "health_risk_score": ":.3f",
        "respiratory_cases": True,
        "heat_related_cases": True
    }
)

fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        center=dict(
            lat=map_df["lat"].mean(),
            lon=map_df["lon"].mean()
        ),
        zoom=4
    ),
    height=580,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(
        title="Risk Level",
        bgcolor="rgba(255,255,255,0.85)",
        borderwidth=0
    )
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INSIGHT STRIP ----------------
st.caption(
    "🔴 High risk indicates immediate intervention needed • "
    "🟡 Medium risk suggests caution • "
    "🟢 Low risk reflects stable conditions"
)

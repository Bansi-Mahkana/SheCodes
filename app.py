import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Public Health Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #f5f7fa;
    border-radius: 10px;
    padding: 14px;
    border-left: 6px solid #4c78a8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "final.db")

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM health_data", conn)
conn.close()

# ---------------- DATA CLEANING ----------------
df.columns = df.columns.str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"])

# ---------------- CITY COORDINATES ----------------
city_coords = {
    "Ahmedabad": (23.0225, 72.5714),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707)
}

df["lat"] = df["city"].map(lambda x: city_coords.get(x, (None, None))[0])
df["lon"] = df["city"].map(lambda x: city_coords.get(x, (None, None))[1])

# ---------------- FEATURE ENGINEERING ----------------
df["respiratory_ratio"] = df["respiratory_cases"] / df["total_hospital_visits"]
df["heat_ratio"] = df["heat_related_cases"] / df["total_hospital_visits"]

df["health_risk_score"] = (
    0.6 * df["respiratory_ratio"] +
    0.4 * df["heat_ratio"]
)

def risk_level(score):
    if score < 0.05:
        return "Low"
    elif score < 0.15:
        return "Medium"
    else:
        return "High"

df["risk_level"] = df["health_risk_score"].apply(risk_level)

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 Filters")

city = st.sidebar.selectbox(
    "Select City",
    sorted(df["city"].unique())
)
# Default city comes from sidebar
top_city = city

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["date"].min(), df["date"].max()]
)

filtered_df = df[
    (df["city"] == top_city) &
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1]))
].copy()


# ---------------- TITLE ----------------


st.title("🌍 Public Health Intelligence Dashboard")
st.markdown(
    "Data-driven insights for **urban health risk assessment** and **policy decision-making**."
)

# 🔍 City selector at top (UX improvement)
top_city = st.selectbox(
    "🔎 Search & Select City",
    sorted(df["city"].unique()),
    index=sorted(df["city"].unique()).index(city)
)

st.markdown("---")



# ---------------- KPI METRICS ----------------
st.subheader("📌 City Health Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("🏥 Avg Hospital Visits", int(filtered_df["total_hospital_visits"].mean()))
k2.metric("🌬 Avg Respiratory Cases", int(filtered_df["respiratory_cases"].mean()))
k3.metric("🔥 Avg Heat Cases", int(filtered_df["heat_related_cases"].mean()))
k4.metric("⚠️ Risk Score", round(filtered_df["health_risk_score"].mean(), 3))

# ---------------- RISK STATUS ----------------
st.markdown("### ⚠️ Current Risk Status")

risk_mode = filtered_df["risk_level"].mode()[0]

if risk_mode == "Low":
    st.success("🟢 LOW RISK — Conditions are stable")
elif risk_mode == "Medium":
    st.warning("🟡 MEDIUM RISK — Monitoring recommended")
else:
    st.error("🔴 HIGH RISK — Immediate intervention required")

st.markdown("---")

# ---------------- MAP ----------------
st.subheader("🗺️ City Health Risk Map")

map_df = filtered_df.dropna(subset=["lat", "lon"])

fig_map = px.scatter_mapbox(
    map_df,
    lat="lat",
    lon="lon",
    size="health_risk_score",
    color="risk_level",
    hover_name="city",
    hover_data={
        "health_risk_score": True,
        "respiratory_cases": True,
        "heat_related_cases": True
    },
    zoom=6,
    height=520
)

fig_map.update_layout(
    mapbox_style="carto-positron",
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(bgcolor="rgba(255,255,255,0.8)")
)

st.plotly_chart(fig_map, use_container_width=True)

st.caption("💡 Higher bubble size and darker colors indicate elevated public health risk.")

st.markdown("---")

# ---------------- VISUAL GRID ----------------
st.subheader("📊 Health Analytics")

col1, col2 = st.columns(2)

# --- PIE: Risk Distribution ---
risk_pie = filtered_df["risk_level"].value_counts().reset_index()
risk_pie.columns = ["Risk Level", "Days"]

fig_pie = px.pie(
    risk_pie,
    names="Risk Level",
    values="Days",
    hole=0.4,
    title="Health Risk Distribution"
)

col1.plotly_chart(fig_pie, use_container_width=True)

# --- LINE: Trends ---
fig_trend = px.line(
    filtered_df,
    x="date",
    y=["respiratory_cases", "heat_related_cases"],
    title="Disease Trend Over Time"
)

col2.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ---------------- BAR: DAILY LOAD ----------------
st.subheader("🏥 Daily Hospital Load")

fig_bar = px.bar(
    filtered_df,
    x="date",
    y=["respiratory_cases", "heat_related_cases"],
    title="Hospital Case Composition",
    labels={"value": "Cases", "variable": "Case Type"}
)

st.plotly_chart(fig_bar, use_container_width=True)

st.caption("💡 Respiratory cases contribute more consistently to hospital load.")

st.markdown("---")

# ---------------- WHAT-IF SCENARIO ----------------
with st.expander("🔮 What-If Scenario: Heatwave Impact", expanded=True):

    increase = st.slider(
        "Increase Heat-Related Cases (%)",
        0, 50, 10
    )

    filtered_df["simulated_heat"] = (
        filtered_df["heat_related_cases"] * (1 + increase / 100)
    )

    fig_sim = px.line(
        filtered_df,
        x="date",
        y=["heat_related_cases", "simulated_heat"],
        title="Simulated Heatwave Scenario"
    )

    st.plotly_chart(fig_sim, use_container_width=True)

    st.caption(
        "Simulation highlights potential surge in healthcare burden during extreme heat events."
    )

# ---------------- ROLLING AVERAGE ----------------
st.markdown("---")
st.subheader("📉 Early Warning Indicator")

filtered_df = filtered_df.sort_values("date")
filtered_df["resp_7d_avg"] = filtered_df["respiratory_cases"].rolling(7).mean()

fig_roll = px.line(
    filtered_df,
    x="date",
    y="resp_7d_avg",
    title="7-Day Rolling Average of Respiratory Cases"
)

st.plotly_chart(fig_roll, use_container_width=True)

# ---------------- HEATMAP ----------------
st.markdown("---")
st.subheader("🔥 Weekly Risk Intensity Heatmap")

filtered_df["week"] = filtered_df["date"].dt.isocalendar().week

heatmap_df = (
    filtered_df
    .groupby(["week", "risk_level"])
    .size()
    .reset_index(name="days")
)

fig_heat = px.density_heatmap(
    heatmap_df,
    x="week",
    y="risk_level",
    z="days",
    title="Weekly Health Risk Intensity"
)

st.plotly_chart(fig_heat, use_container_width=True)

# ---------------- FOOTER ----------------
st.sidebar.markdown("---")
st.sidebar.caption("⏱ Live Simulation")
st.sidebar.write(pd.Timestamp.now())
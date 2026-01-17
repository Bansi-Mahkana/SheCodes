import streamlit as st
import sqlite3
import pandas as pd
import os

st.title("🧩 City Data Explorer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "final.db")

df = pd.read_sql("SELECT * FROM health_data", sqlite3.connect(DB_PATH))
df.columns = df.columns.str.lower()

dataset = st.radio(
    "Choose Dataset",
    ["🏥 Health", "🚦 Traffic", "🌫️ Air Quality"]
)

city = st.selectbox("Select City", sorted(df["city"].unique()))

city_df = df[df["city"] == city]

if dataset == "🏥 Health":
    st.dataframe(city_df[
        ["date", "respiratory_cases", "heat_related_cases"]
    ])

elif dataset == "🚦 Traffic":
    st.info("Traffic data integration ready")

elif dataset == "🌫️ Air Quality":
    st.info("Air quality dataset integration ready")

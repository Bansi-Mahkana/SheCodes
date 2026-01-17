import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

@st.cache_data
def load_data():
    weather = pd.read_csv("weather_data.csv")
    health = pd.read_csv("health_data.csv")
    traffic = pd.read_csv("traffic_data.csv")

    df = weather.merge(traffic, on=["date", "city"])
    df = df.merge(health, on=["date", "city"])
    return df

@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

df = load_data()
model = load_model()

st.set_page_config(
    page_title="Urban Health Intelligence",
    layout="wide"
)

st.title("🌆 Urban Health Intelligence Dashboard")
st.markdown("Predicting health risk using **Weather, Air Quality & Traffic Data**")

menu = st.sidebar.radio(
    "Navigate",
    ["Overview", "Insights", "ML Prediction", "Model Explainability"]
)

if menu == "Overview":
    st.subheader("📌 Dataset Snapshot")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg AQI", round(df["aqi"].mean(), 2))
    col2.metric("Avg Hospital Visits", round(df["total_hospital_visits"].mean(), 2))
    col3.metric("Avg Vehicle Count", round(df["vehicle_count"].mean(), 2))

if menu == "Insights":
    st.subheader("🌫 Air Quality vs Respiratory Cases")

    fig = px.scatter(
        df,
        x="aqi",
        y="respiratory_cases",
        color="city",
        title="AQI Impact on Respiratory Health"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        df,
        x="vehicle_count",
        y="total_hospital_visits",
        color="congestion_level",
        title="Traffic Congestion vs Hospital Visits"
    )
    st.plotly_chart(fig2, use_container_width=True)

if menu == "ML Prediction":
    st.subheader("🔮 Predict Urban Health Risk")

    col1, col2 = st.columns(2)

    with col1:
        temp = st.slider("Temperature (°C)", 10, 45, 30)
        humidity = st.slider("Humidity (%)", 20, 100, 60)
        rainfall = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
        aqi = st.slider("AQI", 50, 300, 150)

    with col2:
        heat_index = st.slider("Heat Index", 20, 60, 35)
        vehicle_count = st.slider("Vehicle Count", 50, 500, 200)
        avg_speed = st.slider("Avg Speed (km/h)", 5, 60, 30)
        congestion = st.selectbox("Congestion Level", ["Low", "Medium", "High"])

    congestion_map = {"Low": 0, "Medium": 1, "High": 2}

    input_data = [[
        temp,
        humidity,
        rainfall,
        aqi,
        heat_index,
        2,  # air quality category placeholder
        0, 0, 0,  # health placeholders
        vehicle_count,
        avg_speed,
        congestion_map[congestion],
        aqi * heat_index,
        vehicle_count / (avg_speed + 1),
        temp * humidity
    ]]

    if st.button("Predict Health Risk"):
        prediction = model.predict(input_data)
        risk = ["Low", "Medium", "High"][prediction[0]]

        st.success(f"🏥 Predicted Health Risk Level: **{risk}**")

if menu == "Model Explainability":
    st.subheader("🧠 Model Feature Importance")

    importances = model.feature_importances_
    features = df.drop(["date", "city"], axis=1).columns

    fig = px.bar(
        x=importances,
        y=features,
        orientation="h",
        title="Factors Influencing Health Risk"
    )
    st.plotly_chart(fig, use_container_width=True)

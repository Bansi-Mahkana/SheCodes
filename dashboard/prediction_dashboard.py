def show_health_dashboard(df, health_model):
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression

    def forecast_next_7_days(series):
        series = series.dropna().reset_index(drop=True)
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values

        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(series), len(series) + 7).reshape(-1, 1)
        return model.predict(future_X)

    def heat_index(temp, humidity):
        return (
            -8.784695 +
            1.61139411 * temp +
            2.338549 * humidity -
            0.14611605 * temp * humidity
        )

    # ---------------- Sidebar ----------------
    st.sidebar.title("Controls")
    city = st.sidebar.selectbox("Select City", df['city'].unique())

    city_df = df[df['city'] == city].sort_values('date')

    # ---------------- Forecast ----------------
    aqi_f = forecast_next_7_days(city_df['aqi'])
    temp_f = forecast_next_7_days(city_df['temperature'])
    humidity_f = forecast_next_7_days(city_df['humidity'])
    rainfall_f = forecast_next_7_days(city_df['rainfall'])

    heat_index_f = heat_index(temp_f, humidity_f)

    vehicle_f = np.repeat(city_df['vehicle_count'].rolling(7).mean().iloc[-1], 7)
    speed_f = np.repeat(city_df['avg_speed'].rolling(7).mean().iloc[-1], 7)
    resp_f = np.repeat(city_df['respiratory_cases'].rolling(7).mean().iloc[-1], 7)
    heat_cases_f = np.repeat(city_df['heat_related_cases'].rolling(7).mean().iloc[-1], 7)

    future_input = pd.DataFrame({
        'aqi': aqi_f,
        'temperature': temp_f,
        'humidity': humidity_f,
        'rainfall': rainfall_f,
        'heat_index': heat_index_f
    })
    FEATURES = list(health_model.feature_names_in_)
    future_input = future_input[FEATURES]

    health_score_f = health_model.predict(future_input)

    future_dates = pd.date_range(
        start=city_df['date'].max() + pd.Timedelta(days=1),
        periods=7
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "AQI": aqi_f,
        "Temperature": temp_f,
        "Rainfall": rainfall_f,
        "Health Risk Score": health_score_f
    })

    def risk_level(score):
        if score < 0.4:
            return "Low"
        elif score <= 0.6:
            return "Medium"
        return "High"

    forecast_df["Risk Level"] = forecast_df["Health Risk Score"].apply(risk_level)

    # ---------------- UI ----------------
    st.title("Smart City Health Risk Forecast")
    st.caption(f"7-Day Forecast for {city}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Max AQI", int(forecast_df["AQI"].max()))
    col2.metric("Avg Health Risk", round(forecast_df["Health Risk Score"].mean(), 1))
    col3.metric("High Risk Days", (forecast_df["Risk Level"] == "High").sum())

    st.subheader("Forecast Trends")
    st.line_chart(forecast_df.set_index("Date")[["AQI", "Health Risk Score"]])
    st.line_chart(forecast_df.set_index("Date")[["Temperature", "Rainfall"]])

    st.subheader("Health Alerts")
    high_risk = forecast_df[forecast_df["Risk Level"] == "High"]

    if not high_risk.empty:
        st.error("⚠️ High health risk predicted")
        st.dataframe(high_risk)
    else:
        st.success("✅ No high-risk days predicted")

    st.subheader("Detailed Forecast")
    st.dataframe(forecast_df)

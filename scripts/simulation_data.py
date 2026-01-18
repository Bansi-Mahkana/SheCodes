import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# -----------------------
# CONFIG
# -----------------------
CITIES = {
    "Delhi": {"temp": (10, 42), "aqi_base": 160, "traffic": (200, 600)},
    "Bangalore": {"temp": (16, 34), "aqi_base": 90, "traffic": (150, 450)},
    "Mumbai": {"temp": (22, 38), "aqi_base": 120, "traffic": (250, 650)},
    "Chennai": {"temp": (24, 42), "aqi_base": 130, "traffic": (200, 550)},
    "Ahmedabad": {"temp": (18, 45), "aqi_base": 140, "traffic": (180, 500)},
    "Hyderabad": {"temp": (20, 42), "aqi_base": 110, "traffic": (180, 480)},
    "Lucknow": {"temp": (12, 40), "aqi_base": 150, "traffic": (150, 420)},
    "Shimla": {"temp": (2, 28), "aqi_base": 60, "traffic": (50, 200)},
}

DAYS = 365
START_DATE = datetime.today() - timedelta(days=DAYS)

OUTPUT_DIR = "Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def congestion_level(vehicle_count):
    if vehicle_count < 150:
        return "Low"
    elif vehicle_count < 350:
        return "Medium"
    else:
        return "High"

def air_quality_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    else:
        return "Poor"

def health_risk_score(aqi, temp, congestion):
    congestion_score = {"Low": 0.2, "Medium": 0.5, "High": 0.8}[congestion]
    score = (
        0.4 * min(aqi / 300, 1) +
        0.3 * min(temp / 45, 1) +
        0.3 * congestion_score
    )
    return round(min(score, 1.0), 2)

# -----------------------
# DATA CONTAINERS
# -----------------------
traffic_data = []
weather_data = []
health_data = []
derived_data = []

# -----------------------
# SIMULATION LOOP
# -----------------------
for i in range(DAYS):
    date = (START_DATE + timedelta(days=i)).date()

    for city, cfg in CITIES.items():

        # ----- Traffic -----
        vehicle_count = random.randint(*cfg["traffic"])
        avg_speed = max(10, 65 - vehicle_count // 9)
        congestion = congestion_level(vehicle_count)

        # ----- Weather -----
        temperature = round(random.uniform(*cfg["temp"]), 1)
        humidity = random.randint(30, 85)
        rainfall = round(random.choice([0, 0, 0, random.uniform(0, 20)]), 1)

        # AQI influenced by traffic
        aqi = int(cfg["aqi_base"] + vehicle_count * 0.25 + random.randint(-20, 20))
        aqi = min(max(aqi, 30), 350)

        heat_index = round(temperature + humidity * 0.04, 1)
        aqi_category = air_quality_category(aqi)

        # ----- Health -----
        respiratory_cases = int(aqi * random.uniform(0.25, 0.5))
        heat_cases = int(max(0, temperature - 30) * random.uniform(1.5, 4))
        hospital_visits = respiratory_cases + heat_cases + random.randint(20, 70)

        # ----- Risk -----
        risk_score = health_risk_score(aqi, temperature, congestion)
        risk_level = "Low" if risk_score < 0.4 else "Medium" if risk_score < 0.7 else "High"

        # ----- Append -----
        traffic_data.append([date, city, vehicle_count, avg_speed, congestion])
        weather_data.append([date, city, temperature, humidity, rainfall, aqi, heat_index, aqi_category])
        health_data.append([date, city, hospital_visits, respiratory_cases, heat_cases])
        derived_data.append([
            date, city, vehicle_count, avg_speed, congestion,
            temperature, heat_index, humidity, rainfall,
            aqi, aqi_category,
            respiratory_cases, heat_cases,
            risk_score, risk_level
        ])

# -----------------------
# CREATE DATAFRAMES
# -----------------------
traffic_df = pd.DataFrame(
    traffic_data,
    columns=["date", "city", "vehicle_count", "avg_speed", "congestion_level"]
)

weather_df = pd.DataFrame(
    weather_data,
    columns=["date", "city", "temperature", "humidity", "rainfall", "aqi", "heat_index", "air_quality_category"]
)

health_df = pd.DataFrame(
    health_data,
    columns=["date", "city", "total_hospital_visits", "respiratory_cases", "heat_related_cases"]
)

derived_df = pd.DataFrame(
    derived_data,
    columns=[
        "date", "city", "vehicle_count", "avg_speed", "congestion_level",
        "temperature", "heat_index", "humidity", "rainfall",
        "aqi", "air_quality_category",
        "respiratory_cases", "heat_related_cases",
        "health_risk_score", "risk_level"
    ]
)

# -----------------------
# SAVE
# -----------------------
traffic_df.to_csv(f"{OUTPUT_DIR}/traffic_data.csv", index=False)
weather_df.to_csv(f"{OUTPUT_DIR}/weather_data.csv", index=False)
health_df.to_csv(f"{OUTPUT_DIR}/health_data.csv", index=False)
derived_df.to_csv(f"{OUTPUT_DIR}/derived_data.csv", index=False)

print("✅ 1 year realistic data generated for 8 Indian cities")
print("📁 Files saved in /data folder")

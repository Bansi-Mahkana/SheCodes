# CityBAND

This project predicts **city-wise health risk** by combining traffic, weather, air quality, and health data.  
It helps provide **early warnings** by forecasting health risks for the **next 7 days** using machine learning.

### [Video-link](https://drive.google.com/file/d/1OUeHXLqWm3DOqsWAWNyf7shnR3HI7pd6/view?usp=sharing)
</br>

## Features
- City-wise data analysis
- 7-day forecasting of AQI, temperature, humidity, and rainfall
- Machine learning–based health risk score prediction
- Risk classification: Low / Medium / High
- Interactive Streamlit dashboard

## Dataset Used (Simulated)
- Traffic data (vehicle count, speed, congestion)
- Weather data (temperature, humidity, rainfall, AQI)
- Health data (hospital visits, respiratory and heat cases)
- Derived dataset combining all sources

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Pickle

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py


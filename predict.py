
import pandas as pd
import joblib

temp_model = joblib.load("weather_model.pkl")
rain_model = joblib.load("rain_model.pkl")
rain_threshold = joblib.load("rain_threshold.pkl")

def predict_weather(today):

    temp_features = [
        "temperature_2m_mean (°C)",
        "temperature_2m_min (°C)",
        "temperature_2m_max (°C)",
        "precipitation_sum (mm)",
        "wind_speed_10m_max (km/h)"
    ]

    X_temp = pd.DataFrame([{
        feature: today[feature]
        for feature in temp_features
    }])

    temp_prediction = temp_model.predict(X_temp)[0]

    precipitation = today["precipitation_sum (mm)"]
    wind_speed = today["wind_speed_10m_max (km/h)"]

    date = pd.to_datetime(today["time"])

    month = date.month
    day_of_year = date.dayofyear

    temp_range = (
        today["temperature_2m_max (°C)"]
        - today["temperature_2m_min (°C)"]
    )

    rain_sum = precipitation

    precipitation_hours = (0 if precipitation == 0 else 1)

    if precipitation == 0:
        weather_code = 0
    elif precipitation < 5:
        weather_code = 61
    elif precipitation < 15:
        weather_code = 63
    else:
        weather_code = 65

    wind_gusts = wind_speed * 1.2
    wind_direction = 0

    X_rain = pd.DataFrame([{
        "temperature_2m_mean (°C)": today["temperature_2m_mean (°C)"],
        "temperature_2m_max (°C)": today["temperature_2m_max (°C)"],
        "temperature_2m_min (°C)": today["temperature_2m_min (°C)"],
        "precipitation_sum (mm)": precipitation,
        "rain_sum (mm)": rain_sum,
        "precipitation_hours (h)": precipitation_hours,
        "weather_code (wmo code)": weather_code,
        "wind_speed_10m_max (km/h)": wind_speed,
        "wind_gusts_10m_max (km/h)": wind_gusts,
        "wind_direction_10m_dominant (°)": wind_direction,
        "month": month,
        "day_of_year": day_of_year,
        "temp_range": temp_range
    }])

    rain_probability = rain_model.predict_proba(X_rain)[0][1]

    rain_prediction = rain_probability >= rain_threshold

    return {
        "temperature": temp_prediction,
        "rain_probability": rain_probability,
        "rain": rain_prediction
    }

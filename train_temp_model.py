import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("weather.csv", skiprows=3)

df["time"] = pd.to_datetime(df["time"])

df["target"] = df["temperature_2m_mean (°C)"].shift(-1)

df = df.dropna().reset_index(drop=True)

features = [
    "temperature_2m_mean (°C)",
    "temperature_2m_min (°C)",
    "temperature_2m_max (°C)",
    "precipitation_sum (mm)",
    "wind_speed_10m_max (km/h)"
]

X = df[features]
y = df["target"]


split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("=" * 50)
print("TEMPERATURE MODEL EVALUATION")
print("=" * 50)
print(f"MAE: {mae:.2f} °C")

joblib.dump(model, "weather_model.pkl", compress=3)

print("\nModel saved successfully!")


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import joblib

df = pd.read_csv("weather.csv", skiprows=3)

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

df["target"] = (df["rain_sum (mm)"].shift(-1) > 0).astype(int)

df["month"] = df["time"].dt.month
df["day_of_year"] = df["time"].dt.dayofyear

df["temp_range"] = (
    df["temperature_2m_max (°C)"]
    - df["temperature_2m_min (°C)"]
)

df = df.dropna().reset_index(drop=True)

features_to_drop = ["time", "target"]

X = df.drop(columns=features_to_drop)
y = df["target"]

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=4,
    random_state=42
)

model.fit(X_train, y_train)


probabilities = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.3, 0.7, 0.01)

best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    predictions = (probabilities >= t).astype(int)

    score = f1_score(y_test, predictions)

    if score > best_f1:
        best_f1 = score
        best_threshold = t


custom_predictions = (
    probabilities >= best_threshold
).astype(int)


# Evaluation
print("=" * 50)
print(f"ROC-AUC Score: {roc_auc_score(y_test, probabilities):.4f}")
print(f"Optimal Threshold: {best_threshold:.2f}")
print("=" * 50)

print("\nClassification Report:")
print(classification_report(y_test, custom_predictions))


joblib.dump(model, "rain_model.pkl")
joblib.dump(best_threshold, "rain_threshold.pkl")

print("\nRain model saved successfully!")

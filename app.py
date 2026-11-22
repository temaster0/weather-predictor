from datetime import date
from flask import Flask, render_template, request
from predict import predict_weather

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None
    if request.method == "POST":
        try:
            # Get input values
            mean_temp = float(request.form["mean_temp"])
            min_temp = float(request.form["min_temp"])
            max_temp = float(request.form["max_temp"])
            precipitation = float(request.form["precipitation"])
            wind_speed = float(request.form["wind_speed"])

            if not min_temp <= mean_temp <= max_temp:
                raise ValueError(
                    "Temperature values must satisfy: min <= mean <= max"
                )

            if precipitation < 0 or wind_speed < 0:
                raise ValueError(
                    "Precipitation and wind speed cannot be negative"
                )

            today = {
                "time": date.today(),
                "temperature_2m_mean (°C)": mean_temp,
                "temperature_2m_min (°C)": min_temp,
                "temperature_2m_max (°C)": max_temp,
                "precipitation_sum (mm)": precipitation,
                "wind_speed_10m_max (km/h)": wind_speed
            }

            prediction = predict_weather(today)

            result = {
                "temperature": round(float(prediction["temperature"]), 2),
                "rain_probability": round(
                    float(prediction["rain_probability"]) * 100, 2
                ),
                "rain": bool(prediction["rain"])
            }

        except ValueError as e:
            error = str(e)

        except Exception:
            error = "An error occurred while making the prediction."

    return render_template(
        "index.html",
        result=result,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)